import os
import io
import requests
import docx
import pandas as pd
import hashlib
import logging
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm
from dotenv import load_dotenv
from db import init_db


logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

load_dotenv()

ELASTIC_URL = os.getenv("ELASTIC_URL_LOCAL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
INDEX_NAME = os.getenv("INDEX_NAME")

BASE_URL_TEMPLATE = "https://docs.google.com/document/d/{file_id}/export?format=docx"

OLX_DOCUMENTS = {
    "olx-qa": "18i5tEWeNp3lVB0uWmNWPUyXKVwPS1KZByr-lDg-UkK0",
    "olx-tc": "1i-RS_ZeatwnqmJ1sVNJjQoJiqT1yIy2VoA-gcjfKv1g",
}


def generate_document_id(doc: dict) -> str:
    combined = f"{doc['question']}-{doc['text'][:10]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:8]
    return document_id


def fetch_documents() -> list:
    logger.info("Fetching documents...")
    documents = []
    for file_id in OLX_DOCUMENTS.values():
        questions = read_faq(file_id)
        logger.info(f"Fetched {len(questions)} questions from document with ID: {file_id}")
        documents.extend(questions)
    for doc in documents:
        doc["id"] = generate_document_id(doc)
    logger.info(f"Fetched {len(documents)} documents")
    return documents


def clean_line(line: str) -> str:
    line = line.strip()
    line = line.strip("\uFEFF")
    return line


def read_faq(file_id: str) -> list:
    url = BASE_URL_TEMPLATE.format(file_id=file_id)

    response = requests.get(url)
    response.raise_for_status()

    with io.BytesIO(response.content) as f_in:
        doc = docx.Document(f_in)

    questions = []

    question_heading_style = "heading 2"
    section_heading_style = "heading 1"

    heading_id = ""
    section_title = ""
    question_title = ""
    answer_text_so_far = ""

    for p in doc.paragraphs:
        style = p.style.name.lower()
        p_text = clean_line(p.text)

        if len(p_text) == 0:
            continue

        if style == section_heading_style:
            section_title = p_text
            continue

        if style == question_heading_style:
            answer_text_so_far = answer_text_so_far.strip()
            if (
                answer_text_so_far != ""
                and section_title != ""
                and question_title != ""
            ):
                questions.append(
                    {
                        "text": answer_text_so_far,
                        "section": section_title,
                        "question": question_title,
                    }
                )
                answer_text_so_far = ""

            question_title = p_text
            continue

        answer_text_so_far += "\n" + p_text

    answer_text_so_far = answer_text_so_far.strip()
    if answer_text_so_far != "" and section_title != "" and question_title != "":
        questions.append(
            {
                "text": answer_text_so_far,
                "section": section_title,
                "question": question_title,
            }
        )

    return questions


def setup_elasticsearch() -> Elasticsearch:
    logger.info("Setting up Elasticsearch...")
    es_client = Elasticsearch(ELASTIC_URL)

    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "id": {"type": "keyword"},
                "question_text_vector": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }
    es_client.indices.delete(index=INDEX_NAME, ignore_unavailable=True)
    es_client.indices.create(index=INDEX_NAME, body=index_settings)
    logger.info(f"Elasticsearch index '{INDEX_NAME}' created")
    
    logs_index_name = 'logs-index'
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "@timestamp": {
                    "type": "date"
                },
                "level": {
                    "type": "keyword"
                },
                "message": {
                    "type": "text"
                }
            }
        }
    }

    if not es_client.indices.exists(index=logs_index_name):
        es_client.indices.create(index=logs_index_name, body=index_settings)
        print(f'Index "{index_name}" created successfully.')
    else:
        print(f'Index "{index_name}" already exists.')

    return es_client


def index_documents(es_client: Elasticsearch,
                    documents: list) -> None:
    logger.info("Indexing documents...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    for doc in tqdm(documents):
        question = doc["question"]
        text = doc["text"]
        doc["question_text_vector"] = model.encode(question + " " + text).tolist()
        es_client.index(index=INDEX_NAME, document=doc)
    logger.info(f"Indexed {len(documents)} documents")


def main():
    logger.info("Starting the indexing process...")
    documents = fetch_documents()
    es_client = setup_elasticsearch()
    index_documents(es_client, documents)
    # if new documents are added to the Google Docs,
    # run the fetch_documents and index_documents functions again
    # consider changing INDEX_NAME in the .env file if needed

    logger.info("Initializing database...")
    os.environ["POSTGRES_HOST"] = "localhost"
    init_db()
    # if the above line fails, consider changing postgres to localhost 
    # in get_db_connection function in the db.py file

    logger.info("Indexing process completed successfully!")


if __name__ == "__main__":
    main()
