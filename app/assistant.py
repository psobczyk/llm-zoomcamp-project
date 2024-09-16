import os
import time
import json
import re

from openai import OpenAI

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from prompt_templates import (
    assistant_prompt_template,
    reranking_prompt_template,
    retrieval_eval_prompt_template,
)

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")
ELASTIC_INDEX_NAME = os.getenv("INDEX_NAME", "documents_olx_20240907")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "xlm-r-distilroberta-base-paraphrase-v1"
)
NUMBER_OF_DOCUMENTS_FOR_RERANKING = (
    15  # needs to be fixed, because it is used in the prompt reranking_prompt_template
)

es_client = Elasticsearch(ELASTIC_URL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

model_name = EMBEDDING_MODEL_NAME
model = SentenceTransformer(model_name)


def search_reranked_retrieval(query: str, 
                              query_vector: list, k: int=5, 
                              max_doc_candidate_length:int =200):
    """
    Search reranked retrieval using GPT-4o

    :param query: question
    :param k: number of documents to return
    :param max_doc_candidate_length: maximum length of the document candidate. We use it to use less tokens in the prompt
    """

    doc_id = q["document"]
    doc = [doc for doc in documents if doc["id"] == doc_id][0]
    found_docs = question_hybrid(query_vector, k=NUMBER_OF_DOCUMENTS_FOR_RERANKING)
    found_docs_questions = [
        "id: " + doc["id"] + ". " + doc["question"] + " " + doc["text"]
        for doc in found_docs
    ]

    prompt = reranking_prompt_template.format(
        k=k,
        question=query,
        doc1_question=found_docs_questions[0][:max_doc_candidate_length],
        doc2_question=found_docs_questions[1][:max_doc_candidate_length],
        doc3_question=found_docs_questions[2][:max_doc_candidate_length],
        doc4_question=found_docs_questions[3][:max_doc_candidate_length],
        doc5_question=found_docs_questions[4][:max_doc_candidate_length],
        doc6_question=found_docs_questions[5][:max_doc_candidate_length],
        doc7_question=found_docs_questions[6][:max_doc_candidate_length],
        doc8_question=found_docs_questions[7][:max_doc_candidate_length],
        doc9_question=found_docs_questions[8][:max_doc_candidate_length],
        doc10_question=found_docs_questions[9][:max_doc_candidate_length],
        doc11_question=found_docs_questions[10][:max_doc_candidate_length],
        doc12_question=found_docs_questions[11][:max_doc_candidate_length],
        doc13_question=found_docs_questions[12][:max_doc_candidate_length],
        doc14_question=found_docs_questions[13][:max_doc_candidate_length],
        doc15_question=found_docs_questions[14][:max_doc_candidate_length],
    )

    response = client.chat.completions.create(
        model=RAG_MODEL_NAME, messages=[{"role": "user", "content": prompt}]
    )

    json_response = response.choices[0].message.content

    if len(json_response) == 0:
        logger.info("Empty response from GPT-4o")
        return [doc["id"] for doc in found_docs[:5]]

    # quick preprocessing
    json_response_preprocessed = re.sub(r"```json", "", json_response)
    json_response_preprocessed = re.sub(r"```", "", json_response_preprocessed)
    ids = json.loads(json_response_preprocessed)
    ids = [re.sub("id_", "", str(id)) for id in ids]
    ids = [re.sub("id:", "", str(id)) for id in ids]
    ids = [id.strip() for id in ids]
    ids = [id for id in ids if len(id) > 2]

    if len(ids) > k:
        logger.warning(f"More than {k} ids returned from GPT-4o")
        ids = ids[:k]
    elif len(ids) < k:
        logger.warning(f"Less than {k} ids returned from GPT-4o after preprocessing")
        logger.warning(json_response)
        tmp_ids = ids + [doc["id"] for doc in found_docs if doc["id"] not in ids]
        logger.info(f"Returning {tmp_ids[:k]}")
        ids = tmp_ids[:k]

    # select found_docs based on ids
    final_docs = [doc for doc in found_docs if doc["id"] in ids]
    return final_docs


def question_hybrid(query: str, query_vector: list, k: int = 5):
    return elastic_search_hybrid(query, query_vector, field="question_text_vector", k=k)


def build_prompt(query: str, search_results):
    context = ""

    for doc in search_results:
        context = (
            context
            + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
        )

    prompt = assistant_prompt_template.format(question=query, context=context).strip()
    return prompt


def llm(prompt: str, temperature: float = 0):
    start_time = time.time()
    response = openai_client.chat.completions.create(
        model=RAG_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    answer = response.choices[0].message.content
    tokens = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    end_time = time.time()
    response_time = end_time - start_time

    return answer, tokens, response_time


def evaluate_relevance(question: str, answer: str):
    prompt = retrieval_eval_prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _ = llm(prompt)

    try:
        json_eval = json.loads(evaluation)
        return json_eval["Relevance"], json_eval["Explanation"], tokens
    except json.JSONDecodeError:
        return "UNKNOWN", "Failed to parse evaluation", tokens


def calculate_openai_cost(model_choice, tokens):
    openai_cost = (
        tokens["prompt_tokens"] * 0.03 + tokens["completion_tokens"] * 0.06
    ) / 1000

    return openai_cost


def elastic_search_hybrid(
    query: str,
    vector: list,
    index_name: str = ELASTIC_INDEX_NAME,
    field: str = "question_text_vector",
    section_to_exclude: str = "Definicje",
    k: int = 5,
    size: int = 10,
):

    # Exclusion filter. The definitions section is excluded from the search
    exclusion_filter = {
        "term": {"section": section_to_exclude}  # Term query for exclusion
    }

    knn_query = {
        "field": field,
        "query_vector": vector,
        "k": k,
        "num_candidates": 10000,
        "boost": 0.5,
        "filter": {"bool": {"must_not": [exclusion_filter]}},
    }

    keyword_query = {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question", "text", "section"],
                    "type": "best_fields",
                    "boost": 0.5,
                }
            },
            "filter": {"bool": {"must_not": [exclusion_filter]}},
        }
    }

    search_query = {
        "knn": knn_query,
        "query": keyword_query,
        "size": k,
        "_source": ["text", "section", "question", "id"],
    }

    es_results = es_client.search(index=index_name, body=search_query)

    result_docs = []

    for hit in es_results["hits"]["hits"]:
        result_docs.append(hit["_source"])

    return result_docs


def get_answer(query: str, temperature: float = 0):

    vector = model.encode(query)
    search_results = elastic_search_hybrid(query, vector)

    prompt = build_prompt(query, search_results)
    answer, tokens, response_time = llm(prompt, temperature)

    relevance, explanation, eval_tokens = evaluate_relevance(query, answer)

    openai_cost = calculate_openai_cost(RAG_MODEL_NAME, tokens)

    return {
        "answer": answer,
        "response_time": response_time,
        "relevance": relevance,
        "relevance_explanation": explanation,
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "eval_prompt_tokens": eval_tokens["prompt_tokens"],
        "eval_completion_tokens": eval_tokens["completion_tokens"],
        "eval_total_tokens": eval_tokens["total_tokens"],
        "openai_cost": openai_cost,
    }
