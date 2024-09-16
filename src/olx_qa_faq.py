import streamlit as st
import time
import logging
from elasticsearch import Elasticsearch
from openai import OpenAI

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

client = OpenAI(
    api_key=''
)

es_client = Elasticsearch('http://localhost:9200') 

model_name = 'xlm-r-distilroberta-base-paraphrase-v1'
embedding_model = SentenceTransformer(model_name)

def elastic_search(query_vector, index_name = "documents_olx_20240821_4322"):
    
    knn = {
        "field": "question_text_vector",
        "query_vector": query_vector,
        "k": 5,
        "num_candidates": 10000,
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
            index=index_name,
            body=search_query
        )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs


def build_prompt(query, search_results):
    prompt_template = """
Jesteś pracownikiem działu obsługi klienta firmy OLX. 
Odpowiedz na PYTANIE klienta bazując na KONTEKST pochodzący z bazy danych częstych pytań i odpowiedzi oraz regulaminu serwisu.
Bazuj jedynie na wiedzy z KONTEKST kiedy odpowiadasz na PYTANIE.

PYTANIE: {question}

KONTEKST: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def rag(query):
    logger.warning(f"Query: {query}")
    query_vector = embedding_model.encode(query)
    search_results = elastic_search(query_vector)
    prompt = build_prompt(query, search_results)
    logger.warning(f"Prompt: {prompt}")
    answer = llm(prompt)
    return answer


# Streamlit app
def main():
    st.title("RAG Function Invocation")

    # Input box
    user_input = st.text_input("Enter your input:")

    # Button to invoke the RAG function
    if st.button("Ask"):
        with st.spinner('Processing...'):
            # Call the RAG function
            output = rag(user_input)
            st.success("Completed!")
            st.write(output)

if __name__ == "__main__":
    main()
