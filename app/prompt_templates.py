assistant_prompt_template = """
You work as a customer support specialist at a classifieds platform OLX in Poland.
Respond to the cusomer's QUESTION based on the CONTEXT from the database of frequently asked questions and answers and the terms and conditions of the service. 
Base your response solely on the CONTEXT when answering the QUESTION.
If you are unsure about the answer, you can ask for more information.
Respond to the QUESTION in Polish.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()


reranking_prompt_template = """
The following documents were retrieved from the database of a faq and terms of service of a classifieds platform OLX in Poland.

Rerank the documents based on their relevance to the QUESTION and select the top k={k} documents that are most relevant to the QUESTION.

QUESTION: {question}

DOCUMENTS:
1. {doc1_question}
2. {doc2_question}
3. {doc3_question}
4. {doc4_question}
5. {doc5_question}
6. {doc6_question}
7. {doc7_question}
8. {doc8_question}
9. {doc9_question}
10. {doc10_question}
11. {doc11_question}
12. {doc12_question}
13. {doc13_question}
14. {doc14_question}
15. {doc15_question}

Your response should be in parsable JSON format. Do not use code blocks:

["id_1", "id_2", ..., "id_k"]
""".strip()


retrieval_eval_prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    The system's purpose is to be a customer support assistant for a classifieds platform. 
    It should not provide legal advice or personal opinions.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()
