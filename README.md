# OLX helpdesk assistant

## Problem description

The goal is to create an assistant that can anwser user (client) questions based on the provided FAQ documents and terms&conditions.

The assistant should be able to:
1. reformulate the user query to make it compatible with the documents
2. find relevant documents (using elastic search)
3. provide an answer to the user query based on the retrieved documents

Notebooks, scripts and notes are based on/inspired by the LLM zoomcamp course.


## Requirements

- OPEN AI API key. Passed as an environment variable `OPENAI_API_KEY` e.g. in `.env` file.
- Python 3.8
- Make

## Creating virtual environment

```bash
make base-venv
make requirements.txt
make install
```

## Running streamlit app

```bash
make stremlit-app
```

If you ran it for the first time make sure that:

1. `.env` file is present in the root directory. Example file is provided in `.env.example`.
2. You have the `OPENAI_API_KEY` from the OpenAI platform.
2. You run `app/prep.py` script to prepare the elastic search index and database tables.
    - if you run the script outside of docker you need to adjust POSTGRES_HOST in `app/prep.py` to `localhost`
    - make sure that the code for indexing the documents is uncommented in `app/prep.py`. If you want a new index name, make sure to adjust it in `.env` file.
    - run the script with `python -u app/prep.py` to avoid buffering the logs.
    - if you don't use elastic search that runs from a docker-compose file, you need to adjust the `ELASTIC_URL_LOCAL` in `.env` file to point to the correct URL.

To shut down the app press `Ctrl+C` in the terminal. To clean up the docker containers run

```bash
make streamlit-app-down
```

## Research notes

Notes on the research and development process can be found in the `notebooks` directory. In particular in notebook `04` there is an evaluation of the RAG performance on the ground truth data.

## Ideas for the future development

1. Treat definitions as a separate group of documents. Use them to enrich the prompt separately from the main RAG flow.
2. Check different sentence embeddings.

