import os
import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

class ElasticsearchLoggingHandler(logging.Handler):
    def __init__(self, es_client, index):
        super().__init__()
        self.es_client = es_client
        self.index = index

    def emit(self, record):
        log_entry = self.format(record)
        doc = {
            '@timestamp': record.created,
            'level': record.levelname,
            'message': log_entry
        }
        self.es_client.index(index=self.index, document=doc)
        
ELASTIC_URL = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")

es_client = Elasticsearch(ELASTIC_URL)

def setup_es_handler():
    es_handler = ElasticsearchLoggingHandler(es_client, 'logs-index')
    es_handler.setLevel(logging.INFO)
    return es_handler
