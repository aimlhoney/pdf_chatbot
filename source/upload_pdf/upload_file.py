from typing import Any


from data_embedding.embedding_creation import get_embedding_storage
from data_extraction.data_extractor import extract_file_data
from models.get_models import get_models, get_prompt, get_retriever
import chromadb.api



def upload_pdf(pdf_path: str, api_key) :
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    splitter = extract_file_data(pdf_path)
    storage = get_embedding_storage(api_key, splitter)
    model = get_models(api_key)
    QA_CHAIN_PROMPT = get_prompt()
    qa_chain = get_retriever(model, storage, QA_CHAIN_PROMPT)
    return qa_chain



