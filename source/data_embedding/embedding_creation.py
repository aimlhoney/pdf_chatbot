# Assuming `splits` contains the list of Document objects
from pydantic_settings import BaseSettings
from dotenv import main
import os

main.load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.vectorstores import Chroma

def get_embedding_storage(api_key: SecretStr, splits, k=3, model="models/embedding-001"):
    return Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        # Directory for storing Chroma's data
    ).as_retriever(search_kwargs={"k": k})
