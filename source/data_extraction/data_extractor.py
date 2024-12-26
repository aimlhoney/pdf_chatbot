from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def get_splitter(chunk_size = 1000, overlap = 150):
    return RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = overlap
)

def get_pages(path):
    loader = PyPDFLoader(path)
    return loader.load()

def extract_file_data(path, chunk_size = 1000, overlap = 150):
    pages = get_pages(path)
    splitter = get_splitter(chunk_size = chunk_size, overlap = overlap)
    return splitter.split_documents(pages)