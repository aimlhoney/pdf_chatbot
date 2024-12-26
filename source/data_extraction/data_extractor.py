from subprocess import CalledProcessError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tabula

def get_table_json_using_page_no(path, page_no):
    try:
        tables = tabula.read_pdf(path, stream=True,  pages=page_no)
        if not tables:
            return {"tables": ""}
        else:
            return {"tables": '\n'.join([df.to_json(orient='records') for df in tables])}
    except CalledProcessError:
        return {"tables": ""}

def get_splitter(chunk_size = 1000, overlap = 150):
    return RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = overlap
)

def get_pages(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    for index, page in enumerate(pages):
        contents = page.page_content
        page.page_content = {"body": contents, "tables": get_table_json_using_page_no(path, index + 1)}
    return pages


def extract_file_data(path, chunk_size = 1000, overlap = 150):
    pages = get_pages(path)
    splitter = get_splitter(chunk_size = chunk_size, overlap = overlap)
    return splitter.split_documents(pages)