from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

def get_prompt():
    template = """You are an agent who can provide verry accurate answers to the given question only from the uploaded pdf. 
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you are so sorry that you could not get the answer from given pdf, don't try to make up an answer. Keep the answer as concise as possible and always provide in a json format.
    {context}
    Question: {question}
    Helpful Answer:"""
    return PromptTemplate.from_template(template)

def get_models( api_key:SecretStr, model = "gemini-1.5-pro",):
    return ChatGoogleGenerativeAI(model=model,google_api_key=api_key,
                             temperature=0.2,convert_system_message_to_human=True)

def get_retriever(model, storage, prompt):
    return RetrievalQA.from_chain_type(
        model,
        retriever=storage,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

