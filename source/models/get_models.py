from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

def get_prompt():
    template = """You are an agent capable of answering questions based on both a provided PDF document and a tabular DataFrame.

    **PDF Context:**
    You will first be provided with a text excerpt from a PDF document. Use this information when appropriate to answer the user's question.

    {pdf_context}

    **DataFrame Information:**
    You will also be provided with a description of a DataFrame table. This will include column names, an example row, and summary information. Use this information when appropriate for the user's question.
    DataFrame Description:
    {df_description}

    **Question:** {question}

    **Instructions:**

    1. Analyze the question and determine if the PDF document is more relevant to answer the question. If so, use PDF context to get the answer.
    2. If the PDF context alone can not satisfy the answer, analyze whether the table is required and get the answer from the table.
    3. Do not try to make up answers if the pdf context or dataframe can not help.
    4. **Combine the information** from both sources, where appropriate, to provide a thorough and clear answer.
    5. Return the answer as a JSON object with the following structure:
        ```json
        {{
          "pdf_answer": "answer from the PDF or 'not found'",
          "dataframe_answer": "answer from the DataFrame or 'not found'",
          "combined_answer": "Combined answer using both sources if applicable"
        }}
        ```

    Helpful Answer:
    """

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

