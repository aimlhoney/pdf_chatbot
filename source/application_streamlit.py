import tempfile
import pathlib
import os
import streamlit as st
from upload_pdf.upload_file import upload_pdf
st.set_page_config(layout="wide")
temp_dir = tempfile.TemporaryDirectory()
st.title("PDF BOT")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_file_uploaded" not in st.session_state:
    st.session_state.is_file_uploaded = False


with st.chat_message("bot"):
    st.write("Please Upload a pdf file.")
    file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False, label_visibility="collapsed")
    uploaded_file_name = "uploaded_file.pdf"
    uploaded_file_path = pathlib.Path(temp_dir.name) / uploaded_file_name
    if file is not None:
        with open(uploaded_file_path, 'wb') as output_temporary_file:
            output_temporary_file.write(file.read())
        qa_chain = upload_pdf(str(uploaded_file_path), os.getenv("GEMINI_API_KEY"))
        st.session_state.is_file_uploaded = True

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.is_file_uploaded:
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = qa_chain({"query": prompt})["result"]
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
