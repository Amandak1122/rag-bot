from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

st.title("Document Q&A Bot")
st.write("PDF upload karo aur sawaal pucho!")

uploaded_file = st.file_uploader("PDF choose karo", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(model="llama-3.1-8b-instant")

    prompt = ChatPromptTemplate.from_template("""
    Neeche diye gaye context ke basis par sawaal ka jawab do.
    Context: {context}
    Sawaal: {question}
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    st.success("PDF ready hai! Ab sawaal pucho.")
    question = st.text_input("Apna sawaal likho:")

    if question:
        answer = chain.invoke(question)
        st.write("**Jawab:**", answer.content)