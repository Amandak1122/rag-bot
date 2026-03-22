from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import chromadb 
import streamlit as st

load_dotenv()

st.title("📄 Document Q&A Bot")
st.write("Upload your PDF and start asking questions!")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox(
        "Answer language:",
        ["English", "Hindi", "Hinglish"]
    )

# Session state initialize
if "chain" not in st.session_state:
    st.session_state.chain = None
if "last_file" not in st.session_state:
    st.session_state.last_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and st.session_state.last_file != uploaded_file.name:
    with st.spinner("Processing your PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        chunks = [c for c in chunks if c.page_content.strip()]
        client = chromadb.EphemeralClient()
vectorstore = Chroma.from_documents(
      documents=chunks,
    embedding=embeddings,
    client=client
)
retriever = vectorstore.as_retriever()

llm = ChatGroq(model="llama-3.1-8b-instant")

def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
            formatted = ""
            for msg in history:
                if isinstance(msg, HumanMessage):
                    formatted += f"User: {msg.content}\n"
                else:
                    formatted += f"Assistant: {msg.content}\n"
            return formatted

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based on the provided context and conversation history.
If the answer is not found in the context, honestly say so.
Always respond in {language}.

Conversation History:
{chat_history}

Context:
{context}

Sawaal: {question}
""")
chain = (
            {
                "context": RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))),
                "question": RunnableLambda(lambda x: x["question"]),
                "chat_history": RunnableLambda(lambda x: format_history(x["chat_history"])),
                "language": RunnableLambda(lambda x: x["language"]),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        st.session_state.chain = chain
        st.session_state.last_file = uploaded_file.name
        st.session_state.chat_history = []

st.success("PDF is ready! You can now ask your questions.")

# Chat UI
if st.session_state.chain:
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    question = st.chat_input("Type your question here...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke({
                    "question": question,
                    "chat_history": st.session_state.chat_history,
                    "language": language
                })
                st.write(answer)

        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])