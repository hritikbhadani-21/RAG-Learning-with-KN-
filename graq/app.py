import streamlit as st
import os
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

# -------------------------------
# API KEY
# -------------------------------
groq_api_key = os.environ["GROQ_API_KEY"]

# -------------------------------
# INIT VECTOR STORE (once)
# -------------------------------
if "vectors" not in st.session_state:

    st.write("🔄 Building vector DB...")

    loader = WebBaseLoader("https://docs.smith.langchain.com/")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectors = FAISS.from_documents(chunks, embeddings)

    st.session_state.vectors = vectors

    st.write("✅ Ready")

# -------------------------------
# UI
# -------------------------------
st.title("💬 RAG Chat (Modern LangChain)")

query = st.text_input("Ask something:")

# -------------------------------
# LLM + PROMPT
# -------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Gemma-7b-it"
)

prompt = ChatPromptTemplate.from_template(
    """
Answer based ONLY on the context below:

<context>
{context}
</context>

Question: {question}
"""
)

# -------------------------------
# RETRIEVER
# -------------------------------
retriever = st.session_state.vectors.as_retriever()

# -------------------------------
# FORMAT DOCS FUNCTION
# -------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------
# LCEL CHAIN (MODERN WAY)
# -------------------------------
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# -------------------------------
# RUN
# -------------------------------
if query:
    start = time.time()

    response = chain.invoke(query)

    st.write("### ✅ Answer")
    st.write(response.content)

    st.write(f"⏱ Time: {time.time() - start:.2f}s")


# -------------------------------
# DEBUG VIEW (IMPORTANT)
# -------------------------------
if st.checkbox("Show retrieved docs"):
    docs = retriever.invoke(query if query else "LangSmith")

    for i, d in enumerate(docs):
        st.write(f"### Doc {i+1}")
        st.write(d.page_content[:400])