from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant ,please response to the user querys"),
        ("user","Question:{question}")
    ]
)

st.title('Langchain Demo with OpenSource')
input_text=st.text_input("Search the topic you want")

llm = Ollama(model="tinyllama")
outputParser=StrOutputParser()
chain=prompt|llm|outputParser

if input_text:
    st.write(chain.invoke({"question":input_text}))
