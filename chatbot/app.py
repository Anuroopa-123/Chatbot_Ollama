from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question: {question}")
    ]
)

# Streamlit UI
st.title('LangChain Demo with Ollama (Free LLM)')
input_text = st.text_input("Search the topic you want")

# ✅ Use Ollama instead of OpenAI
llm = ChatOllama(model="mistral")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Run
if input_text:
    st.write(chain.invoke({'question': input_text}))