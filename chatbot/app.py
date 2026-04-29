from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

# Set OpenRouter config
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

# VERY IMPORTANT → OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Question: {question}")
])

# UI
st.title('LangChain Demo with OpenRouter (Free LLM)')
input_text = st.text_input("Search the topic you want")

# ✅ Free fast model (recommended)
llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    base_url=OPENROUTER_BASE_URL,
    temperature=0.7
)

# Chain
chain = prompt | llm | StrOutputParser()

# Run
if input_text:
    st.write(chain.invoke({"question": input_text}))