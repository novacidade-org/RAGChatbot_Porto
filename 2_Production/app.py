import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import langchain
import os
from langchain_chroma import Chroma
import chromadb
import itertools


from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever

import pandas as pd
import torch

import Code
from Code import Chatbot

st.title("Porto & Norte Tourism Assistant")

# Initialize message history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Receive user input
prompt = st.chat_input("How can I help you?")
if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})  # Add to session state

    # Process the prompt using the Chatbot function
    # Treat the prompt if needed (uncomment and replace Code.limpar_texto_testset if necessary)
    prompt_treated = Code.limpar_texto_testset(prompt)  # Optional cleaning step
    response = Chatbot(prompt_treated)  # Replace with your Chatbot function
    st.session_state.messages.append({"role": "assistant", "content": response})  # Add to session state

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)