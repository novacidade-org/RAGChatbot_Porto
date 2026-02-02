import os
import re
import itertools
import pandas as pd

import chromadb
import langchain
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


import torch
print(torch.cuda.is_available())
device = ["cuda" if torch.cuda.is_available() else "cpu"]


def limpar_texto(texto):
    # Transformar em minúsculas
    texto = texto.lower()
    # Remover caracteres específicos
    texto = texto.replace("(", "").replace(")", "").replace(",", "").replace("\n", " ").replace("\xa0"," ")
    texto = re.sub(r"\. ", " ", texto)
    texto = re.sub(r"\! ", " ", texto)
    texto = re.sub(r"\" ", " ", texto)
    texto = re.sub(r'"(\w+)', r'\1', texto)  # Remove apenas a aspas antes da palavra
    texto = re.sub(r"\: ", " ", texto)
    texto = texto.replace("  ", " ")
    texto = texto.replace(" - ", " ")
    texto = re.sub(r"'s\b", "", texto)  # Remove apenas 's no final de palavras
    
    return texto

def limpar_texto_testset(texto):
    # Transformar em minúsculas
    texto = texto.lower()
    # Remover caracteres específicos
    texto = texto.replace("(", "").replace(")", "").replace(",", "").replace("\n", " ").replace("\xa0"," ")
    texto = re.sub(r"\. ", " ", texto)
    texto = re.sub(r"\! ", " ", texto)
    texto = re.sub(r"\" ", " ", texto)
    texto = re.sub(r'"(\w+)', r'\1', texto)  # Remove apenas a aspas antes da palavra
    texto = re.sub(r"\: ", " ", texto)
    texto = texto.replace("  ", " ")
    texto = texto.replace(" - ", " ")
    texto = re.sub(r"'s\b", "", texto)  # Remove apenas 's no final de palavras

    texto = re.sub(r"\?", " ", texto)

    return texto

def BM25TextPreparation(data,Series_metadata, chunk_size):
    text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,  # Max number of characters per chunk
    # chunk_overlap=chunk_overlap  # Overlap between chunks for better context retention
    )

    # Step 2: Prepare an empty list to store documents with their associated metadata
    split_docs = []

    # Step 3: Loop through each record in `data["Final"]` and split it, keeping the source ("Fonte")
    for i, final_text in enumerate(data):
        # Split the text into chunks
        docs = text_splitter.create_documents([final_text])
        
        # Add the "Fonte" to the metadata of each document
        for doc in docs:
        
            doc.metadata = {"Source":Series_metadata[i]}  # Associate each chunk with the correct source
            split_docs.append(doc)  # Store the document with the metadata

    texts = [str(doc.page_content) for doc in split_docs]

    return split_docs

main_path = os.path.dirname(os.getcwd())
# need to import the Corpus again to create the sparse vector for BM25Retriever
df_db = pd.read_excel(os.path.join(main_path, "1_ModelDevelopment", "2_Data", "1_Final", "Dataset_Final.xlsx")).drop("Unnamed: 0", axis=1)
# Clean text in order to have exactly the same text that is stored in the vector databases. Otherwise the Reciprocal Ranking Fusion formula will not work.
df_db["Texto_lower"] = [limpar_texto(texto) for texto in df_db["Texto"].to_list()]
# list with the documents that compõe o Corpus with 7000 chunk-size
texts_3000 = BM25TextPreparation(df_db["Texto_lower"].to_list(),df_db["Source"],3000)




model_kwargs = {'device': device[0]}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


vector_store = Chroma(collection_name="DB_Porto_Final_bge_embeddings_treated_3000",
                      persist_directory=os.path.join(main_path, "1_ModelDevelopment", "2_Data", "2_DBs"),
                      embedding_function=embeddings)



bm25_retriever = BM25Retriever.from_documents(texts_3000)
bm25_retriever.k = 50
retriever = vector_store.as_retriever(search_kwargs={"k": 50})

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5])
CrossEncoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", model_kwargs = {'device': device[0]})
compressor = CrossEncoderReranker(model=CrossEncoder_model, top_n=8)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

chat_history = [] 

def Chatbot(query,model="llama3.1:8b"):
    
    system_prompt = (
    """
    System: This is a Chatbot that only answers to questions related to Porto (Portugal) Tourism. More specifically, to topics related to attractions, accessibility, amenities, activities, available packages, and Ancillary Services. 
    When not specicified by the user assume the question is related to Porto.
    If the question is not about Porto Tourism just write: "I am sorry, but my knowledge only allows me to help you with Porto Tourism topics. Can I help you with something related to Porto Tourism?"
    
    Answer to the user's question objectively with only the necessary information, using correct syntax and based on context writen below: 
    {context}\n

    User: {input}
    """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("system", "{context}"),
            ("human", "{input}"),
        ]
    )


    generator_model = ChatOllama(
                    model=model,
                    base_url="http://localhost:11434",
                    temperature=0,
                    num_gpu=1,
                    num_ctx=50000,
                )
        

        
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, formulate a question that rewrites the user question and the chat history. \
        Do NOT answer the question just reformulate it if needed and otherwise return it as is. Write everything lower case without punctuation"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        generator_model, compression_retriever, contextualize_q_prompt
    )
        
        
    # Chain que monta o prompt para passar no modelo
    question_answer_chain = create_stuff_documents_chain(generator_model, prompt)
    
    # chain que pega no contexto e adiciona ao pompt para passar no modelo
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    response = rag_chain.invoke(
    {"input":query,
     "chat_history": chat_history})
    
    chat_history.append("User: "+response['input']+". Assistant: "+response['answer'])

    return response["answer"]