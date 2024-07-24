#librerie necessarie

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel, RunnableLambda
from IPython.display import JSON
import json
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements, elements_to_json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

#ollama

model = ChatOpenAI(
    base_url = "http://localhost:11434/v1",
    temperature = 0,
    api_key = "not-need",
    model_name ="llama3",
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Inizializzazione dell'indice FAISS
cartella = "algoritmo3"

if os.path.exists(cartella):
    # Carica indice FAISS dalla cartella corrente
    faiss_index = FAISS.load_local(
        cartella,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    # Crea indice FAISS dei chunk nella cartella attuale
    faiss_index = FAISS.from_documents(
        splits,
        embeddings
    )
    faiss_index.save_local(cartella)

retriever = faiss_index.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

prompt = ChatPromptTemplate.from_template("Sei un assistente preciso e attento; Rispondi a questa domanda in italiano: {question}, considera il seguente contesto {context}. rispondi in italiano")


def format_documents(all_documents):
     return "\n\n".join(doc.page_content for doc in all_documents)

rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# %%
def query(query):
    answer = rag_chain.invoke(query)
    return answer

def queryStream(query):
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)

# %%
#query("che cosa è un prompt")

# %%
queryStream("Dammi esempi di prompt efficaci")

