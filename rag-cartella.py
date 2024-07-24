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

#ollama

model = ChatOpenAI(
    base_url = "http://localhost:11434/v1",
    temperature = 0,
    api_key = "not-need",
    model_name ="llama3",
)

# Percorso della cartella contenente i PDF
folder_path = "/home/nugh75/git-repository/env1/Nugh75-didattico-repository/pdfs2"

# Lista per memorizzare tutti i documenti caricati
all_documents = []

# Iterare su tutti i file nella cartella
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
        all_documents.extend(documents)

# Verifica e stampa il contenuto delle pagine
for doc in all_documents:
    if hasattr(doc, 'page_content'):
        print(doc.page_content)

# Funzione per formattare i documenti
def format_documents(documents):
    return "\n\n".join(doc.page_content for doc in documents if hasattr(doc, 'page_content'))


from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =500,
    chunk_overlap=200
)
splits =text_splitter.split_documents(all_documents)
for sp in splits:
    if (len(sp.page_content) <100):
        splits.remove(sp)

#Select Embeddings model questo funziona

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

cartella="linguistica"


if os.path.exists(cartella):
    #Carica indice FAISS cartella corrente attuale
    faiss_index=FAISS.load_local(
        cartella,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    #crea indice FAISS dei chunk nella cartella attuale
    faiss_index = FAISS.from_documents(
        splits,
        embeddings
    )
    faiss_index.save_local(cartella)

