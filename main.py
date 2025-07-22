from fastapi import FastAPI, HTTPException
import re
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

EMBEDDING_MODEL = "text-embedding-3-small"
DB_NAME = "pdf_chunks"

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\b\d+\b', '', text) 
    return text

def process_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found at {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    full_text = ' '.join(doc.page_content for doc in docs)
    cleaned_text = clean_text(full_text)
    
    doc = Document(page_content=cleaned_text, metadata={"source": pdf_path})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents([doc])
    return chunks

def create_chroma_collection(chunks):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )
    print(f"Chroma collection created with {vector_db._collection.count()} documents")
    return vector_db

@app.get("/preprocess")
async def preprocess_pdf():
    pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
    try:
        chunks = process_pdf(pdf_path)
        create_chroma_collection(chunks)
        chunked_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in chunks]
        return {"chunks": chunked_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "PDF preprocessing endpoint ready. Send POST requests to /preprocess with a JSON body containing 'query'."}