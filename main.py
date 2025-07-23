from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os
from langdetect import detect
import torch

load_dotenv()

app = FastAPI()

DB_NAME = "pdf_chunks"

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\u0980-\u09FF\s\w.,!?]', '', text)
    return text

def process_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 100):
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
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )
    
    return vector_store

def create_retrieval_chain(vector_store):
    model_name = "facebook/mbart-large-50"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        device=0 if torch.cuda.is_available() else -1
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vector_store.as_retriever()

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    return retrieval_chain

class QueryRequest(BaseModel):
    query: str

@app.get("/preprocess")
async def preprocess_pdf():
    pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
    try:
        chunks = process_pdf(pdf_path)
        vector_store = create_chroma_collection(chunks)
        chunked_data = [{"content": doc.page_content, "metadata": doc.metadata} for doc in chunks]
        return {"message": "Processed successfully", "chunks": chunked_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        if not os.path.exists(DB_NAME):
            raise HTTPException(status_code=400, detail="Vector store not initialized. Run /preprocess first.")
        
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
        
        retrieval_chain = create_retrieval_chain(vector_store)
        
        query_lang = detect(request.query)
        print(f"Detected query language: {query_lang}")
        
        result = retrieval_chain({"question": request.query})
        
        return {
            "answer": result["answer"],
            "detected_language": query_lang
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "PDF preprocessing endpoint ready. Send POST requests to /preprocess or /query with a JSON body containing 'query'."}