from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import os
from pathlib import Path
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langdetect import detect
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import numpy as np
from system_prompts import BANGLA_SYSTEM_PROMPT, ENGLISH_SYSTEM_PROMPT

load_dotenv()

DB_NAME = "pdf_chunks"
PDF_PATH = Path("data/HSC26-Bangla1st-Paper.pdf")
base = Path(__file__).parent
tessdata_path = base / "assets/fonts"
os.environ["TESSDATA_PREFIX"] = str(tessdata_path)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found")

app = FastAPI(title="Bangla-PDF RAG")

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text

def preprocess_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.medianBlur(image, 3)
    return Image.fromarray(image)

def ocr_pdf_to_text(pdf_path: Path, dpi: int = 300) -> str:
    images = convert_from_path(pdf_path, dpi=dpi, fmt="png")
    text = ""
    for page_num, page in enumerate(images, 1):
        page = preprocess_image(page)
        page_text = pytesseract.image_to_string(
            page,
            lang="ben+eng",
            config=f'--psm 6 --oem 3 --tessdata-dir "{tessdata_path}"'
        )
        text += f"\n[Page {page_num}]\n{page_text}\n"
    return clean_text(text)

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences respecting both English and Bangla punctuation"""
    sentence_endings = r'[।!?\.]\s*'
    
    sentences = re.split(f'({sentence_endings})', text)
    
    complete_sentences = []
    current_sentence = ""
    
    for i, part in enumerate(sentences):
        if re.match(sentence_endings, part):
            current_sentence += part.strip()
            if current_sentence.strip():
                complete_sentences.append(current_sentence.strip())
            current_sentence = ""
        else:
            current_sentence += part
    
    if current_sentence.strip():
        complete_sentences.append(current_sentence.strip())
    
    return [s for s in complete_sentences if len(s.strip()) > 10]

def create_semantic_chunks(sentences: list[str], max_chunk_size: int = 800, overlap_sentences: int = 1) -> list[str]:
    """Create chunks based on sentences with semantic overlap"""
    if not sentences:
        return []
    
    chunks = []
    current_chunk = ""
    current_sentences = []
    
    for sentence in sentences:
        if len(current_chunk + " " + sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            if overlap_sentences > 0 and len(current_sentences) > overlap_sentences:
                overlap_text = " ".join(current_sentences[-overlap_sentences:])
                current_chunk = overlap_text + " " + sentence
                current_sentences = current_sentences[-overlap_sentences:] + [sentence]
            else:
                current_chunk = sentence
                current_sentences = [sentence]
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_sentences.append(sentence)
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

async def process_pdf_with_sentence_chunking(pdf_path: Path):
    """Process PDF with sentence-aware chunking"""
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")
    
    full_text = ocr_pdf_to_text(pdf_path)
    print(f"Extracted text length: {len(full_text)}")
    
    sentences = split_into_sentences(full_text)
    print(f"Found {len(sentences)} sentences")
    print(f"Sample sentences: {sentences[:3]}")
    
    chunk_texts = create_semantic_chunks(sentences, max_chunk_size=800, overlap_sentences=2)
    print(f"Created {len(chunk_texts)} chunks")
    
    chunks = []
    for i, chunk_text in enumerate(chunk_texts):
        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": str(pdf_path),
                "chunk_id": i,
                "total_chunks": len(chunk_texts),
                "chunk_length": len(chunk_text),
                "chunking_method": "sentence_based"
            }
        )
        chunks.append(doc)
    
    return chunks

def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    if Path(DB_NAME).exists():
        try:
            existing_store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
            existing_store.delete_collection()
        except:
            pass
    
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_NAME
    )
    
    print(f"Vector store created with {len(chunks)} documents")
    return vector_store

def generate_improved_prompt(context: str, question: str) -> str:
    query_lang = detect(question)
    
    if query_lang == 'bn':
        return f"""প্রসঙ্গ: {context}

প্রশ্ন: {question}

উত্তর:"""
    else:
        return f"""Context: {context}

Question: {question}

Answer:"""

def query_rag_system(query: str, vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )
    
    docs = retriever.get_relevant_documents(query)
    
    context_parts = []
    for doc in docs:
        context_parts.append(doc.page_content)
    
    context = "\n\n".join(context_parts)
    
    print(f"\n=== RETRIEVAL DEBUG ===")
    print(f"Query: {query}")
    print(f"Retrieved {len(docs)} documents")
    print(f"Context length: {len(context)}")
    
    prompt = generate_improved_prompt(context, query)
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    query_lang = detect(query)
    system_prompt = BANGLA_SYSTEM_PROMPT if query_lang == 'bn' else ENGLISH_SYSTEM_PROMPT
    
    messages = [
        ("system", system_prompt),
        ("human", prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "answer": response.content,
        "source_documents": docs,
        "context": context,
        "debug_info": {
            "num_retrieved": len(docs),
            "context_length": len(context),
            "query_detected_lang": detect(query)
        }
    }

class QueryRequest(BaseModel):
    query: str

@app.get("/preprocess")
async def preprocess_pdf():
    try:
        chunks = await process_pdf_with_sentence_chunking(PDF_PATH)
        vector_store = build_vector_store(chunks)
        
        return {
            "message": "PDF processed with sentence-based chunking", 
            "total_chunks": len(chunks),
            "chunking_method": "sentence_based",
            "sample_chunks": [
                {
                    "id": i,
                    "preview": chunk.page_content[:200] + "...",
                    "length": len(chunk.page_content),
                    "sentences": len(chunk.page_content.split('।')) + len(chunk.page_content.split('.'))
                }
                for i, chunk in enumerate(chunks[:3])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(req: QueryRequest):
    if not Path(DB_NAME).exists():
        raise HTTPException(status_code=400, detail="Run /preprocess first")
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
        
        result = query_rag_system(req.query, vector_store)
        
        return {
            "answer": result["answer"],
            "debug_info": result["debug_info"],
            "sources": [
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "preview": doc.page_content[:200] + "...",
                    "length": len(doc.page_content)
                }
                for doc in result["source_documents"]
            ],
            "context_preview": result["context"][:500] + "..."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Sentence-Aware Bangla RAG ready", 
        "features": [
            "Sentence boundary detection for Bangla (।) and English (.)",
            "Semantic chunking with sentence overlap",
            "MMR retrieval for diverse results",
            "Improved prompting for better extraction"
        ]
    }