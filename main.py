from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import os
from pathlib import Path
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langdetect import detect
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import numpy as np
import requests
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks import CallbackManager
from langchain_core.outputs import LLMResult, Generation
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from typing import Any, List, Optional
from system_prompts import BANGLA_SYSTEM_PROMPT, ENGLISH_SYSTEM_PROMPT

load_dotenv()

DB_NAME = "pdf_chunks"
PDF_PATH = Path("data/HSC26-Bangla1st-Paper.pdf")
base = Path(__file__).parent
tessdata_path = base / "assets/fonts"
os.environ["TESSDATA_PREFIX"] = str(tessdata_path)

os.environ["SILICONFLOW_API_KEY"] = os.getenv("SILICONFLOW_API_KEY") or ""
if not os.getenv("SILICONFLOW_API_KEY"):
    raise RuntimeError("SILICONFLOW_API_KEY not found")

app = FastAPI(title="Bangla-PDF RAG with Memory")

class Embeddings:
    def embed_documents(self, texts):
        return [get_embeddings(text) for text in texts]
    
    def embed_query(self, text):
        return get_embeddings(text)

memory = ConversationBufferMemory(return_messages=True, input_key="query", output_key="answer")

class SiliconFlowLLM(BaseLLM):
    """Custom LLM for SiliconFlow API with Kimi-K2-Instruct model"""
    api_key: str
    model: str = "moonshotai/Kimi-K2-Instruct"
    url: str = "https://api.siliconflow.com/v1/chat/completions"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "stream": False,
            "max_tokens": 512,
            "enable_thinking": False,
            "thinking_budget": 4096,
            "min_p": 0.05,
            "temperature": 0.3,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "stop": stop or [],
            "messages": kwargs.get("messages", [
                {"role": "system", "content": kwargs.get("system_prompt", "")},
                {"role": "user", "content": prompts[0]}
            ])
        }

        response = requests.post(self.url, json=payload, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to get response from SiliconFlow API")

        answer = response.json()["choices"][0]["message"]["content"]
        return LLMResult(generations=[[Generation(text=answer)]])

    def invoke(self, input: str, **kwargs: Any) -> str:
        """Override invoke to return a string directly, as expected by LangChain"""
        result = self._generate([input], stop=kwargs.get("stop"), **kwargs)
        return result.generations[0][0].text
    
    @property
    def _llm_type(self) -> str:
        return "siliconflow"

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

def create_sentence_boundary_chunks(sentences: list[str], max_chunk_size: int = 800, overlap_sentences: int = 1) -> list[str]:
    """Create chunks based on sentences with sentence-boundary overlap"""
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
    """Process PDF with sentence-boundary chunking"""
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")
    
    full_text = ocr_pdf_to_text(pdf_path)
    print(f"Extracted text length: {len(full_text)}")
    
    sentences = split_into_sentences(full_text)
    print(f"Found {len(sentences)} sentences")
    print(f"Sample sentences: {sentences[:3]}")
    
    chunk_texts = create_sentence_boundary_chunks(sentences, max_chunk_size=800, overlap_sentences=2)
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

def get_embeddings(text: str) -> list[float]:
    """Get embeddings using Qwen3-Embedding-8B via SiliconFlow API"""
    url = "https://api.siliconflow.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": text,
        "encoding_format": "float"
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to get embeddings from SiliconFlow API")
    
    return response.json()["data"][0]["embedding"]

def build_vector_store(chunks):
    embeddings = Embeddings()
    
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

def generate_improved_prompt(context: str, question: str, history: str = "") -> str:
    query_lang = detect(question)
    
    if query_lang == 'bn':
        return f"""প্রসঙ্গ: {context}

        পূর্ববর্তী কথোপকথন: {history}

        প্রশ্ন: {question}

        উত্তর:
        # শুধু বাংলায় উত্তর দিন"""
    else:
        return f"""Context: {context}

        Previous Conversation: {history}

        Question: {question}

        Answer:
        # Answer in English only"""

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
    
    memory_dict = memory.load_memory_variables({})
    history = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
        for msg in memory_dict.get('history', [])
    ])
    
    prompt = generate_improved_prompt(context, query, history)
    
    llm = SiliconFlowLLM(api_key=os.getenv("SILICONFLOW_API_KEY"))
    query_lang = detect(query)
    system_prompt = BANGLA_SYSTEM_PROMPT if query_lang == 'bn' else ENGLISH_SYSTEM_PROMPT
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    answer = llm.invoke(prompt, messages=messages, system_prompt=system_prompt)
    
    memory.save_context({"query": query}, {"answer": answer})
    
    return {
        "answer": answer,
        "source_documents": docs,
        "context": context,
        "debug_info": {
            "num_retrieved": len(docs),
            "context_length": len(context),
            "query_detected_lang": detect(query),
            "conversation_history": history
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
        embeddings = Embeddings()
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
        "message": "Sentence-Aware Bangla RAG ready with Memory", 
        "features": [
            "Sentence boundary detection for Bangla (।) and English (.)",
            "Semantic chunking with sentence overlap",
            "MMR retrieval for diverse results",
            "Improved prompting for better extraction",
            "Short-term memory for conversation history"
        ]
    }