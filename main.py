from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import os
from pathlib import Path
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from googletrans import Translator, LANGUAGES
from langdetect import detect
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import numpy as np

load_dotenv()

MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-large"
DB_NAME = "pdf_chunks"
PDF_PATH = Path("data/HSC26-Bangla1st-Paper.pdf")
base = Path(__file__).parent
tessdata_path = base / "assets/fonts"
os.environ["TESSDATA_PREFIX"] = str(tessdata_path)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found")

app = FastAPI(title="Bangla-PDF RAG")
translator = Translator()

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
    for page in images:
        page = preprocess_image(page)
        text += pytesseract.image_to_string(
            page,
            lang="ben",
            config=f'--psm 6 --oem 3 --tessdata-dir "{tessdata_path}"'
        )
    return clean_text(text)

async def translate_text(text: str, target_language: str) -> str:
    translator = Translator()
    try:
        lang_code = (
            'en' if target_language.lower() == 'english' else
            'bn' if target_language.lower() == 'bengali' else
            target_language.lower()
        )
        if lang_code not in LANGUAGES:
            raise ValueError(f"Unsupported target language: {target_language}")

        result = await translator.translate(text, dest=lang_code) 
        print(f"Translated text to {lang_code}: {result.text}...") 
        return result.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

async def process_pdf(pdf_path: Path, chunk_size: int = 2000, chunk_overlap: int = 300):
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")
    full_text = ocr_pdf_to_text(pdf_path)
    doc = Document(page_content=full_text, metadata={"source": str(pdf_path)})
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents([doc])
    translated = []
    for chk in chunks:
        en_text = await translate_text(chk.page_content, "english")
        translated.append(Document(page_content=en_text, metadata=chk.metadata))
    return translated

def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if Path(DB_NAME).exists():
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)

def build_chain(store):
    llm = ChatOpenAI(model=MODEL, temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=store.as_retriever(), memory=memory)

class QueryRequest(BaseModel):
    query: str

@app.get("/preprocess")
async def preprocess_pdf():
    try:
        chunks = await process_pdf(PDF_PATH)
        build_vector_store(chunks)
        print(chunks)
        return {"message": "PDF processed and vector store created", "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(req: QueryRequest):
    if not Path(DB_NAME).exists():
        raise HTTPException(status_code=400, detail="Run /preprocess first")
    query_lang = detect(req.query)
    question_en = await translate_text(req.query, "english") if query_lang != "en" else req.query
    store = Chroma(persist_directory=DB_NAME, embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL))
    chain = build_chain(store)
    result = chain.invoke({"question": question_en, "chat_history": []})
    answer = await translate_text(result["answer"], "bengali") if query_lang == "bn" else result["answer"]
    return {"answer": answer, "detected_language": query_lang}

@app.get("/")
async def root():
    return {"message": "Bangla-PDF RAG ready", "endpoints": {"/preprocess": "ingest & translate PDF", "/query": "ask questions"}}