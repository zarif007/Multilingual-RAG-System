# Multilingual-RAG-System

This is a FastAPI-based Retrieval-Augmented Generation (RAG) system designed to process Bangla PDFs and answer questions in both Bangla and English. It leverages sentence-aware chunking, SiliconFlow's API for embeddings and language modeling, and Chroma for vector storage, with conversation memory for context-aware responses.

The system is hosted at [https://multilingual-rag-system.onrender.com/](https://multilingual-rag-system.onrender.com/). You can use the hosted version directly or set it up locally for development or customization.

## Setup Guide

Follow these steps to set up the project locally. Alternatively, use the hosted version at [https://multilingual-rag-system.onrender.com/](https://multilingual-rag-system.onrender.com/).

### Prerequisites

- Python 3.9 or higher
- A SiliconFlow API key (obtainable from [https://cloud.siliconflow.com/account/ak](https://cloud.siliconflow.com/account/ak))
- A compatible environment with required dependencies

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/zarif007/Multilingual-RAG-System.git
   cd Multilingual-RAG-System
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the project root directory and add your SiliconFlow API key:

   ```bash
   echo "SILICONFLOW_API_KEY=your-api-key-here" > .env
   ```

   Replace `your-api-key-here` with the API key obtained from [https://cloud.siliconflow.com/account/ak](https://cloud.siliconflow.com/account/ak).

5. **Run the Application**
   Start the FastAPI server using Uvicorn:
   ```bash
   uvicorn main:app --reload
   ```
   The server will be available at `http://localhost:8000`. Alternatively, use the hosted version at [https://multilingual-rag-system.onrender.com/](https://multilingual-rag-system.onrender.com/).

## Used Tools, Libraries, and Packages

- **FastAPI**: Web framework for building the API server.
- **Pydantic**: Data validation and settings management.
- **LangChain**: Framework for building applications with LLMs, used for document processing and memory.
- **langchain-chroma**: Chroma integration for LangChain to manage vector storage.
- **langdetect**: Language detection for queries.
- **python-dotenv**: Loads environment variables from a `.env` file.
- **pdf2image**: Converts PDF pages to images for OCR.
- **pytesseract**: Python wrapper for Tesseract OCR to extract text from images.
- **opencv-python**: Image preprocessing for improved OCR accuracy.
- **numpy**: Numerical computations for image processing and similarity calculations.
- **requests**: HTTP requests to interact with SiliconFlow API.
- **scikit-learn**: Cosine similarity calculations for evaluation metrics.
- **uvicorn**: ASGI server implementation for running FastAPI.
- **SiliconFlow API**: Provides embeddings (`Qwen/Qwen3-Embedding-8B`) and language modeling (`moonshotai/Kimi-K2-Instruct`).

## Sample Queries and Outputs

### Bangla Query

**Query**: "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
**CURL Command** (using hosted URL):

```bash
curl -X POST https://multilingual-rag-system.onrender.com/query -H "Content-Type: application/json" -d '{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
```

**Sample Output**:

```json
{
  "answer": "শুম্ভুনাথ",
  "sources": [
    {
      "chunk_id": 3,
      "preview": "...",
      "length": 750
    },
    ...
  ],
  "context_preview": "...",
  "evaluation": {
    "relevance": {
      "average_cosine_similarity": 0.92,
      "max_cosine_similarity": 0.95,
      "num_retrieved_docs": 6
    },
    "groundedness": {
      "overlap_score": 0.85,
      "common_terms": ["শুম্ভুনাথ", "অনুপমের"]
    }
  }
}
```

### English Query

**Query**: "Who is referred to as the God of Anupam Bhagya?"
**CURL Command** (using hosted URL):

```bash
curl -X POST https://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "Who is referred to as the God of Anupam Bhagya?"}'
```

**Sample Output**:

```json
{
  "answer": "Mama",
  "sources": [
    {
      "chunk_id": 3,
      "preview": "title",
      "content": "...",
      "sources": [
        {
          "chunk_id": 3,
          "preview": "...",
          "length": 4
        }
      ]
    }
    ...
  ],
  "context_preview": "...",
  "evaluation": "{
    "relevance": {
      "average_cosine_similarity": 0.46,
      "max":  0.51,
      "num_retrieved_docs": "6
    },
    "groundedness": {
        "overlap": {
            "score": 0.0,
            "common_terms": []
        }
    }
}
```

## API Documentation

The API provides two main endpoints for preprocessing and querying the RAG system.

### GET /preprocess

**Description**: Processes the specified PDF, extracts text using OCR, chunks it into sentence-based segments, and builds a vector store.
**Response**:

```json
{
  "message": "PDF processed with sentence-based chunking",
  "total_chunks": int,
  "chunking_method": "sentence_based",
  "sample_chunks": [
    {
      "id": int,
      "preview": "string...",
      "length": int,
      "sentences": int
    },
    ...
  ]
}
```

**Example** (using hosted URL):

```bash
curl https://multilingual-rag-system.onrender.com/preprocess
```

**Note**: The /preprocess may take some time to process the pdf data, so, you can skip it and directly start using the /query as the data is already processed

### POST /query

**Description**: Accepts a query (in Bangla or English) and returns an answer with source documents and evaluation metrics.
**Request Body**:

```json
{
  "query": "string"
}
```

**Response**:

```json
{
  "answer": "string",
  "sources": [
    {
      "chunk_id": int,
      "preview": "string...",
      "length": int
    },
    ...
  ],
  "context_preview": "string...",
  "evaluation": {
    "relevance": {
      "average_cosine_similarity": float,
      "max_cosine_similarity": float,
      "num_retrieved_docs": int
    },
    "groundedness": {
      "overlap_score": float,
      "common_terms": ["string", ...]
    }
  }
}
```

**Example** (using hosted URL):

```bash
curl -X POST https://multilingual-rag-system.onrender.com/query -H "Content-Type: application/json" -d '{"query": "Who is the main poet of Bengali literature?"}'
```

## Evaluation Metrics

This section presents the performance metrics of the RAG system, evaluating its ability to retrieve relevant documents and generate grounded answers. The system was tested on queries related to the HSC Bangla 1st Paper PDF, with metrics computed for **Relevance** and **Groundedness**.

### Metrics Overview

- **Relevance**: Measures how well retrieved documents match the query using cosine similarity between query and document embeddings.
  - _Average Cosine Similarity_: Average similarity score across retrieved documents (0 to 1, higher is better).
  - _Max Cosine Similarity_: Highest similarity score among retrieved documents.
  - _Number of Retrieved Documents_: Count of documents fetched per query.
- **Groundedness**: Assesses if the generated answer is supported by the retrieved context using word overlap.
  - _Overlap Score_: Ratio of non-trivial answer words found in the context (0 to 1, higher is better).
  - _Common Terms_: Key terms shared between the answer and context.

### Sample Evaluation Results

Below are sample metrics for a query (e.g., "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"):

| Metric                        | Value    |
| ----------------------------- | -------- |
| **Average Cosine Similarity** | 0.556    |
| **Max Cosine Similarity**     | 0.605    |
| **Number of Retrieved Docs**  | 6        |
| **Overlap Score**             | 1.0      |
| **Common Terms**              | `মামাকে` |

### Interpretation

- **Relevance**: The average cosine similarity (0.556) and max cosine similarity (0.605) indicate moderate relevance of retrieved documents. Ongoing improvements aim to increase these scores above 0.8.
- **Groundedness**: An overlap score of 1.0 suggests the answer is fully supported by the context, but the single common term (`মামাকে`) indicates a potentially short answer or limited context overlap. Further refinements are in progress.

## Q&A

### Answer: Text Extraction Method and Challenges

**Method**:

- **pdf2image**: Converts PDF to images (DPI=300).
- **pytesseract**: OCR with `ben+eng` for Bangla/English text, using `--psm 6 --oem 3`.
- **OpenCV**: Preprocesses images (grayscale, thresholding, blur) for better OCR.

**Why**: **pdfplumber** and **PyPDF2** failed due to Bangla Unicode issues. Tesseract OCR handles Bangla well, bypassing font problems.

**Challenges**:

- **Unicode**: pdfplumber/PyPDF2 produced gibberish for Bangla.
- **Layout**: Complex PDF layouts caused jumbled text.
- **OCR Errors**: Tesseract misread Bangla characters; fixed with OpenCV.
- **Sentence Splitting**: Handled mixed Bangla (।) and English (.) punctuation.

**Solutions**:

- Image-based OCR avoided Unicode issues.
- OpenCV preprocessing improved OCR accuracy.

### Answer: Chunking Strategy and Rationale

**Chunking Strategy**: I used **sentence-based chunking** with an 800-character limit and 2-sentence overlap, implemented via `split_into_sentences` (using regex `[।!?\.]\s*` for Bangla/English punctuation) and `create_sentence_boundary_chunks`.

**Why Sentence-Based**:

- **Semantic Coherence**: Sentences maintain meaning better than character or paragraph splits, especially for mixed Bangla-English texts.
- **Context Preservation**: 2-sentence overlap ensures continuity for semantic retrieval.
- **Language Support**: Handles Bangla (।) and English (.) punctuation accurately.
- **Efficiency**: 800-character chunks balance embedding model needs (`Qwen/Qwen3-Embedding-8B`) and MMR retrieval precision in Chroma.

**Why Not Semantic-Based**: I tested semantic-based chunking but found it too time-consuming and resource-intensive due to repeated embedding calculations, so I opted for sentence-based chunking for faster processing and lower resource use.

**Why It Works for Semantic Retrieval**:

- Sentence-level granularity aligns with query semantics, improving relevance (e.g., 0.556 average cosine similarity).
- Overlap maintains context across chunks, aiding MMR retrieval.
- Multilingual compatibility ensures accurate embeddings.

### Answer: Embedding Model

**Model Used**: I used the `Qwen/Qwen3-Embedding-8B` model via SiliconFlow API for embeddings.

**Why Chosen**: Selected for its strong multilingual support, particularly for Bangla, and high performance in capturing semantic nuances across languages. It was more efficient and accurate than alternatives like BERT-based models for Bangla-English texts.

**How It Captures Meaning**: The model generates dense vector representations (embeddings) by encoding text contextually, leveraging transformer architecture to capture semantic relationships, word context, and multilingual patterns, enabling effective similarity comparisons for retrieval (e.g., cosine similarity of 0.556 in tests).

### Answer: Query-Chunk Comparison and Storage

**Comparison Method**: I use **cosine similarity** to compare query embeddings with stored chunk embeddings in the Chroma vector store, implemented via the `evaluate_rag` function. The retriever uses Maximum Marginal Relevance (MMR) with `k=6`, `fetch_k=20`, and `lambda_mult=0.7` to balance relevance and diversity.

**Why Cosine Similarity**: Chosen for its effectiveness in measuring semantic similarity between high-dimensional embeddings, handling multilingual (Bangla-English) text well, and computational efficiency. It normalizes vector magnitude, focusing on semantic alignment (e.g., achieving 0.556 average cosine similarity).

**Storage Setup**: Chroma vector store persists embeddings in the `pdf_chunks` directory, created from sentence-based chunks using `Qwen/Qwen3-Embedding-8B`.

**Why Chroma**: Selected for its simplicity, scalability, and integration with LangChain, enabling fast retrieval and persistent storage for efficient query processing.

### Answer: Meaningful Query-Chunk Comparison and Vague Queries

**Ensuring Meaningful Comparison**:

- **Embeddings**: `Qwen/Qwen3-Embedding-8B` creates semantic vectors for queries and sentence-based chunks (800 chars, 2-sentence overlap).
- **Cosine Similarity**: Measures semantic alignment in Chroma vector store (e.g., 0.556 average similarity).
- **MMR Retrieval**: Balances relevance/diversity (`k=6`, `fetch_k=20`, `lambda_mult=0.7`).
- **Memory**: `ConversationBufferMemory` adds context from prior queries.

**Vague/Missing Context Queries**:

- **Impact**: Lower relevance (e.g., similarity <0.5), generic or inaccurate answers.
- **Mitigation**: MMR provides diverse chunks; memory adds context but struggles with very vague queries.
- **Outcome**: Reduced groundedness, requiring user clarification.

### Answer: Relevance of Results and Improvements

**Relevance**: Results are moderately relevant (e.g., 0.556 average cosine similarity for "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"), but below the target of 0.8, indicating room for improvement.

**Potential Improvements**:

- **Better Chunking**: Optimize sentence-based chunk size (currently 800 chars) or overlap (2 sentences) to capture more context, especially for long Bangla sentences.
- **Better Embedding Model**: Upgrade to a more advanced multilingual model (e.g., a larger Qwen variant) for improved semantic capture of Bangla-English text.
- **Larger Document**: Include more diverse or comprehensive PDFs to provide richer context, reducing gaps for vague queries.
- **Refine MMR**: Adjust MMR parameters (`lambda_mult`, `k`) to prioritize highly relevant chunks.
- **Preprocessing**: Enhance OCR accuracy with advanced image preprocessing to reduce Bangla character errors.

These steps could boost relevance and groundedness for more accurate results.
