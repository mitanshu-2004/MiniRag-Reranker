# Mini RAG + Reranker Sprint

This project implements a small Question-Answering service over a tiny set of industrial and machine safety documents. It starts with a basic similarity search (baseline) and then enhances it with a learned reranker to improve the relevance of retrieved information. The service provides short, extractive answers grounded in the retrieved text, along with citations.

## Features

- **Data Ingestion:** Processes PDF documents, chunks them into sensible pieces, and stores them in an SQLite database.
- **Embeddings:** Uses a local `all-MiniLM-L6-v2` Sentence Transformer model to create vector embeddings, stored in ChromaDB.
- **Baseline Search:** Cosine similarity search to retrieve top-k relevant document chunks.
- **Learned Reranker:** A logistic regression model trained on a few simple features (vector score, keyword score, title match, etc.) to reorder candidate chunks, ensuring better evidence rises to the top.
- **Extractive Answers:** Generates short answers by extracting the most relevant sentence from the top-ranked document chunk.
- **Abstention:** The service abstains from answering if the confidence score of the top-ranked chunk falls below a defined threshold.
- **API:** A FastAPI endpoint for asking questions with different search modes.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd RAG
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    .venv\Scripts\activate  # On Windows
    # source venv/bin/activate # On macOS/Linux
    pip install -r requirements.txt
    ```
    

3.  **Prepare Data and Ingest:**
    Ensure the `industrial-safety-pdfs.zip` and `sources.json` are in the `data/` directory. The `ingest.py` script will automatically unzip the PDFs, chunk them, create embeddings, and train the reranker model.
    ```bash
    python ingest.py
    ```
    This step might take some time as it processes PDFs, converts to vectors and trains the model.

## How to Run the API

1.  **Start the FastAPI server:**
    ```bash
    uvicorn app:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

## API Endpoint

-   **POST `/ask`**
    -   **Request Body:**
        ```json
        {
            "query": "string",
            "top_k": "integer" (default: 5),
            "mode": "string" (options: "baseline", "hybrid", "learned", default: "learned")
        }
        ```
    -   **Response:**
        ```json
        {
            "answer": "string | null",
            "reranker_used": "string",
            "contexts": [
                {
                    "doc_name": "string",
                    "doc_title": "string",
                    "doc_url": "string",
                    "page_num": "integer",
                    "chunk_index": "integer",
                    "score": "float",
                    "content": "string"
                }
            ]
        }
        ```

## Example cURL Requests

### Easy Question (using learned reranker)

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What is the purpose of OSHA 3170?",
           "top_k": 3,
           "mode": "learned"
         }'
```

### Tricky Question (comparing baseline vs. learned reranker)

```bash
# Baseline search
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Explain the differences between PLd and PLe in ISO 13849-1.",
           "top_k": 5,
           "mode": "baseline"
         }'

# Learned reranker
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Explain the differences between PLd and PLe in ISO 13849-1.",
           "top_k": 5,
           "mode": "learned"
         }'
```

## Results Table (Placeholder)

| Question | Baseline Answer | Baseline Contexts | Learned Answer | Learned Contexts | Improvement (Y/N) |
|----------|-----------------|-------------------|----------------|------------------|-------------------|
| 1        |                 |                   |                |                  |                   |
| 2        |                 |                   |                |                  |                   |
| ...      |                 |                   |                |                  |                   |

## What I Learned

This sprint provided valuable insights into building a robust RAG system. A key learning was the significant impact of reranking on the quality of retrieved information. While a baseline vector search provides a good starting point, a learned reranker, even a simple logistic regression model, can dramatically improve the relevance of the top results by incorporating multiple features beyond just vector similarity. This directly leads to more accurate and grounded answers. Additionally, implementing extractive answer generation and an abstention mechanism is crucial for delivering a reliable and user-friendly Q&A service, ensuring that only confident and directly supported answers are provided. The modular design, separating ingestion, search methods, and training, proved essential for managing complexity and allowing for iterative improvements.
