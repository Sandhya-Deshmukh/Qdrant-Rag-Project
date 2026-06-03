<img width="1328" height="705" alt="image" src="https://github.com/user-attachments/assets/2bde821d-f297-4e5b-bc3c-fea8789695cd" />

<img width="1328" height="705" alt="image" src="https://github.com/user-attachments/assets/1cc626ae-bca4-4443-bbdf-655c73dcf517" />

# Qdrant RAG Project

A Retrieval-Augmented Generation (RAG) pipeline using Qdrant as the vector database and a local HuggingFace LLM (flan-t5-base) — no API key required.

## How to Build & Run

### 1. Prerequisites
- Python 3.x
- Docker

### 2. Clone & install dependencies
```bash
git clone https://github.com/Sandhya-Deshmukh/Qdrant-Rag-Project.git
cd Qdrant-Rag-Project
pip install -r requirements.txt
```

### 3. Start Qdrant (Docker)
```bash
docker run -d --name qdrant_db -p 6333:6333 qdrant/qdrant
```
> If already created: `docker start qdrant_db`

### 4. Ingest documents into Qdrant
```bash
python3 ingest.py
```
Reads `documents/notes.txt`, splits it into chunks, embeds them, and stores them in Qdrant.

### 5. (Optional) Raw similarity search
```bash
python3 search.py
```
Searches Qdrant directly without an LLM — returns raw matching chunks.

### 6. Run the full RAG pipeline (terminal)
```bash
python3 langchain_qdrant.py
```
Retrieves relevant chunks from Qdrant and passes them to `flan-t5-base` to generate an answer.

### 7. Run the Web UI
```bash
streamlit run app.py
```
Opens a browser interface at **http://localhost:8501** where you can type a query and see the LLM answer and source chunks.

## Project Flow
```
notes.txt → ingest.py → Qdrant (vector store)
                              ↓
query → app.py (browser UI) → retriever → flan-t5-base → answer
      → langchain_qdrant.py (terminal)
```

