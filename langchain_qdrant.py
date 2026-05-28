from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Connect database
client = QdrantClient(
    host="localhost",
    port=6333
)

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# LangChain vector DB
vector_db = Qdrant(
    client=client,
    collection_name="ai_collection",
    embeddings=embedding_model
)

print("LangChain connected with Qdrant!")