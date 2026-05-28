from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect Qdrant
client = QdrantClient(
    host="localhost",
    port=6333
)

# Query
query = "What is used in AI?"

# Query embedding
query_embedding = model.encode(query)

# Similarity search
results = client.search(
    collection_name="ai_collection",
    query_vector=query_embedding.tolist(),
    limit=2
)

print("\nSearch Results:\n")

for result in results:
    print(result.payload)