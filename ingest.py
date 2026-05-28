from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Read document
with open("documents/notes.txt", "r") as file:
    text = file.read()

print("\nOriginal Text:\n")
print(text)

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print("\nChunks:\n")

for chunk in chunks:
    print(chunk)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(chunks)

# Connect Qdrant
client = QdrantClient(
    host="localhost",
    port=6333
)

# Create collection
client.recreate_collection(
    collection_name="ai_collection",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# Prepare vectors
points = []

for i, embedding in enumerate(embeddings):

    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": chunks[i]
            }
        )
    )

# Insert vectors
client.upsert(
    collection_name="ai_collection",
    points=points
)

print("\nData inserted successfully into Qdrant!")