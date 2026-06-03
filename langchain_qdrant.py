from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from transformers import pipeline

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
    embeddings=embedding_model,
    content_payload_key="text"
)

print("LangChain connected with Qdrant!")

# Retriever — fetch top 3 relevant chunks
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Local LLM via HuggingFace pipeline (no API key needed)
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query
query = "What is used in AI?"
response = qa_chain.invoke({"query": query})

print("\nAnswer:")
print(response["result"])

print("\nSource Chunks:")
for doc in response["source_documents"]:
    print("-", doc.page_content)