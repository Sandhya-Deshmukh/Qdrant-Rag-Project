import streamlit as st
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from transformers import pipeline

st.set_page_config(page_title="Qdrant RAG Search", page_icon="🔍")
st.title("Qdrant RAG Search")
st.caption("Ask a question — answers are generated from your documents using a local LLM.")

@st.cache_resource
def load_pipeline():
    client = QdrantClient(host="localhost", port=6333)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Qdrant(
        client=client,
        collection_name="ai_collection",
        embeddings=embedding_model,
        content_payload_key="text"
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

with st.spinner("Loading models..."):
    qa_chain = load_pipeline()

query = st.text_input("Enter your question:", placeholder="e.g. What is used in AI?")

if st.button("Search", disabled=not query):
    with st.spinner("Searching and generating answer..."):
        response = qa_chain.invoke({"query": query})

    st.subheader("Answer")
    st.success(response["result"])

    st.subheader("Source Chunks")
    for i, doc in enumerate(response["source_documents"], 1):
        st.info(f"**Chunk {i}:** {doc.page_content}")
