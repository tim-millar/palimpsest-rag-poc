import streamlit as st
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from InstructorEmbedding import INSTRUCTOR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
import chromadb
import os
import tempfile
import requests

# --- Config ---
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
# CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PORT_RAW = os.getenv("CHROMA_PORT", "8000")
print(f"CHROMA_PORT raw value: {CHROMA_PORT_RAW!r}")  # For debug
CHROMA_PORT = int(CHROMA_PORT_RAW)
print(f"CHROMA_PORT: {CHROMA_PORT!r}")  # For debug
VLLM_ENDPOINT = "http://your-vllm-endpoint:8000/generate"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# --- Init Chroma and LlamaIndex ---
# chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT_RAW)
vector_store = ChromaVectorStore(host=CHROMA_HOST, port=CHROMA_PORT, collection_name="docs")

embed_model = HuggingFaceEmbedding(
    model_name="hkunlp/instructor-xl",
    embed_batch_size=8,
    normalize=True
)
# Settings.embed_model = INSTRUCTOR()
Settings.embed_model = embed_model
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# --- Streamlit UI ---
st.title("Palimpsest AI: RAG Demo")

menu = st.sidebar.radio("Navigation", ["Chat", "Upload Documents"])

if menu == "Upload Documents":
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if st.button("Ingest Documents") and uploaded_files:
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in uploaded_files:
                file_path = os.path.join(tmpdir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            docs = SimpleDirectoryReader(tmpdir).load_data()
            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents(docs)
            index.insert_nodes(nodes)
            st.success("Documents ingested and indexed!")

elif menu == "Chat":
    st.header("Ask a question about your documents")
    query = st.text_input("Your question:")
    if query:
        retriever = index.as_retriever(similarity_top_k=3)
        context_nodes = retriever.retrieve(query)
        context = "\n\n".join([n.text for n in context_nodes])

        # Send to vLLM
        response = requests.post(VLLM_ENDPOINT, json={
            "model": MODEL_NAME,
            "prompt": f"Context:\n{context}\n\nQuestion: {query}",
            "max_tokens": 512
        })

        if response.ok:
            result = response.json()
            st.markdown("### Answer")
            st.write(result.get("text", "No response"))
        else:
            st.error("Failed to get response from LLM API.")
