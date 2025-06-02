import streamlit as st
from instructor import INSTRUCTOR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import ChromaVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms import ChatMessage, ChatResponse
import chromadb
import os
import tempfile
import requests

# --- Config ---
CHROMA_PATH = "chroma_store"
VLLM_ENDPOINT = "http://your-vllm-endpoint:8000/generate"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# --- Init Chroma and LlamaIndex ---
chroma_client = chromadb.Client()
vector_store = ChromaVectorStore(chroma_client=chroma_client, collection_name="docs", persist_dir=CHROMA_PATH)

service_context = ServiceContext.from_defaults(embed_model=INSTRUCTOR(), llm=None)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

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
