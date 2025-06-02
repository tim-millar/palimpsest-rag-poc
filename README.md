# Palimpsest AI Monolith (MVP)

This is a single-file Streamlit app that allows you to:
- Upload documents
- Embed with InstructorXL
- Store/retrieve with ChromaDB
- Query using LlamaIndex + remote Mistral 7B (via vLLM)

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configure

- Set `VLLM_ENDPOINT` in `app.py` to your CoreWeave-hosted vLLM API.
- Uses local ChromaDB for vector storage.
