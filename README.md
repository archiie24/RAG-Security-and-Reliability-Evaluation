# Secure Cybersecurity RAG

Production-style multi-document RAG system built over OWASP and NIST security documents.

### Features
- FAISS + BM25 hybrid retrieval, RRF & cross-encoder reranking
- Gemini-based grounded answers with source/page attribution
- Groundedness, unsupported-claim detection & evidence-based abstention
- Self-red-teaming with adversarial/poisoned context attacks
- Baseline vs hardened RAG evaluation
- FastAPI, Streamlit monitoring, SQLite logging & Docker

### Folder Structure

```text
├── app/
│   ├── rag_core.py
│   └── api.py
├── evaluation/
│   ├── redteam.py
│   ├── redteam_cases.json
│   └── reliability.py
├── dashboard.py
├── run_benchmarks.py
├── download_corpus.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── RAG_Security_Reliability_Platform_Final.ipynb
