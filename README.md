# Secure Cybersecurity RAG

Production-style multi-document RAG system over OWASP and NIST cybersecurity documents, extended with retrieval, reliability, and security evaluation.

## Architecture

```text
Documents → Chunking → Embeddings
                     ↓
              FAISS + BM25
                     ↓
                 RRF Fusion
                     ↓
            Cross-Encoder Rerank
                     ↓
              Evidence Selection
                     ↓
              Gemini Generation
                     ↓
        NLI Grounding + Abstention
                     ↓
          Answer + Source Attribution
```

## Key Features

- Hybrid **FAISS + BM25** retrieval with RRF fusion and cross-encoder reranking
- Gemini-based answers with **source/page attribution**
- Sentence-level **NLI groundedness** and unsupported-claim detection
- **Evidence-based abstention** for insufficient retrieval
- Adversarial/poisoned-context **self-red-teaming**
- Baseline vs hardened RAG evaluation
- **FastAPI + Streamlit + SQLite + Docker** production-style setup

## Evaluation

- 3-document cybersecurity corpus
- 621 pages → 2,169 chunks
- 384-dimensional SentenceTransformer embeddings
- Retrieval benchmark: **100% Hit@5**, **0.712 average top-1 similarity**
- Query-level monitoring of groundedness, evidence score, abstention, unsupported claims, and latency

## Results

### RAG Answer & Attribution

![RAG Answer](screenshots/rag_answer.png)

### Security & Reliability Dashboard

![Dashboard Overview](screenshots/dashboard_overview.png)

### Retrieval & Grounding

![Retrieval and Grounding Quality](screenshots/retrieval_grounding_quality.png)

### Reliability & Safety

![Query Reliability](screenshots/query_reliability.png)

![Safety Behavior](screenshots/query_latency_safety.png)

### Red-Team Evaluation

![Red Team Results](screenshots/red_team_results.png)

## Project Structure

```text
├── app/
│   ├── rag_core.py
│   └── api.py
├── evaluation/
│   ├── redteam.py
│   ├── redteam_cases.json
│   └── reliability.py
├── screenshots/
├── dashboard.py
├── run_benchmarks.py
├── download_corpus.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── RAG_Security_Reliability_Platform_Final.ipynb
```

## Run

```bash
pip install -r requirements.txt
python download_corpus.py
python run_benchmarks.py
streamlit run dashboard.py
```

For the API:

```bash
uvicorn app.api:app --reload
```

Docker:

```bash
docker compose up --build
```
