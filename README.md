# Cybersecurity Multi-Document RAG System

A Retrieval-Augmented Generation (RAG) system for answering cybersecurity-related questions using trusted security documentation from **OWASP** and **NIST**. The system retrieves relevant information from multiple PDF documents and uses an LLM to generate grounded responses with source and page references.

## Overview

Large cybersecurity documents can contain hundreds of pages, making it difficult to locate relevant information quickly. This project combines semantic search with large language models to provide concise, context-aware answers while reducing reliance on the LLM's internal knowledge.

The system follows a **retrieve-then-generate** architecture:

**PDF Documents → Text Extraction & Cleaning → Chunking → Embeddings → FAISS Retrieval → Context Construction → Gemini LLM → Grounded Answer**

## Key Features

* Supports retrieval across multiple cybersecurity PDF documents.
* Extracts and cleans text while preserving document and page metadata.
* Splits documents into overlapping chunks to maintain contextual continuity.
* Generates **384-dimensional embeddings** using SentenceTransformers.
* Uses **FAISS** for efficient vector similarity search.
* Applies **cosine similarity** to identify the most relevant document chunks.
* Provides retrieved source information and page references with generated answers.
* Uses retrieved document context to reduce unsupported or hallucinated responses.

## Technology Stack

* **Python**
* **SentenceTransformers** – text embedding generation
* **FAISS** – vector indexing and similarity search
* **Gemini 2.5 Flash-Lite** – answer generation
* **PyPDF** – PDF text extraction
* **NumPy** – vector and numerical operations

## RAG Pipeline

### 1. Document Ingestion

Cybersecurity documents from sources such as OWASP and NIST are loaded from PDF files. Text is extracted page-by-page so that page-level metadata can be retained for later source attribution.

### 2. Text Cleaning

Extracted text is cleaned to remove unnecessary formatting and improve the quality of the text supplied to the embedding model.

### 3. Chunking

The extracted text is divided into smaller overlapping chunks. Overlap helps preserve context when important information spans the boundary between two chunks.

Each chunk retains metadata such as its document name and page number.

### 4. Embedding Generation

Each text chunk is converted into a numerical vector using a SentenceTransformers embedding model. The resulting embeddings have **384 dimensions** and represent the semantic meaning of the corresponding text.

### 5. FAISS Indexing

The embeddings are stored in a FAISS vector index. When a user submits a query, the query is embedded using the same embedding model and compared against the indexed document vectors.

Cosine similarity is used to identify the chunks that are semantically closest to the query.

### 6. Context Retrieval

The highest-ranked chunks are retrieved and combined with their metadata to construct the context supplied to the language model.

This allows the LLM to answer using information retrieved from the cybersecurity documents rather than relying only on its pretrained knowledge.

### 7. Grounded Answer Generation

The retrieved context is passed to **Gemini 2.5 Flash-Lite** along with the user's question. The model generates an answer based on the supplied context.

The response also includes the relevant **document and page references**, making the answer easier to verify.

## Example Workflow

**User Query:**

> What are the main risks associated with SQL injection?

The system:

1. Converts the query into an embedding.
2. Searches the FAISS index using cosine similarity.
3. Retrieves the most relevant chunks from the OWASP/NIST documents.
4. Passes the retrieved chunks and query to Gemini.
5. Generates a concise answer grounded in the retrieved material.
6. Provides the corresponding document and page references.

## Why RAG?

A standard LLM may contain general cybersecurity knowledge, but it may not reliably answer questions according to a specific set of reference documents.

RAG addresses this by retrieving relevant information from the selected knowledge base at query time. This provides:

* **Document-grounded responses**
* **Reduced hallucination risk**
* **Up-to-date or domain-specific reference material**
* **Traceability through source and page references**

## Limitations

* Retrieval quality depends on the quality of document extraction, chunking, and embeddings.
* Answers are limited by the information successfully retrieved from the indexed documents.
* Poorly phrased or highly ambiguous queries may retrieve less relevant context.
* The system does not guarantee that every generated statement is factually correct, so retrieved sources should be used for verification.

## Future Improvements

Potential extensions include:

* Hybrid keyword + semantic retrieval.
* Reranking retrieved chunks using a cross-encoder.
* Improved chunking strategies based on document structure.
* Retrieval and answer-quality evaluation using benchmark questions.
* Support for additional cybersecurity standards and documentation.
* Conversation history for multi-turn question answering.

## Author

**Archita**
