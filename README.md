# RAG From Scratch

Production-grade RAG system built on Uber 2024 10-K Annual Report.

## What I Built
End-to-end RAG pipeline from zero.

## Stack
| Component | Tool |
|---|---|
| PDF Parser | pymupdf4llm |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector DB | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Fusion | RRF (Reciprocal Rank Fusion) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.3 70B via Groq |
| UI | Streamlit |

## Modules
- M1: Embeddings and cosine similarity from scratch
- M2: Vector database (ChromaDB)
- M3: PDF parsing with table preservation
- M4: Chunking strategies (fixed, structure-based)
- M5: Naive RAG pipeline
- M6: Evaluation with LLM as judge
- M7: Hybrid search (BM25 + Dense + RRF)
- M8: Cross-encoder reranking
- M9: Query decomposition, HyDE, grounding check
- M10: Streamlit UI

## Results
| Pipeline | Faithfulness | Relevancy |
|---|---|---|
| Naive RAG | 0.30 | 0.30 |
| Hybrid Search | 0.44 | 0.52 |
| Hybrid + Reranker | 0.54 | 0.84 |

## Setup
pip install -r requirements.txt
Add GROQ_API_KEY to .env
streamlit run app.py
