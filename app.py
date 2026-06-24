
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
import numpy as np
from groq import Groq
import pymupdf4llm

st.set_page_config(page_title="Uber 10-K RAG", layout="wide")
st.title("Uber 10-K Intelligence")
st.caption("Ask anything about Uber 2024 Annual Report")

import os
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

@st.cache_resource
def load_pipeline():
    md_text = pymupdf4llm.to_markdown("/content/RAG POC SAMPLE DOC.pdf")
    
    def chunk_text(text, chunk_size=400, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
    
    chunks = chunk_text(md_text)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="uber_10k")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return model, collection, bm25, reranker, chunks

model, collection, bm25, reranker, chunks = load_pipeline()
groq_client = Groq(api_key=GROQ_KEY)

def ask(question):
    q_vec = model.encode(question).tolist()
    dense_results = collection.query(query_embeddings=[q_vec], n_results=20)
    dense_ids = dense_results["ids"][0]
    tokenized_q = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_q)
    bm25_top20_idx = np.argsort(bm25_scores)[::-1][:20]
    bm25_ids = [f"chunk_{i}" for i in bm25_top20_idx]
    rrf_scores = {}
    k = 60
    for rank, cid in enumerate(dense_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1/(k+rank+1)
    for rank, cid in enumerate(bm25_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1/(k+rank+1)
    top20_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:20]
    top20_chunks = [chunks[int(cid.split("_")[1])] for cid in top20_ids]
    pairs = [[question, chunk] for chunk in top20_chunks]
    rerank_scores = reranker.predict(pairs)
    top5_idx = np.argsort(rerank_scores)[::-1][:5]
    found_chunks = [top20_chunks[i] for i in top5_idx]
    context = "\n\n".join([c[:300] for c in found_chunks[:3]])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer only from context. If not found say not found."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content, found_chunks

question = st.text_input("Ask:", placeholder="What was Uber revenue in 2024?")
if question:
    with st.spinner("Searching..."):
        answer, found_chunks = ask(question)
    st.subheader("Answer")
    st.write(answer)
    with st.expander("View chunks"):
        for i, chunk in enumerate(found_chunks):
            st.markdown(f"**Chunk {i+1}**")
            st.text(chunk[:300])
            st.divider()
