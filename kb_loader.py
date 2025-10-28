# kb_loader.py
import os
import csv
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# --------- Config ----------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

MODEL = SentenceTransformer(EMBED_MODEL_NAME)

# --------- Helpers ---------
def _ensure_dirs():
    os.makedirs("sources", exist_ok=True)

def _to_numpy32(x):
    arr = np.asarray(x, dtype="float32")
    return arr

def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# --------- CSV Ingestion (Structured KB) ---------
def ingest_csv(
    path: str,
    index_path: str = "kb_index.faiss",
    meta_path: str = "metadata.pkl"
) -> Tuple[str, str]:
    """
    Reads a CSV with columns:
    id, theme, sample_questions, short_answer, detailed_answer, risk_level, source
    Builds embeddings and saves FAISS index + metadata.
    """
    _ensure_dirs()
    entries: List[Dict[str, Any]] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_questions = [q.strip() for q in (row.get("sample_questions") or "").split(";") if q.strip()]
            short_answer = (row.get("short_answer") or "").strip()
            detailed_answer = (row.get("detailed_answer") or "").strip()
            theme = (row.get("theme") or "").strip()
            source = (row.get("source") or "").strip()
            id_ = (row.get("id") or f"row-{len(entries)+1}").strip()
            risk = (row.get("risk_level") or "Info").strip()

            embed_text = f"{theme} || {' ; '.join(sample_questions)} || {short_answer}"

            entries.append({
                "id": id_,
                "theme": theme,
                "sample_questions": sample_questions,
                "short_answer": short_answer,
                "detailed_answer": detailed_answer,
                "risk_level": risk,
                "source": source,
                "text": embed_text,
                "page": None,
                "chunk_id": id_,
            })

    if not entries:
        raise RuntimeError("CSV appears empty or has no valid rows.")

    texts = [e["text"] for e in entries]
    print(f"Embedding {len(texts)} CSV rows using {EMBED_MODEL_NAME}...")
    embeddings = MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = _to_numpy32(embeddings)
    faiss.normalize_L2(embeddings)

    index = _build_faiss_index(embeddings)
    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(entries, f)

    print("✅ CSV ingestion complete. Index vectors:", index.ntotal)
    return index_path, meta_path

# --------- PDF Ingestion (Chunking) ---------
def read_pdf_text(path: str):
    doc = fitz.open(path)
    pages = []
    for pageno, page in enumerate(doc, start=1):
        pages.append((pageno, page.get_text()))
    return pages

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    import re
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) <= chunk_size:
            cur = cur + " " + s if cur else s
        else:
            if cur:
                chunks.append(cur.strip())
            cur = s
    if cur:
        chunks.append(cur.strip())

    if overlap and len(chunks) > 1:
        merged = []
        for i in range(len(chunks)):
            start = max(0, i-1)
            piece = " ".join(chunks[start:i+1])
            merged.append(piece)
        return merged
    return chunks

def build_index(
    source_folder: str = "sources",
    index_path: str = "kb_index.faiss",
    meta_path: str = "metadata.pkl"
) -> Tuple[str, str]:
    """
    Reads all PDFs under /sources, chunks text, embeds, and builds FAISS index.
    """
    _ensure_dirs()
    docs = []

    for fn in sorted(os.listdir(source_folder)):
        if not fn.lower().endswith(".pdf"):
            continue
        full = os.path.join(source_folder, fn)
        pages = read_pdf_text(full)
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            chunks = chunk_text(page_text)
            for i, c in enumerate(chunks):
                chunk_id = f"{fn}::p{page_num}::c{i}"
                docs.append({
                    "id": chunk_id,
                    "source": fn,
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "text": c,
                    "short_answer": "",
                    "detailed_answer": c,
                    "risk_level": "Info",
                })

    if not docs:
        raise RuntimeError("No text extracted from PDFs in /sources.")

    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} PDF chunks using {EMBED_MODEL_NAME}...")
    embeddings = MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = _to_numpy32(embeddings)
    faiss.normalize_L2(embeddings)

    index = _build_faiss_index(embeddings)
    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(docs, f)

    print("✅ PDF index built:", index.ntotal, "vectors.")
    return index_path, meta_path

# --------- Load / Retrieve ----------
def load_index(index_path: str = "kb_index.faiss", meta_path: str = "metadata.pkl"):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or metadata not found. Build or import first.")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve(query: str, index: faiss.IndexFlatIP = None, metadata: List[Dict[str, Any]] = None, top_k: int = 4):
    """
    Returns list of results: {score, id, short_answer, detailed_answer, risk_level, source, text, page, chunk_id}
    """
    if index is None or metadata is None:
        return []

    q_emb = MODEL.encode([query], convert_to_numpy=True)
    q_emb = _to_numpy32(q_emb)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = metadata[idx]
        results.append({
            "score": float(score),
            "id": m.get("id") or m.get("chunk_id"),
            "short_answer": m.get("short_answer", ""),
            "detailed_answer": m.get("detailed_answer", m.get("text", "")),
            "risk_level": m.get("risk_level", m.get("risk", "Info")),
            "source": m.get("source", "unknown"),
            "text": m.get("text", ""),
            "page": m.get("page"),
            "chunk_id": m.get("chunk_id"),
        })
    return results
