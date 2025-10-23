# build_index.py
import hashlib, io, re, os
from typing import List, Optional
from pypdf import PdfReader

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings

load_dotenv() 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding_model = UpstageEmbeddings(model="solar-embedding-1-large")

# ===== 설정 =====
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 80MB로 대폭 축소
FAISS_DIR = ".faiss"       
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
RETRIEVE_K = 6

_embeddings = None
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = embedding_model
    return _embeddings

# ===== PDF → 텍스트 → 청크 =====
def extract_pdf_text(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = re.sub(r"[ \t]+\n", "\n", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        pages.append(txt.strip())
    return pages

def chunk_text(pages: List[str], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    for page in pages:
        if not page:
            continue
        start = 0
        while start < len(page):
            end = min(len(page), start + chunk_size)
            chunk = page[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(page):
                break
            start = max(0, end - overlap)
    return chunks

# ===== FAISS 유틸 =====
def _sanitize_dir(name: str) -> str:
    """파일 시스템 안전한 폴더명으로 변환(ASCII). 원본 해시 8자리로 충돌 방지."""
    base = re.sub(r'[^A-Za-z0-9_.-]+', '-', name).strip('-.')
    if not base:
        base = "subject"
    suffix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{suffix}"

def _subject_path(subject: str) -> str:
    os.makedirs(FAISS_DIR, exist_ok=True)
    safe = _sanitize_dir(subject)
    path = os.path.join(FAISS_DIR, safe)
    os.makedirs(path, exist_ok=True)  
    return path

def load_faiss(subject: str, embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
    path = _subject_path(subject)
    if os.path.isdir(path):
        try:
            return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            
            return None
    return None

def save_faiss(subject: str, vectordb: FAISS):
    path = _subject_path(subject)
    os.makedirs(path, exist_ok=True)  
    vectordb.save_local(path)         


def upsert_to_faiss(subject: str, title: str, chunks: List[str]) -> FAISS:
    """
    문서 청크를 해당 subject 인덱스에 업서트하고 디스크에 저장.
    subject별로 별도 인덱스를 유지합니다.
    """
    embeddings = get_embeddings()
    docs = [
        Document(
            page_content=ch,
            metadata={"subject": subject, "title": title, "chunk_id": f"{title}_{i+1}"}
        )
        for i, ch in enumerate(chunks)
    ]
    vectordb = load_faiss(subject, embeddings)
    if vectordb is None:
        vectordb = FAISS.from_documents(docs, embeddings)
    else:
        vectordb.add_documents(docs)
    save_faiss(subject, vectordb)
    return vectordb

def build_context(vectordb: FAISS, query: str, k=RETRIEVE_K) -> str:
    
    docs = vectordb.similarity_search(query, k=k)
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

def context_from_pdf_bytes(pdf_bytes: bytes, subject: str, title: str) -> str:
    pages = extract_pdf_text(pdf_bytes)
    if not any(pages):
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다.")
    chunks = chunk_text(pages)
    if not chunks:
        raise ValueError("텍스트 청크가 비었습니다.")
    vectordb = upsert_to_faiss(subject=subject, title=title, chunks=chunks)
    return build_context(vectordb, query=title or subject, k=RETRIEVE_K)

