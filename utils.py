# utils.py
import fitz  # PyMuPDF
import re
from typing import List, Dict
import tiktoken
import markdown2
from bs4 import BeautifulSoup

# Tokenizer (OpenAI compatible)
encoder = tiktoken.get_encoding("cl100k_base")

def extract_text_from_pdf(path: str) -> str:
    """Extract all text from a PDF file"""
    doc = fitz.open(path)
    text_pages = []
    for page in doc:
        text_pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(text_pages)

def extract_text_from_markdown(md: str) -> str:
    """Convert Markdown to plain text"""
    html = markdown2.markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")

def clean_text(s: str) -> str:
    """Normalize whitespace and clean text"""
    s = s.replace("\r\n", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    return len(encoder.encode(text))

def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Split text into chunks with token limits.
    Returns list of dicts: {'id', 'text', 'tokens'}.
    """
    words = text.split()
    chunks = []
    cur = []
    idx = 0
    i = 0

    while i < len(words):
        cur.append(words[i])
        cur_tok = count_tokens(" ".join(cur))
        if cur_tok >= max_tokens:
            chunk_str = " ".join(cur)
            chunks.append({"id": idx, "text": chunk_str, "tokens": cur_tok})
            idx += 1
            cur = cur[-overlap:] if overlap > 0 else []
        i += 1

    if cur:
        chunk_str = " ".join(cur)
        chunks.append({"id": idx, "text": chunk_str, "tokens": count_tokens(chunk_str)})

    return chunks
