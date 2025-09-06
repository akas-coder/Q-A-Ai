# app.py
import streamlit as st
import os
from utils import extract_text_from_pdf, clean_text, chunk_text, extract_text_from_markdown
import numpy as np
import faiss
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time
import pickle
import google.generativeai as genai

# ----------------------------
# Load API key
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Set GEMINI_API_KEY in your .env file")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "models/embedding-001"
GEN_MODEL = "gemini-1.5-flash"
st.title("📄 Document Q&A (RAG) — Gemini Prototype")

# ----------------------------
# File upload
# ----------------------------
uploaded = st.file_uploader(
    "Upload PDF / Markdown / Text / HTML", type=["pdf", "md", "markdown", "txt", "html"]
)

# ----------------------------
# Initialize session state
# ----------------------------
if "chunks" not in st.session_state: st.session_state["chunks"] = None
if "embeds" not in st.session_state: st.session_state["embeds"] = None
if "index" not in st.session_state: st.session_state["index"] = None
if "meta" not in st.session_state: st.session_state["meta"] = None

# ----------------------------
# Utility: Get embeddings with rate-limit handling and caching
# ----------------------------
def get_embeddings_safe(texts, batch_size=8):
    embeds = []
    cache_file = "embeddings_cache.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_to_request = [t for t in batch if t not in cache]
        if not batch_to_request:
            embeds.extend([cache[t] for t in batch])
            continue

        success = False
        while not success:
            try:
                resp = genai.embed_content(
                    model=EMBED_MODEL,
                    content=batch_to_request,
                )
                if isinstance(resp, dict) and "embedding" in resp:
                    # single input case
                    embeddings = [resp["embedding"]]
                else:
                    # multi input case
                    embeddings = [d["embedding"] for d in resp["embedding"]]

                for t, emb in zip(batch_to_request, embeddings):
                    cache[t] = emb
                    embeds.append(emb)
                success = True
            except Exception as e:
                st.warning(f"Embedding error: {e}. Retrying in 10s...")
                time.sleep(10)

    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    return np.array(embeds).astype("float32")

# ----------------------------
# Process uploaded file
# ----------------------------
if uploaded:
    bytes_data = uploaded.read()
    fname = uploaded.name

    # Extract text
    if fname.lower().endswith(".pdf"):
        with open(f"tmp_{fname}", "wb") as f: f.write(bytes_data)
        raw_text = extract_text_from_pdf(f"tmp_{fname}")
        os.remove(f"tmp_{fname}")
    elif fname.lower().endswith((".md", ".markdown")):
        raw_text = extract_text_from_markdown(bytes_data.decode("utf-8"))
    elif fname.lower().endswith((".html", ".htm")):
        soup = BeautifulSoup(bytes_data.decode("utf-8"), "html.parser")
        raw_text = soup.get_text(separator="\n")
    else:
        raw_text = bytes_data.decode("utf-8")

    raw_text = clean_text(raw_text)
    st.write(f"Document length (chars): {len(raw_text)}")

    # Chunk & embed only once
    if st.session_state["chunks"] is None:
        st.session_state["chunks"] = chunk_text(raw_text, max_tokens=400, overlap=50)
        st.write(f"Created {len(st.session_state['chunks'])} chunks")

        with st.spinner("Creating embeddings..."):
            texts = [c["text"] for c in st.session_state["chunks"]]
            st.session_state["embeds"] = get_embeddings_safe(texts)

        # Build FAISS index
        dim = len(st.session_state["embeds"][0])
        st.session_state["index"] = faiss.IndexFlatL2(dim)
        st.session_state["index"].add(st.session_state["embeds"])
        st.success("Index created ✅")

        # Metadata
        st.session_state["meta"] = {
            i: {"text": st.session_state["chunks"][i]["text"], "source": fname}
            for i in range(len(st.session_state["chunks"]))
        }

# ----------------------------
# Query input
# ----------------------------
query = st.text_input("🔎 Ask a question about the document")
k = st.slider("Top-k retrievals", 1, 10, 3)

if st.button("Search") and query.strip():
    if st.session_state["index"] is None:
        st.warning("Please upload a document first!")
    else:
        try:
            q_emb = genai.embed_content(model=EMBED_MODEL, content=query)["embedding"]
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()

        q_vec = np.array(q_emb).astype("float32").reshape(1, -1)
        D, I = st.session_state["index"].search(q_vec, k)
        indices = I[0].tolist()
        scores = D[0].tolist()

        # Prepare context
        context_texts = []
        for idx, score in zip(indices, scores):
            meta = st.session_state.get("meta", {}).get(idx)
            if meta:
                context_texts.append(
                    f"---\nSource chunk id: {idx}\n{meta['text']}\n"
                )
        combined_context = "\n\n".join(context_texts)

        # Prompt
        prompt = f"""
You are a helpful assistant. Use the context below to answer the user's question. 
If the answer is not in the context, say 'I don't know'.

Context:
{combined_context}

Question: {query}

Provide:
1) A short answer (2–5 sentences)
2) A short list of sources (chunk ids) used
3) Quote the exact sentence(s) from the context that support your answer
"""

        try:
            model = genai.GenerativeModel(GEN_MODEL)
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            st.error(f"Gemini API error: {e}")
            st.stop()

        # Display results
        st.subheader("💡 Answer")
        st.write(answer)

        st.subheader("📌 Retrieved chunks")
        for idx, score in zip(indices, scores):
            meta = st.session_state.get("meta", {}).get(idx)
            if meta:
                st.markdown(
                    f"*Chunk {idx}* (score {score:.4f}) — source: {meta['source']}"
                )
                snippet = meta["text"]
                st.write(snippet[:600] + ("..." if len(snippet) > 600 else ""))
