# 📄 Document Q&A (RAG) — Gemini + Streamlit

This project is a **Retrieval-Augmented Generation (RAG) system** that allows you to upload documents (PDF, Markdown, TXT, HTML) and ask questions about them.  
It uses:

- [Google Gemini API](https://ai.google.dev/) for **embeddings** & **answer generation**
- [FAISS](https://github.com/facebookresearch/faiss) for **vector similarity search**
- [Streamlit](https://streamlit.io/) for a simple **web interface**
- Custom `utils.py` for **text extraction, cleaning, and chunking**

---

##Features
 Upload PDF, Markdown, TXT, or HTML documents  
 Extract & clean text automatically  
 Chunk documents into manageable pieces  
 Create embeddings with Gemini  
 Store and search embeddings with FAISS  
 Ask natural language questions about your document  
 Get answers with **sources & supporting quotes**  

---

