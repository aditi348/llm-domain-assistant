# 🤖 LLM-Domain Assistant with Chat Memory & Embedding Visualization

This project is a lightweight and powerful **Retrieval-Augmented Generation (RAG)** system that lets you upload PDF documents, ask questions about their contents, and get instant, context-aware answers from an LLM — with memory support, summarization, and 2D embedding visualization.

---

## 🔍 Features

- 📄 **Upload Multiple PDFs**
- 🧠 **Ask Questions with Memory** (Multi-turn chat)
- 📚 **Retrieves Context** with FAISS and Sentence Transformers
- 🧾 **Summarizes Entire PDFs**
- 🎯 **Visualize Embeddings** using UMAP to show semantic clustering
- ⚡ Uses **Open Source LLM via OpenRouter API** (Free to use, no paid key needed)
- 🧩 Modular design using `rag_engine.py`, `llm_interface.py`, and `app.py`

---

## 🖼️ Demo Screenshot

![App Screenshot](assets/demo.png) <!-- Optional if you have one -->

---

## 🚀 Getting Started

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/aditi348/llm-domain-assistant.git
cd llm-domain-assistant