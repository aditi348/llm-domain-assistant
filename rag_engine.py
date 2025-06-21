import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from llm_interface import generate_answer

MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)

def split_text(text, max_length=500):
    """Split long text into chunks of max_length words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i:i+max_length])
        chunks.append(chunk)
    return chunks

def extract_text_from_pdf(pdf_file):
    """Extract all text from uploaded PDF file."""
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def embed_chunks(chunks):
    """Embed list of text chunks to vectors."""
    embeddings = model.encode(chunks)
    return embeddings

def build_faiss_index(chunks, embeddings, save_path="vector_store"):
    """Build and save FAISS index and chunks."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype(np.float32))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    faiss.write_index(index, os.path.join(save_path, "faiss.index"))
    with open(os.path.join(save_path, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

def load_faiss_index(save_path="vector_store"):
    index = faiss.read_index(os.path.join(save_path, "faiss.index"))
    with open(os.path.join(save_path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def query_rag_system(chat_history, save_path="vector_store", top_k=5):
    index, chunks = load_faiss_index(save_path)

    # Extract latest user question
    latest_user_msg = next((msg for msg in reversed(chat_history) if msg["role"] == "user"), None)
    if latest_user_msg is None:
        return "No user question found.", []

    question = latest_user_msg["content"]

    # Embed question and search
    question_embedding = model.encode([question])
    distances, indices = index.search(np.array(question_embedding).astype(np.float32), top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    context = "\n\n".join(retrieved_chunks)

    # Build system prompt with context
    system_prompt = {
        "role": "system",
        "content": f"You are an AI assistant that answers questions based on the following context:\n\n{context}"
    }

    # Include chat history except previous system prompts
    filtered_history = [msg for msg in chat_history if msg["role"] != "system"]

    messages = [system_prompt] + filtered_history

    answer = generate_answer(messages)
    return answer, retrieved_chunks






