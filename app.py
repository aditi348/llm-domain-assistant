import streamlit as st
from rag_engine import (
    extract_text_from_pdf, split_text, embed_chunks, build_faiss_index, query_rag_system
)
from llm_interface import generate_answer

st.title("📚 RAG Multi-Document PDF Chatbot with Memory & Summarization")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "index_built" not in st.session_state:
    st.session_state.index_built = False

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_chunks = []
    for pdf_file in uploaded_files:
        text = extract_text_from_pdf(pdf_file)
        chunks = split_text(text)
        all_chunks.extend(chunks)

    embeddings = embed_chunks(all_chunks)
    build_faiss_index(all_chunks, embeddings)
    st.session_state.index_built = True
    st.success(f"Processed {len(uploaded_files)} files, total chunks: {len(all_chunks)}")

    if st.button("Summarize All Documents"):
        full_text = "\n\n".join(all_chunks)
        summary_prompt = f"Summarize the following documents in simple terms:\n\n{full_text}\n\nSummary:"
        summary = generate_answer([{"role": "user", "content": summary_prompt}])
        st.markdown("### 📝 Summary")
        st.write(summary)

if st.session_state.index_built:
    user_question = st.text_input("Ask a question about the documents:")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        answer, retrieved_chunks = query_rag_system(st.session_state.chat_history)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        st.markdown("### 🤖 Answer:")
        st.write(answer)

        with st.expander("📄 Show Retrieved Context"):
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.code(chunk, language="text")

