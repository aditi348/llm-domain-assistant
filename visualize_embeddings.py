import os
import pickle
import numpy as np
import faiss
import umap
import matplotlib.pyplot as plt

# Load FAISS index and chunks
def load_index_and_chunks(path="vector_store"):
    index = faiss.read_index(os.path.join(path, "faiss.index"))
    with open(os.path.join(path, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def visualize_embeddings():
    index, chunks = load_index_and_chunks()
    embeddings = index.reconstruct_n(0, index.ntotal)

    # Reduce dimensions
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding_2d = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7)
    plt.title("UMAP Projection of Document Chunk Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_embeddings()
