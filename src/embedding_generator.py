import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    print(f"Loading SentenceTransformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model Loaded.")
    return model


def generate_embeddings(text_list, model: SentenceTransformer):
    # ensure everything is a clean string
    cleaned = []
    for t in text_list:
        if t is None or (isinstance(t,float) and np.isnan(t)):
            cleaned.append("")
        else:
            cleaned.append(str(t))

    print("Generating embeddings...")
    embeddings = model.encode(
        cleaned,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print("Embeddings generated.")
    return embeddings


def embed_movies_and_books(
    movies_path: str = "data/processed/clean_movies.csv",
    books_path: str = "data/processed/clean_books.csv",
    model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = "models/embeddings"
):
    print("Loading processed datasets...")
    movies = pd.read_csv(movies_path)
    books = pd.read_csv(books_path)

    if "clean_overview" in movies.columns:
        movie_texts = movies["clean_overview"].fillna("").astype(str).tolist()
    else:
        raise KeyError("Movie file must contain 'clean_overview' column")

    if "combined_text" in books.columns:
        books_texts = books["combined_text"].fillna("").astype(str).tolist()
    else:
        raise KeyError("Books file must contain 'combined_text' column")

    # load model and generate embeddings
    model = load_embedding_model(model_name)
    movie_embeddings = generate_embeddings(movie_texts, model)
    book_embeddings = generate_embeddings(books_texts, model)

    # save embeddings
    os.makedirs(output_dir,exist_ok=True)
    movie_emb_path = os.path.join(output_dir, "movie_embeddings.npy")
    book_emb_path = os.path.join(output_dir, "book_embeddings.npy")

    np.save(movie_emb_path, movie_embeddings)
    np.save(book_emb_path,book_embeddings)

    print(f" Movie embeddings saved to: {movie_emb_path}")
    print(f" Book embeddings saved to: {book_emb_path}")

    return movie_embeddings, book_embeddings
