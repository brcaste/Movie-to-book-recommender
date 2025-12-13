import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


N_MOVIES = 500           # n movie samples
TOP_K = 3               # top-k books per movie
POS_THRESHOLD = 0.4     # cosine similarity threshold for "positive" label


def load_data(
    movies_path = "data/processed/clean_movies.csv",
    books_path= "data/processed/clean_books.csv",
    movie_emb_path = "models/embeddings/movie_embeddings.npy",
    book_emb_path = "models/embeddings/book_embeddings.npy"
):
    print("Loading processed datasets and embeddings...")

    movies = pd.read_csv(movies_path)
    books = pd.read_csv(books_path)

    movie_embeddings = np.load(movie_emb_path)
    book_embeddings = np.load(book_emb_path)

    print(f"Movies: {movies.shape[0]} rows, embeddings: {movie_embeddings.shape}")
    print(f"Books: {books.shape[0]} rows, embeddings: {book_embeddings.shape}")

    # basic sanity check
    assert movies.shape[0] == movie_embeddings.shape[0], "Movie rows != movie embeddings"
    assert books.shape[0] == book_embeddings.shape[0], "Book rows != book embeddings"

    # add row index as ID if needed
    movies = movies.reset_index().rename(columns={"index": "movie_row"})
    books = books.reset_index().rename(columns={"index": "book_row"})

    return movies, books, movie_embeddings, book_embeddings


def build_labeled_pairs(movies, books, movie_emb, book_emb,
                        n_movies=N_MOVIES, top_k=TOP_K, pos_threshold=POS_THRESHOLD,
                        random_state=42):
    print("Building labeled movie-book pairs...")

    rng = np.random.default_rng(random_state)

    n_movies = min(n_movies, movie_emb.shape[0])
    sampled_movie_indices = rng.choice(movie_emb.shape[0], size=n_movies, replace=False)

    pairs = []

    for i in sampled_movie_indices:
        movie_vec = movie_emb[i : i + 1]
        sims = cosine_similarity(movie_vec, book_emb)[0]

        # indices of top_k most similar books
        top_indices = np.argsort(sims)[::-1][:top_k]

        movie_title = movies.loc[i, "movie_title"] if "movie_title" in movies.columns else movies.loc[i, "title"]

        for j in top_indices:
            book_title = books.loc[j,"book_title"] if "book_title" in books.columns else books.loc[j, "title"]
            sim = float(sims[j])
            label = 1 if sim >= pos_threshold else 0

            pairs.append({
                "movies_row": int(movies.loc[i, "movie_row"]),
                "movie_title": movie_title,
                "book_row": int(books.loc[j, "book_row"]),
                "book_title": book_title,
                "cosine_sim": sim,
                "label": label,
            })

    labeled_df = pd.DataFrame(pairs)
    print(f"Created {len(labeled_df)} labeled pairs"
          f"({n_movies} movies x {top_k} books each)")

    # basic stats
    print("Label distribution:")
    print(labeled_df["label"].value_counts())

    return labeled_df


def save_labeled_pairs(df, output_path="data/processed/labeled_pairs.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Labeled pairs saved to {output_path}")

