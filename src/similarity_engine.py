import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

# Configurations
PROCESSED_DATA_DIR = "data/processed"
EMBEDDED_DATA_DIR = "models/embeddings"
MOVIES_CSV = os.path.join(PROCESSED_DATA_DIR, "clean_movies.csv")
BOOKS_CSV = os.path.join(PROCESSED_DATA_DIR, "clean_books.csv")

MOVIE_EMBEDDINGS_PATH = os.path.join(EMBEDDED_DATA_DIR, "movie_embeddings.npy")
BOOK_EMBEDDINGS_PATH = os.path.join(EMBEDDED_DATA_DIR, "book_embeddings.npy")

# Loading helpers


def load_data_and_embeddings(
    movie_path: str = MOVIES_CSV,
    books_path: str = BOOKS_CSV,
    movie_emb_path: str = MOVIE_EMBEDDINGS_PATH,
    book_emb_path: str = BOOK_EMBEDDINGS_PATH
):
    print("Loading processed data and embeddings...")
    movies_df = pd.read_csv(movie_path)
    books_df = pd.read_csv(books_path)

    movie_embeddings = np.load(movie_emb_path)
    book_embeddings = np.load(book_emb_path)

    # Testing for proper loading
    assert len(movies_df) == movie_embeddings.shape[0], \
        "Number of movie rows and movie embeddings must match"
    assert len(books_df) == book_embeddings.shape[0], \
        "Number of book rows and book embeddings must match"

    print(f"Movies: {len(movies_df)} | Books: {len(books_df)} ")
    return movies_df, books_df, movie_embeddings, book_embeddings


def find_movie_index(movies_df: pd.DataFrame, movie_title: str) -> Optional[int]:
    title_lower = movie_title.strip().lower()

    # Exact case-insensitive match
    exact_matches = movies_df[
        movies_df["movie_title"].str.lower() == title_lower
    ]
    if not exact_matches.empty:
        return exact_matches.index[0]

    # Fallback: partial match (contains)
    contains_matches = movies_df[
        movies_df["movie_title"].str.lower().str.contains(title_lower, na=False)
    ]
    if not contains_matches.empty:
        return contains_matches.index[0]

    return None


def recommend_books_for_movie(
        movie_title: str,
        movies_df: pd.DataFrame,
        books_df: pd.DataFrame,
        movie_embeddings: np.ndarray,
        book_embeddings: np.ndarray,
        top_n: int= 3,
        min_similarity: float = 0.0,
) -> List[Dict]:

    movie_idx = find_movie_index(movies_df, movie_title)
    if movie_idx is None:
        raise ValueError(f"Movie '{movie_title}' not found in dataset")

    print(f"Found movie: {movies_df.loc[movie_idx, 'movie_title']}(index{movie_idx}")
    movie_vec = movie_embeddings[movie_idx].reshape(1,-1)

    # Compute cosine similarity between this movie and all books
    sims = cosine_similarity(movie_vec, book_embeddings)[0]

    # Get sorted indices of books by similarity (decending)
    ranked_indices = np.argsort(-sims)

    # Apply threshold and pick top_n
    recommendations = []
    for idx in ranked_indices:
        score = float(sims[idx])
        if score < min_similarity:
            continue

        rec= {
            "book_title": books_df.loc[idx, "book_title"],
            "book_author": books_df.loc[idx, "book_author"],
            "similarity": round(score, 4),
        }
        recommendations.append(rec)

        if len(recommendations) >= top_n:
            break
    return recommendations


def recommend(
        movie_title: str,
        top_n: int = 5,
        min_similarity: float = 0.0,
        movies_path: str = MOVIES_CSV,
        books_path: str = BOOKS_CSV,
        movie_emb_path: str = MOVIE_EMBEDDINGS_PATH,
        book_emb_path: str = BOOK_EMBEDDINGS_PATH,
) -> List[Dict]:

    movies_df, books_df, movie_emb, book_emb = load_data_and_embeddings(
        movies_path, books_path, movie_emb_path, book_emb_path
    )
    recs = recommend_books_for_movie(
        movie_title=movie_title,
        movies_df=movies_df,
        books_df=books_df,
        movie_embeddings=movie_emb,
        book_embeddings=book_emb,
        top_n=top_n,
        min_similarity=min_similarity
    )
    return recs
