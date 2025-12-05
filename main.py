
from src.data_preprocessing import *
from src.embedding_generator import embed_movies_and_books
import pandas as pd

if __name__ == "__main__":

    # Loading and cleaning raw data
    movie_path = "data/raw/tmdb_5000_movies.csv"
    book_path = "data/raw/books.csv"
    book_tags_path = "data/raw/book_tags.csv"
    tags_path = "data/raw/tags.csv"

    movies_raw, books_raw, book_tags, tags = load_datasets(movie_path, book_path, book_tags_path, tags_path)

    movies_clean, books_clean = preprocess_data(movies_raw, books_raw, book_tags, tags)

    # saving processed data
    save_processed_data(movies_clean, books_clean)

    movie_embeddings, book_embeddings = embed_movies_and_books()