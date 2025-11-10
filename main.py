
from src.data_preprocessing import *

if __name__ == "__main__":
    # Example paths (adjust these to your local files)
    movie_path = "data/raw/tmdb_5000_movies.csv"
    book_path = "data/raw/books.csv"
    book_tags_path = "data/raw/book_tags.csv"
    tags_path = "data/raw/tags.csv"

    movies, books, book_tag, tags = load_datasets(movie_path, book_path, book_tags_path, tags_path)
    movies, books = preprocess_data(movies, books)
    books = merge_book_tags(books, book_tag, tags)
    save_processed_data(movies, books)