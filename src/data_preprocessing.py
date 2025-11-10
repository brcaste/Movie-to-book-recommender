import os
import re
import pandas as pd
from tqdm import tqdm


def clean_text(text: str) -> str:
    # clean and normalize text by removing punctuations, coverting to lowercase
    # and stripping whitespace

    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]"," ", text)      # remove special characters
    text = re.sub(r"\s+"," ", text)     # remove extra spaces
    return text.strip()


def load_datasets(movie_path: str, book_path: str, book_tags_path: str, tags_path: str):
    # Load the movie and book datasets from CSV files

    print("Loading datasets...")
    movies = pd.read_csv(movie_path)
    books = pd.read_csv(book_path)
    book_tags = pd.read_csv(book_tags_path)
    tags = pd.read_csv(tags_path)

    print(f"Movies loaded: {movies.shape[0]} rows")
    print(f"Books loaded: {books.shape[0]} rows")
    print(f"Book Tags loaded: {book_tags.shape[0]} rows")
    print(f"Tag Names loaded: {tags.shape[0]} rows")
    return movies, books, book_tags, tags


def preprocess_data(movies: pd.DataFrame, books: pd.DataFrame):
    # Clean and prepare movie and book datasets for embedding generation
    tqdm.pandas()

    # keep only relevant columns
    movies= movies[['title', 'overview', "keywords"]].dropna().rename(
        columns={'title': 'movie_title', 'overview': 'movie_overview', 'keywords': 'movie_keywords'}
    )
    books = books[['id','title', 'authors', 'original_title']].dropna().rename(
        columns={
            'id': 'book_id',
            'title': 'book_title',
            'authors': 'book_author',
            'original_title': 'book_description'
        }
    )

    # clean text
    print("Cleaning movie and book text fields...")
    movies['clean_overview'] = movies['movie_overview'].progress_apply(clean_text)
    books['clean_description'] = books['book_description'].progress_apply(clean_text)

    # print("Cleaning Complete.")
    return movies, books


def merge_book_tags(books, book_tags, tags):
    print("Merging book tags...")

    # Attach tag names to book_tags
    book_tags_named = pd.merge(book_tags, tags, on='tag_id', how='left')

    # Merge with books using book_id matching goodreads_book_id
    books_with_tags = pd.merge(
        books,
        book_tags_named,
        left_on='book_id',
        right_on='goodreads_book_id',
        how='left'
    )

    # Group tag names for each book
    books_tag_grouped = books_with_tags.groupby(
        ['book_id', 'book_title', 'book_author', 'clean_description']
    )['tag_name'].apply(lambda x: list(set(x.dropna()))).reset_index()

    # rename for consistency
    books_tag_grouped = books_tag_grouped.rename(columns={'tag_name': 'tags'})

    print("Tags successfully merged")
    return books_tag_grouped


def save_processed_data(movies: pd.DataFrame, books: pd.DataFrame, output_dir: str = "data/processed") -> None:

    # Save cleaned datasets to the processed data folder

    os.makedirs(output_dir, exist_ok=True)
    movie_out = os.path.join(output_dir, "clean_movies.csv")
    book_out = os.path.join(output_dir, "clean_books.csv")

    movies.to_csv(movie_out, index=False)
    books.to_csv(book_out, index=False)

    print(f"Cleaned datasets saved to {output_dir}")
    print(f" - {movie_out}")
    print(f" - {book_out}")