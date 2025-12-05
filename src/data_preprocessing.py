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


def load_datasets(movie_path: str, book_path: str, book_tags_path: str, tags_path: str)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load the movie and book datasets from CSV files

    print("Loading datasets...")
    movies = pd.read_csv(movie_path)
    books = pd.read_csv(book_path)
    book_tags = pd.read_csv(book_tags_path)
    tags = pd.read_csv(tags_path)

    print(f"Movies loaded: {movies.shape[0]} rows")
    print(f"Books loaded: {books.shape[0]} rows")
    print(f"Book Tags: {book_tags.shape[0]} rows")
    print(f"Tag Names: {tags.shape[0]} rows")
    return movies, books, book_tags, tags


def merge_book_tags(books: pd.DataFrame, book_tags: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    print("Merging book tags...")

    # Attach tag names to book_tags
    book_tags_named = pd.merge(book_tags, tags, on='tag_id', how='left')

    # Merge with books using book_id matching goodreads_book_id
    books_with_tags = pd.merge(
        books,
        book_tags_named,
        left_on='book_id',
        right_on='goodreads_book_id',
        how='left',
    )
    # Base text: book title + author
    books_with_tags["title"] = books_with_tags["title"].fillna("").astype(str)
    books_with_tags["authors"] = books_with_tags["authors"].fillna("").astype(str)

    # tag_name may be NAN if no tag; fill with empty string
    books_with_tags["tag_name"] = books_with_tags["tag_name"].fillna("").astype(str)

    # Group tags by book so each book has a single row
    grouped = (
        books_with_tags
        .groupby(["book_id", "title", "authors"], dropna=False)["tag_name"]
        .apply(lambda x: " ".join(sorted(set([t for t in x if t])))) # unique tags joined
        .reset_index()
    )

    grouped = grouped.rename(columns={"title": "book_title",
                                      "authors": "book_author",
                                      "tag_name": "tag_text"})

    # Building combined text for text embedding
    grouped["combined_text"] =(
        grouped["book_title"].fillna("").astype(str)
        + " "
        + grouped["book_author"].fillna("").astype(str)
        + " "
        + grouped["tag_text"].fillna("").astype(str)
    ).str.strip()

    return grouped


def preprocess_data(movies: pd.DataFrame, books: pd.DataFrame, book_tags: pd.DataFrame, tags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Clean and prepare movie and book datasets for embedding generation
    tqdm.pandas()

    # keep only relevant columns
    print("Preprocessing movies...")
    movies = movies[['title', 'overview']].dropna(subset=["title","overview"])
    movies = movies.rename(columns={"title": "movie_title", "overview": "movie_overview"})

    # clean overview text
    movies['clean_overview'] = movies['movie_overview'].progress_apply(clean_text)

    # Ensure text fields are proper strings and remove rows with empty text
    movies["clean_overview"] = movies["clean_overview"].fillna("").astype(str)
    movies = movies[movies["clean_overview"] != ""]

    print(f"Movies after cleaning: {movies.shape[0]} rows")

    print("Preprocessing books + tags...")
    books_merged = merge_book_tags(books,book_tags, tags)

    # Clean combined_text for books
    books_merged["combined_text"] = books_merged["combined_text"].progress_apply(clean_text)
    books_merged["combined_text"] = books_merged["combined_text"].fillna("").astype(str)

    # Drop rows with empty combined_text (incomplete data)
    before = books_merged.shape[0]
    books_merged = books_merged[books_merged["combined_text"] != ""]
    after = books_merged.shape[0]
    print(f"Books after cleaning and dropping empty text: {after} rows (dropped {before - after})")

    return movies, books_merged


def save_processed_data(movies: pd.DataFrame, books: pd.DataFrame, output_dir: str = "data/processed/") -> None:

    # Save cleaned datasets to the processed data folder

    os.makedirs(output_dir, exist_ok=True)
    movie_out = os.path.join(output_dir, "clean_movies.csv")
    book_out = os.path.join(output_dir, "clean_books.csv")

    movies.to_csv(movie_out, index=False)
    books.to_csv(book_out, index=False)

    print(f"Cleaned datasets saved to {output_dir}")
    print(f" - {movie_out}")
    print(f" - {book_out}")