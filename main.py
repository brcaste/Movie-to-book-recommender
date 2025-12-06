
from src.data_preprocessing import *
from src.embedding_generator import embed_movies_and_books
from src.similarity_engine import recommend

if __name__ == "__main__":

    # # raw data paths
    # movie_path = "data/raw/tmdb_5000_movies.csv"
    # book_path = "data/raw/books.csv"
    # book_tags_path = "data/raw/book_tags.csv"
    # tags_path = "data/raw/tags.csv"
    #
    # # loading data set
    # movies_raw, books_raw, book_tags, tags = load_datasets(movie_path, book_path, book_tags_path, tags_path)
    #
    # # cleaning data
    # movies_clean, books_clean = preprocess_data(movies_raw, books_raw, book_tags, tags)
    #
    # # saving processed data
    # save_processed_data(movies_clean, books_clean)
    #
    # # creating embeddings
    # movie_embeddings, book_embeddings = embed_movies_and_books()

    # create recommendations
    test_movie = "Avatar"

    try:
        recs = recommend(test_movie, top_n=5, min_similarity=0.2)
        print(f"Creating recommendations for movie: '{test_movie}'")
        for r in recs:
            print(f" - {r["book_title"]} by {r["book_author"]} (similarity: {r["similarity"]})")

    except ValueError as e:
        print(f"Error: {e}")