
from src.data_preprocessing import load_datasets, preprocess_data, save_processed_data
from src.embedding_generator import embed_movies_and_books
from src.similarity_engine import recommend
from src.build_labeled_pairs import *
from src.model_training import *
if __name__ == "__main__":

    # raw data paths
    movie_path = "data/raw/tmdb_5000_movies.csv"
    book_path = "data/raw/books.csv"
    book_tags_path = "data/raw/book_tags.csv"
    tags_path = "data/raw/tags.csv"

    # loading data set
    movies_raw, books_raw, book_tags, tags = load_datasets(movie_path, book_path, book_tags_path, tags_path)

    # cleaning data
    movies_clean, books_clean = preprocess_data(movies_raw, books_raw, book_tags, tags)

    # saving processed data
    save_processed_data(movies_clean, books_clean)

    # creating embeddings (!!only run once - saves embeddings - time consuming if running on CPU)
    movie_embeddings, book_embeddings = embed_movies_and_books()

    print("=" * 60)
    print("Book recommendation for movie titles")
    print("=" * 60)
    # create recommendations
    test_movie = "Interstellar"

    try:
        recs = recommend(test_movie, top_n=3, min_similarity=0.4)
        print(f"Creating recommendations for movie: '{test_movie}'")
        for r in recs:
            print(f" - {r["book_title"]} by {r["book_author"]} (similarity: {r["similarity"]})")

    except ValueError as e:
        print(f"Error: {e}")

    test_movie = "Titanic"
    try:
        recs = recommend(test_movie, top_n=3, min_similarity=0.3)
        print(f"Creating recommendations for movie: '{test_movie}'")
        for r in recs:
            print(f" - {r["book_title"]} by {r["book_author"]} (similarity: {r["similarity"]})")

    except ValueError as e:
        print(f"Error: {e}")

    test_movie = "Life of pi"
    try:
        recs = recommend(test_movie, top_n=3, min_similarity=0.4)
        print(f"Creating recommendations for movie: '{test_movie}'")
        for r in recs:
            print(f" - {r["book_title"]} by {r["book_author"]} (similarity: {r["similarity"]})")

    except ValueError as e:
        print(f"Error: {e}")
    print("=" * 60)

    #Testing build_label_pairs.py
    movies, books, movie_emb, book_emb = load_data()
    labeled_df = build_labeled_pairs(movies,books, movie_emb,book_emb)
    save_labeled_pairs(labeled_df)

    # Load and prepare data
    df = load_labeled_data()
    X, y = prepare_features_and_labels(df)
    # Cross-validating labeled pairs
    cross_validate_model(X, y, n_splits=5)
    # training and hyperparameter tuning model
    best_model = hyperparameter_tuning(X, y, n_splits=5)
    # saving model
    save_model(best_model)