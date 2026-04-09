import os
from flask import Flask, request, jsonify, render_template
from src.similarity_engine import load_data_and_embeddings, recommend_books_for_movie

app = Flask(__name__)

# Load data and embeddings once at startup
movies_df, books_df, movie_embeddings, book_embeddings = load_data_and_embeddings()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/movies")
def movies():
    titles = movies_df["movie_title"].dropna().tolist()
    return jsonify({"movies": titles})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True)
    if not data or not data.get("movie_title", "").strip():
        return jsonify({"error": "movie_title is required"}), 400

    movie_title = data["movie_title"].strip()
    top_n = int(data.get("top_n", 5))
    top_n = max(3, min(5, top_n))  # clamp to 3–5

    try:
        recs = recommend_books_for_movie(
            movie_title=movie_title,
            movies_df=movies_df,
            books_df=books_df,
            movie_embeddings=movie_embeddings,
            book_embeddings=book_embeddings,
            top_n=top_n,
            min_similarity=0.0,
        )
        return jsonify({"recommendations": recs})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
