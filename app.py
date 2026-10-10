import os
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)
movies = pd.read_csv(os.path.join(BASE_DIR, "movies_deployment.csv"))

# Ensure required text columns exist
for col in ["Series_Title", "Genre", "Overview", "Director",
            "Star1", "Star2", "Star3", "Star4"]:
    if col not in movies.columns:
        movies[col] = ""

movies["Series_Title"] = movies["Series_Title"].fillna("").astype(str)

# Build content features from movie metadata
feature_cols = ["Genre", "Overview", "Director", "Star1", "Star2", "Star3", "Star4"]
movies["tags"] = movies[feature_cols].fillna("").astype(str).agg(" ".join, axis=1)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(movies["tags"])
similarity = cosine_similarity(tfidf_matrix)

def movie_record(row):
    def safe(value):
        return None if pd.isna(value) else value

    return {
        "title": safe(row.get("Series_Title")),
        "Display_Title": safe(row.get("Display_Title")) or safe(row.get("Series_Title")),
        "genre": safe(row.get("Genre")),
        "rating": safe(row.get("IMDB_Rating")),
        "year": safe(row.get("Released_Year")),
        "overview": safe(row.get("Overview")),
        "director": safe(row.get("Director")),
        "cast": [safe(row.get(f"Star{i}")) for i in range(1, 5)],
        "poster": safe(row.get("Poster_Link"))
    }

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "moviemind.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "movies_loaded": len(movies),
        "recommendation_method": "TF-IDF and Cosine Similarity"
    })

@app.route("/api/all-movies")
def all_movies():
    return jsonify([movie_record(row) for _, row in movies.iterrows()])

@app.route("/api/movies")
def search_movies():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Please provide a search query using ?q="}), 400

    results = movies[
        movies["Series_Title"].str.contains(query, case=False, regex=False, na=False)
    ].head(10)

    return jsonify([movie_record(row) for _, row in results.iterrows()])

@app.route("/api/recommend")
def get_recommendations():
    title = request.args.get("title", "").strip()

    try:
        limit = int(request.args.get("limit", "5"))
        if not 1 <= limit <= 20:
            raise ValueError
    except ValueError:
        return jsonify({"error": "Limit must be an integer from 1 to 20"}), 400

    if not title:
        return jsonify({"error": "Provide a movie title using ?title="}), 400

    matches = movies.index[
        movies["Series_Title"].str.casefold() == title.casefold()
    ]

    if len(matches) == 0:
        return jsonify({"error": f"Movie '{title}' not found"}), 404

    movie_index = matches[0]
    scores = [
        (i, score) for i, score in enumerate(similarity[movie_index])
        if i != movie_index
    ]
    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = []
    for index, score in scores[:limit]:
        record = movie_record(movies.iloc[index])
        record["similarity_score"] = round(float(score), 4)
        recommendations.append(record)

    return jsonify({
        "searched_movie": movies.iloc[movie_index]["Series_Title"],
        "recommendations": recommendations
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
