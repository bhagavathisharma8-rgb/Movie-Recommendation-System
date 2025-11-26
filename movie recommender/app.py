from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import pandas as pd

# -----------------------------
# Load data and ML setup
# -----------------------------
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)

def get_recommendations(title):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")  # serve HTML from Flask

@app.route("/recommend", methods=["GET"])
def recommend():
    movie_name = request.args.get("title")
    recs = get_recommendations(movie_name)
    return jsonify({"recommendations": recs})

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
