from flask import Flask, jsonify, render_template, request
import pandas as pd
import joblib
from functools import lru_cache

app = Flask(__name__)

# Helper function to load data and models only when needed
@lru_cache(maxsize=None)
def load_movies_df():
    return pd.read_csv('../movies_df.csv')

@lru_cache(maxsize=None)
def load_ratings_df():
    return pd.read_csv('../ml-latest-small/ratings.csv')

@lru_cache(maxsize=None)
def load_cosine_sim():
    return joblib.load('../content_similarity.pkl')

@lru_cache(maxsize=None)
def load_svd_model():
    return joblib.load('../collaborative_similarity.pkl')

# Recommendation logic
def get_top_recommendations(user_id, svd_model, movies_df, ratings_df, top_n=5):
    user_rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    all_movies = movies_df['movieId'].unique()
    unrated_movies = [movie for movie in all_movies if movie not in user_rated_movies]

    predictions = [
        (movie_id, svd_model.predict(uid=user_id, iid=movie_id).est)
        for movie_id in unrated_movies
    ]
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    recommended_movies = movies_df[movies_df['movieId'].isin([movie[0] for movie in top_movies])].copy()
    recommended_movies['predicted_rating'] = [movie[1] for movie in top_movies]

    return recommended_movies[['title', 'genres', 'predicted_rating']].sort_values(
        by='predicted_rating', ascending=False
    )

def recommend_movies(movie_title, df, cosine_sim, top_n=5):
    matches = df[df['title'].str.contains(movie_title, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]
    top_movies = df.iloc[movie_indices][['title', 'genres']].reset_index(drop=True)
    top_movies['predicted_rating'] = None  # Placeholder for consistency
    return top_movies

def hybrid_recommendations(user_id, movie_title, top_n=5):
    movies_df = load_movies_df()
    ratings_df = load_ratings_df()
    svd_model = load_svd_model()
    cosine_sim = load_cosine_sim()

    cf_recommendations = pd.DataFrame()
    if user_id and user_id in ratings_df['userId'].unique():
        cf_recommendations = get_top_recommendations(
            user_id, svd_model, movies_df, ratings_df, top_n
        )
        cf_recommendations['source'] = 'Collaborative Filtering'

    cb_recommendations = recommend_movies(movie_title, movies_df, cosine_sim, top_n)
    if not cb_recommendations.empty:
        cb_recommendations['source'] = 'Content-Based Filtering'

    combined_recommendations = pd.concat(
        [cf_recommendations, cb_recommendations], ignore_index=True
    ).drop_duplicates(subset='title')

    combined_recommendations['predicted_rating'] = combined_recommendations['predicted_rating'].fillna(0)
    combined_recommendations = combined_recommendations.sort_values(
        by='predicted_rating', ascending=False
    ).head(top_n)
    combined_recommendations.index = range(1, len(combined_recommendations) + 1)

    return combined_recommendations[['title', 'genres', 'predicted_rating', 'source']]

# Routes
@app.route('/recommend', methods=['GET', "POST"])
def recommend():
    recommendations = []
    if request.method == "POST":
        movie_title = request.form.get("movie_title")
        user_id = request.form.get("user_id")
        user_id = None if user_id == "0" or not user_id else int(user_id)

        recommendations_df = hybrid_recommendations(user_id, movie_title)
        recommendations = recommendations_df.to_dict(orient='records')

    return render_template(
        "recommendations.html", recommendations=recommendations
    )

if __name__ == '__main__':
    app.run(debug=True)
