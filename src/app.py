from flask import Flask, jsonify, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
# Load data and models
try:
    movies_df = pd.read_csv('../movies_df.csv')
    ratings_df = pd.read_csv('../ml-latest-small/ratings.csv')  # Ensure ratings data is available
    cosine_sim = joblib.load('../content_similarity.pkl')
    svd_model = joblib.load('../collaborative_similarity.pkl')  # Trained SVD model
except FileNotFoundError as e:
    raise FileNotFoundError(f"Required file not found: {e}")
except Exception as e:
    raise Exception(f"Error loading resources: {e}")

# Helper Functions
def get_top_recommendations(user_id, svd_model, movies_df, ratings_df, top_n=5):
    # Getting all movies the user has already rated, then getting all movies and movies that haven't been rated by the user
    user_rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    all_movies = movies_df['movieId'].unique()
    unrated_movies = [movie for movie in all_movies if movie not in user_rated_movies]

    # Predicting ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        pred = svd_model.predict(uid=user_id, iid=movie_id)
        predictions.append((movie_id, pred.est))  
    # Sorting predictions by rating in descending order
    # top-n movie IDs
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    
    # getting movie details
    recommended_movies = movies_df[movies_df['movieId'].isin([movie[0] for movie in top_movies])]

    recommended_movies['predicted_rating'] = [movie[1] for movie in top_movies]

    # Return the recommendations as a DataFrame
    return recommended_movies[['title', 'genres', 'predicted_rating']].sort_values(by='predicted_rating', ascending=False)



def recommend_movies(movie_title, df=movies_df, cosine_sim=cosine_sim, top_n=5):
    """
    Recommend movies based on the given movie title.
    
    Parameters:
        movie_title (str): Title (or part of title) of the movie to base recommendations on.
        df (pd.DataFrame): The movies DataFrame containing 'title' and 'genres'.
        cosine_sim (ndarray): Precomputed cosine similarity matrix.
        top_n (int): Number of recommendations to return.
    
    Returns:
        pd.DataFrame: DataFrame with titles and genres of recommended movies.
    """
    matches = df[df['title'].str.contains(movie_title, case=False)]
    
    if matches.empty:
        return f"Movie '{movie_title}' not found in the dataset."
    
    idx = matches.index[0]  

    
    sim_scores = list(enumerate(cosine_sim[idx]))

    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top-n movies
    sim_scores = sim_scores[1:top_n + 1]  
    movie_indices = [i[0] for i in sim_scores]

   
    top_movies = df.iloc[movie_indices][['title', 'genres']].reset_index(drop=True)
    top_movies.index = range(1, top_n + 1)  
    
    return top_movies


def hybrid_recommendations(user_id, movie_title, svd_model= svd_model, movies_df =  movies_df, ratings_df = ratings_df, cosine_sim = cosine_sim, top_n=5):
    """
    Generate hybrid movie recommendations using SVD (collaborative filtering) 
    and cosine similarity (content-based filtering).
    
    Parameters:
        user_id (int): The ID of the user.
        movie_title (str): Title (or part of title) of the movie for content-based recommendations.
        svd_model: Trained SVD model for collaborative filtering.
        movies_df (pd.DataFrame): DataFrame with movie details (movieId, title, genres).
        ratings_df (pd.DataFrame): DataFrame with user ratings (userId, movieId, rating).
        cosine_sim (ndarray): Precomputed cosine similarity matrix for content-based filtering.
        top_n (int): Number of recommendations to return from each method.
    
    Returns:
        pd.DataFrame: DataFrame with combined hybrid recommendations.
    """
    # Collaborative Filtering Recommendations
    cf_recommendations = pd.DataFrame()
    if user_id and user_id in ratings_df['userId'].unique():
        cf_recommendations = get_top_recommendations(user_id, svd_model, movies_df, ratings_df, top_n)
        cf_recommendations['source'] = 'Collaborative Filtering'
    else:
        # Handling cold-start
        cf_recommendations = pd.DataFrame(columns=['title', 'genres', 'predicted_rating', 'source'])

    # Content-Based Recommendations
    cb_recommendations = pd.DataFrame(columns=['title', 'genres', 'source', 'predicted_rating'])  # Default empty DataFrame
    if movie_title:
        cb_recommendations = recommend_movies(movie_title, df=movies_df, cosine_sim=cosine_sim, top_n=top_n)
        if isinstance(cb_recommendations, str):  # Handle case when movie_title is not found
            cb_recommendations = pd.DataFrame(columns=['title', 'genres', 'source', 'predicted_rating'])
        else:
            cb_recommendations['source'] = 'Content-Based Filtering'
            cb_recommendations['predicted_rating'] = None  # No predicted rating for content-based

    # Combining Both Recommendations
    combined_recommendations = pd.concat(
        [cf_recommendations, cb_recommendations],
        ignore_index=True
    ).drop_duplicates(subset='title') 

    # Sorting Combined Recommendations
    combined_recommendations['predicted_rating'] = combined_recommendations['predicted_rating'].fillna(0)
    combined_recommendations = combined_recommendations.sort_values(
        by='predicted_rating', ascending=False, na_position='last'
    ).head(top_n)

    combined_recommendations.index = range(1, len(combined_recommendations) + 1)

    return combined_recommendations[['title', 'genres', 'predicted_rating', 'source']]


@app.route('/recommend', methods=['GET', "POST"])
def recommend():
    recommendations = []
    if request.method == "POST":
        movie_title = request.form.get("movie_title")
        user_id = request.form.get("user_id")
        user_id = None if user_id == 0 else int(user_id)
        recommendations = hybrid_recommendations(movie_title=movie_title, user_id=user_id)
        recommendations = recommendations.to_dict(orient='records')

    return render_template(
        "recommendations.html", recommendations=recommendations
    )


if __name__  == '__main__':
  app.run(debug=True)