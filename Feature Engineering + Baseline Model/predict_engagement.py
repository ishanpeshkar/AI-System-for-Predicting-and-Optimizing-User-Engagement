import pandas as pd
import joblib
import os
import argparse

def predict_user_movie_engagement(user_id, movie_id):
    """
    Predicts if a user will like a specific movie based on the trained Random Forest model.
    """
    try:
        # Load the trained model
        rf_model = joblib.load('rf_model.pkl')
        
        # Load the necessary datasets to extract features
        data_dir = 'ml-latest-small'
        ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
        movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))

        # Check if user and movie exist
        if user_id not in ratings['userId'].unique():
            print(f"Warning: New user (userId: {user_id}). Using average user stats.")
            user_avg_rating = ratings['rating'].mean()
            user_rating_count = 1  # Cold start
        else:
            user_data = ratings[ratings['userId'] == user_id]
            user_avg_rating = user_data['rating'].mean()
            user_rating_count = user_data['rating'].count()

        if movie_id not in movies['movieId'].unique():
            print(f"Error: Movie ID {movie_id} not found in the dataset.")
            return
            
        # Get movie stats
        movie_ratings = ratings[ratings['movieId'] == movie_id]
        if not movie_ratings.empty:
            movie_avg_rating = movie_ratings['rating'].mean()
            movie_rating_count = movie_ratings['rating'].count()
        else:
            print(f"Warning: Movie has no prior ratings. Using average movie stats.")
            movie_avg_rating = ratings['rating'].mean()
            movie_rating_count = 1 # Cold start

        # Get movie details (Year and Genres)
        movie_info = movies[movies['movieId'] == movie_id].iloc[0]
        title = movie_info['title']
        genres_string = movie_info['genres']
        
        # Extract year
        import re
        year_match = re.search(r'\((\d{4})\)', title)
        year = int(year_match.group(1)) if year_match else movies['title'].str.extract(r'\((\d{4})\)')[0].fillna(0).astype(int).median()
        
        # Build the feature dictionary in the exact order the model expects
        feature_dict = {
            'movie_avg_rating': movie_avg_rating,
            'movie_rating_count': movie_rating_count,
            'user_avg_rating': user_avg_rating,
            'user_rating_count': user_rating_count,
            'year': year
        }

        # Get all genre columns the model was trained on
        all_genres = set('|'.join(movies['genres']).split('|'))
        movie_genres = set(genres_string.split('|'))
        
        # Add genres to feature dictionary (1 if movie has genre, 0 otherwise)
        for genre in sorted(list(all_genres)):
            feature_dict[genre] = 1 if genre in movie_genres else 0

        # Convert to DataFrame
        features_df = pd.DataFrame([feature_dict])
        
        # Ensure the column order matches exactly what the model was trained on
        if hasattr(rf_model, 'feature_names_in_'):
            expected_columns = rf_model.feature_names_in_
            # Reorder exactly as expected, filling any missing columns with 0
            for col in expected_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[expected_columns]

        # Make Prediction
        prediction = rf_model.predict(features_df)[0]
        probability = rf_model.predict_proba(features_df)[0][1]

        print("-" * 40)
        print(f"Prediction for User {user_id} and Movie '{title}' (ID: {movie_id})")
        print("-" * 40)
        
        if prediction == 1:
            # Use plain ASCII to avoid Windows console encoding issues.
            print("The model predicts the user WILL like this movie.")
        else:
            print("The model predicts the user will NOT like this movie.")
            
        print(f"Confidence (Probability of liking): {probability * 100:.2f}%\n")

    except FileNotFoundError as e:
        print(f"Error: Could not find required file. {e}")
        print("Make sure you have run train_model.py first to generate rf_model.pkl.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict movie engagement for a user.')
    parser.add_argument('--user', type=int, default=1, help='User ID')
    parser.add_argument('--movie', type=int, default=1, help='Movie ID')
    args = parser.parse_args()

    predict_user_movie_engagement(args.user, args.movie)
