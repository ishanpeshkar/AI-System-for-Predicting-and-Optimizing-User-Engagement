import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    print("Loading data...")
    data_dir = 'ml-latest-small'
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))

    # Define target variable 'liked' (1 if rating >= 3.5 else 0)
    print("Creating target variable 'liked'...")
    ratings['liked'] = (ratings['rating'] >= 3.5).astype(int)

    print("Engineering features...")
    # Movie features
    movie_stats = ratings.groupby('movieId').agg(
        movie_avg_rating=('rating', 'mean'),
        movie_rating_count=('rating', 'count')
    ).reset_index()

    # User features
    user_stats = ratings.groupby('userId').agg(
        user_avg_rating=('rating', 'mean'),
        user_rating_count=('rating', 'count')
    ).reset_index()

    # Merge features
    df = ratings.merge(movie_stats, on='movieId', how='left')
    df = df.merge(user_stats, on='userId', how='left')

    # Genres one-hot encoding
    genres_expanded = movies['genres'].str.get_dummies(sep='|')
    genres_expanded['movieId'] = movies['movieId']
    
    # Merge genres and title/year
    # Extract year from title (e.g., "Toy Story (1995)")
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
    # Impute missing years with median
    movies['year'] = movies['year'].fillna(movies['year'].median())

    df = df.merge(movies[['movieId', 'year']], on='movieId', how='left')
    df = df.merge(genres_expanded, on='movieId', how='left')

    # Drop unnecessary columns
    features_to_drop = ['userId', 'movieId', 'rating', 'timestamp', 'liked']
    
    # Check for NaNs and fill
    df = df.fillna(0)

    print("Saving feature_matrix.csv...")
    # Save the feature matrix as requested
    df.to_csv('feature_matrix.csv', index=False)

    X = df.drop(columns=features_to_drop)
    y = df['liked']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    print("Saving rf_model.pkl...")
    joblib.dump(rf, 'rf_model.pkl')

    print("Evaluating model...")
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba)
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    print("Saving evaluation_metrics.csv...")
    metrics_df.to_csv('evaluation_metrics.csv', index=False)
    print(metrics_df)

    print("Saving feature_importance_chart.png...")
    importances = rf.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    top_features = feat_importances.nlargest(15)

    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh').invert_yaxis()
    plt.title('Top 15 Feature Importances for User Engagement')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance_chart.png')
    
    print("Done!")

if __name__ == '__main__':
    main()
