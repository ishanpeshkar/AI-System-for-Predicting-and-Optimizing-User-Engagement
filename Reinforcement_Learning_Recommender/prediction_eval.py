"""
Prediction and Recommendation Module
===================================

Provides inference, recommendation generation, and performance evaluation
for the RL Recommender system.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import pickle

logger = logging.getLogger(__name__)


class RecommendationPredictor:
    """Make predictions and recommendations with trained agent"""
    
    def __init__(self, agent, movies_df: pd.DataFrame, 
                 movie_to_idx: Dict[int, int], idx_to_movie: Dict[int, int]):
        """
        Initialize predictor
        
        Args:
            agent: Trained RLRecommenderAgent
            movies_df: Movies dataframe
            movie_to_idx: MovieId to index mapping
            idx_to_movie: Index to movieId mapping
        """
        self.agent = agent
        self.movies_df = movies_df
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = idx_to_movie
    
    def recommend_for_user(self, user_id: int, top_k: int = 10,
                          user_state: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Get top-k recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            user_state: Optional LSTM state vector for contextual recommendations
        
        Returns:
            List of recommendation dicts with movie information
        """
        try:
            # Get movie indices from agent
            movie_indices = self.agent.recommend(
                user_id, 
                user_state=user_state,
                top_k=top_k
            )
            
            recommendations = []
            for rank, movie_idx in enumerate(movie_indices, 1):
                movie_id = self.idx_to_movie.get(movie_idx)
                
                if movie_id is None:
                    continue
                
                # Get movie info
                movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
                
                if len(movie_row) == 0:
                    continue
                
                movie_info = movie_row.iloc[0]
                
                # Get bandit stats for this movie
                stats = self.agent.bandit.get_arm_stats(movie_idx)
                
                recommendations.append({
                    'rank': rank,
                    'movieId': int(movie_id),
                    'movie_idx': int(movie_idx),
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'mean_rating': float(stats.get('mean_reward', 0)),
                    'times_recommended': int(stats.get('count', 0))
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return []
    
    def get_movie_info(self, movie_id: int) -> Optional[Dict]:
        """Get detailed info for a movie"""
        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        
        if len(movie_row) == 0:
            return None
        
        movie_info = movie_row.iloc[0]
        movie_idx = self.movie_to_idx.get(movie_id)
        
        if movie_idx is None:
            return None
        
        stats = self.agent.bandit.get_arm_stats(movie_idx)
        
        return {
            'movieId': int(movie_id),
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'mean_reward': float(stats.get('mean_reward', 0)),
            'recommendation_count': int(stats.get('count', 0))
        }
    
    def get_top_movies(self, top_k: int = 20) -> List[Dict]:
        """Get top-k movies by average reward"""
        movie_rewards = []
        
        for movie_idx in range(len(self.idx_to_movie)):
            stats = self.agent.bandit.get_arm_stats(movie_idx)
            if stats.get('count', 0) > 0:
                movie_id = self.idx_to_movie.get(movie_idx)
                if movie_id:
                    movie_rewards.append((movie_idx, stats['mean_reward'], movie_id))
        
        # Sort by mean reward
        top_movies = sorted(movie_rewards, key=lambda x: x[1], reverse=True)[:top_k]
        
        result = []
        for rank, (movie_idx, mean_reward, movie_id) in enumerate(top_movies, 1):
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            result.append({
                'rank': rank,
                'movieId': int(movie_id),
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'mean_reward': float(mean_reward)
            })
        
        return result


class EvaluationMetrics:
    """Compute evaluation metrics for recommendation systems"""
    
    @staticmethod
    def hit_rate(recommendations: List[int], ground_truth: int) -> int:
        """Check if ground truth in recommendations"""
        return 1 if ground_truth in recommendations else 0
    
    @staticmethod
    def reciprocal_rank(recommendations: List[int], ground_truth: int) -> float:
        """Compute reciprocal rank (1/rank if found, 0 otherwise)"""
        try:
            rank = recommendations.index(ground_truth) + 1
            return 1.0 / rank
        except ValueError:
            return 0.0
    
    @staticmethod
    def dcg(relevance_scores: List[float]) -> float:
        """Discounted Cumulative Gain"""
        dcg = 0.0
        for i, rel in enumerate(relevance_scores, 1):
            dcg += rel / np.log2(i + 1)
        return dcg
    
    @staticmethod
    def ndcg(recommendations: List[int], ground_truth_ratings: Dict[int, float],
             k: int = 10) -> float:
        """Normalized DCG"""
        # Ideal ranking (all relevant items first)
        ideal_rel = sorted(
            [rating for rating in ground_truth_ratings.values()],
            reverse=True
        )[:k]
        ideal_dcg = EvaluationMetrics.dcg(ideal_rel)
        
        if ideal_dcg == 0:
            return 0.0
        
        # Actual ranking
        actual_rel = []
        for rec in recommendations[:k]:
            if rec in ground_truth_ratings:
                actual_rel.append(ground_truth_ratings[rec])
            else:
                actual_rel.append(0)
        
        actual_dcg = EvaluationMetrics.dcg(actual_rel)
        return actual_dcg / ideal_dcg
    
    @staticmethod
    def diversity(recommendations: List[int], genres_dict: Dict[int, str]) -> float:
        """
        Compute diversity of recommendations
        Based on genre diversity
        """
        if len(recommendations) <= 1:
            return 0.0
        
        genres_list = [set(genres_dict.get(movie_id, "").split("|")) 
                      for movie_id in recommendations]
        
        # Compute pairwise dissimilarity
        dissimilarities = []
        for i in range(len(genres_list)):
            for j in range(i + 1, len(genres_list)):
                intersection = len(genres_list[i] & genres_list[j])
                union = len(genres_list[i] | genres_list[j])
                jaccard = 1.0 - (intersection / union) if union > 0 else 0
                dissimilarities.append(jaccard)
        
        return np.mean(dissimilarities) if dissimilarities else 0.0
    
    @staticmethod
    def coverage(recommendations: List[List[int]], total_items: int) -> float:
        """Fraction of catalog that is recommended at least once"""
        unique_items = set()
        for rec_list in recommendations:
            unique_items.update(rec_list)
        
        return len(unique_items) / total_items if total_items > 0 else 0.0
    
    @staticmethod
    def novelty(recommendations: List[int], popularity: Dict[int, int],
               total_interactions: int) -> float:
        """
        Average popularity-based novelty
        Rare items have higher novelty
        """
        if not recommendations:
            return 0.0
        
        novelties = []
        for item in recommendations:
            pop = popularity.get(item, 0)
            nov = -np.log2((pop + 1) / (total_interactions + 1))
            novelties.append(nov)
        
        return np.mean(novelties)


class OnlineEvaluator:
    """
    Evaluate recommendations in an online fashion
    (simulating real-time user feedback)
    """
    
    def __init__(self, agent, movies_df: pd.DataFrame,
                 ratings_df: pd.DataFrame, movie_to_idx: Dict,
                 idx_to_movie: Dict):
        """Initialize evaluator"""
        self.agent = agent
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.movie_to_idx = movie_to_idx
        self.idx_to_movie = idx_to_movie
        
        # Pre-compute popularity
        self.movie_popularity = self.ratings_df['movieId'].value_counts().to_dict()
        
        # Genre mapping
        self.genres_map = {
            row['movieId']: row['genres']
            for _, row in movies_df.iterrows()
        }
        
        self.metrics_history = []
    
    def evaluate_batch(self, test_ratings: pd.DataFrame, k: int = 10) -> Dict:
        """
        Evaluate on batch of test ratings
        
        Args:
            test_ratings: DataFrame with userId, movieId, rating columns
            k: Number of recommendations
        
        Returns:
            Dictionary with evaluation metrics
        """
        hit_rates = []
        mrr_scores = []
        ndcg_scores = []
        diversity_scores = []
        
        n_unique_users = test_ratings['userId'].nunique()
        
        for user_id in test_ratings['userId'].unique():
            user_ratings = test_ratings[test_ratings['userId'] == user_id]
            
            # Get user's movie history
            user_history = self.ratings_df[self.ratings_df['userId'] == user_id]
            watched_movies = set(user_history['movieId'].unique())
            
            # Get recommendations (excluding seen movies)
            recommendations = self.agent.recommend(user_id, top_k=k * 2)
            
            # Filter out already watched
            filtered_recs = [
                m for m in recommendations 
                if self.idx_to_movie.get(m) not in watched_movies
            ][:k]
            
            if not filtered_recs:
                continue
            
            # Convert to movie IDs
            rec_movie_ids = [self.idx_to_movie.get(m) for m in filtered_recs]
            
            for _, row in user_ratings.iterrows():
                movie_id = int(row['movieId'])
                rating = float(row['rating'])
                
                # Hit rate
                hr = EvaluationMetrics.hit_rate(rec_movie_ids, movie_id)
                hit_rates.append(hr)
                
                # MRR
                mrr = EvaluationMetrics.reciprocal_rank(rec_movie_ids, movie_id)
                mrr_scores.append(mrr)
                
                # Diversity
                if len(rec_movie_ids) > 1:
                    div = EvaluationMetrics.diversity(
                        rec_movie_ids,
                        self.genres_map
                    )
                    diversity_scores.append(div)
        
        # Compute final metrics
        metrics = {
            'hit_rate@k': np.mean(hit_rates) if hit_rates else 0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0,
            'diversity': np.mean(diversity_scores) if diversity_scores else 0,
            'coverage': EvaluationMetrics.coverage(
                [[self.idx_to_movie.get(m) for m in self.agent.recommend(uid, top_k=k)]
                 for uid in test_ratings['userId'].unique()],
                len(self.movies_df)
            ),
            'n_evaluated_users': n_unique_users,
            'n_evaluated_ratings': len(test_ratings)
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of metrics over time"""
        return pd.DataFrame(self.metrics_history)


def load_predictor(agent_path: str, data_dir: str) -> RecommendationPredictor:
    """Load saved agent and create predictor"""
    # Load agent
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)
    
    # Load data
    movies_df = pd.read_csv(Path(data_dir) / "movies.csv")
    movie_to_idx = {mid: idx for idx, mid in enumerate(movies_df['movieId'].values)}
    idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}
    
    return RecommendationPredictor(agent, movies_df, movie_to_idx, idx_to_movie)


if __name__ == "__main__":
    print("Prediction and Evaluation Modules Ready")
