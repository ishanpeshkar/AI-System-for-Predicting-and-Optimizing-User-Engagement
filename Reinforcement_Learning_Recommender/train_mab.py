"""
RL Recommender Training Pipeline
================================

Trains the Multi-Armed Bandit agent on historical user-movie interaction data.
Integrates with Member 3's LSTM state extractor for contextual recommendations.
"""

import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from mab_recommender import (
    RLRecommenderAgent, 
    compute_reward, 
    compute_engagement_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommenderTrainer:
    """Trainer for RL Recommender Agent"""
    
    def __init__(self, data_dir: str, algorithm: str = "thompson"):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing CSV files
            algorithm: Bandit algorithm to use
        """
        self.data_dir = Path(data_dir)
        self.algorithm = algorithm
        
        # Load data
        self.movies_df = pd.read_csv(self.data_dir / "movies.csv")
        self.ratings_df = pd.read_csv(self.data_dir / "ratings.csv")
        self.tags_df = pd.read_csv(self.data_dir / "tags.csv")
        self.links_df = pd.read_csv(self.data_dir / "links.csv")
        
        logger.info(f"Loaded {len(self.movies_df)} movies")
        logger.info(f"Loaded {len(self.ratings_df)} ratings")
        logger.info(f"Loaded {len(self.tags_df)} tags")
        
        # Create movie index mapping
        self.movie_to_idx = {mid: idx for idx, mid in enumerate(self.movies_df['movieId'].values)}
        self.idx_to_movie = {idx: mid for mid, idx in self.movie_to_idx.items()}
        
        # Initialize agent
        n_movies = len(self.movies_df)
        self.agent = RLRecommenderAgent(
            n_movies=n_movies,
            algorithm=algorithm
        )
        
        self.training_metrics = {
            'cumulative_reward': [],
            'avg_reward': [],
            'movie_counts': defaultdict(int),
            'reward_by_movie': defaultdict(list),
            'user_engagement': defaultdict(list)
        }
    
    def train_on_ratings(self, n_samples: Optional[int] = None, 
                        shuffle: bool = True):
        """
        Train agent on historical ratings
        
        Args:
            n_samples: Number of samples to train on (None = all)
            shuffle: Whether to shuffle data before training
        """
        logger.info(f"Training on ratings using {self.algorithm} algorithm...")
        
        # Sort ratings by timestamp to maintain temporal order
        ratings_data = self.ratings_df.sort_values('timestamp').copy()
        
        if shuffle:
            ratings_data = ratings_data.sample(frac=1, random_state=42)
        
        if n_samples:
            ratings_data = ratings_data.head(n_samples)
        
        # Training loop
        for step, (_, row) in enumerate(ratings_data.iterrows(), start=1):
            user_id = int(row['userId'])
            movie_id = int(row['movieId'])
            rating = float(row['rating'])
            
            # Get movie index
            movie_idx = self.movie_to_idx.get(movie_id)
            if movie_idx is None:
                continue
            
            # Observe reward
            reward = self.agent.observe_reward(user_id, movie_idx, rating)
            
            # Track metrics
            self.training_metrics['movie_counts'][movie_idx] += 1
            self.training_metrics['reward_by_movie'][movie_idx].append(reward)
            self.training_metrics['user_engagement'][user_id].append(reward)
            
            # Log cumulative stats
            if step % 10000 == 0:
                stats = self.agent.get_stats()
                logger.info(
                    f"Step {step}: "
                    f"Avg Reward = {stats['avg_reward']:.4f}, "
                    f"Total Reward = {stats['total_reward']:.2f}"
                )
                self.training_metrics['cumulative_reward'].append(stats['total_reward'])
                self.training_metrics['avg_reward'].append(stats['avg_reward'])
        
        logger.info("Training completed!")
        final_stats = self.agent.get_stats()
        logger.info(f"Final Stats: {final_stats}")
        
        return self.agent
    
    def evaluate_on_test_set(self, test_frac: float = 0.2, max_eval_interactions: int = 3000) -> Dict:
        """
        Evaluate agent on held-out test set
        
        Args:
            test_frac: Fraction of data to use for testing
        
        Returns:
            Evaluation metrics dictionary
        """
        logger.info(f"Evaluating on {test_frac*100}% test set...")
        
        # Split data
        ratings_data = self.ratings_df.sort_values('timestamp').copy()
        test_size = int(len(ratings_data) * test_frac)
        test_data = ratings_data.tail(test_size)
        if max_eval_interactions is not None and len(test_data) > max_eval_interactions:
            test_data = test_data.tail(max_eval_interactions)
        
        hit_count = 0
        total_count = 0
        rewards_list = []
        user_recommendations = defaultdict(list)
        
        for _, row in test_data.iterrows():
            user_id = int(row['userId'])
            movie_id = int(row['movieId'])
            rating = float(row['rating'])
            
            movie_idx = self.movie_to_idx.get(movie_id)
            if movie_idx is None:
                continue
            
            # Get recommendation
            recommendation = self.agent.recommend(user_id, top_k=10)
            
            # Check if this movie was in top-10
            if movie_idx in recommendation:
                hit_count += 1
            
            total_count += 1
            reward = compute_reward(rating)
            rewards_list.append(reward)
            user_recommendations[user_id].append(reward)
        
        # Compute metrics
        hit_rate = hit_count / total_count if total_count > 0 else 0
        avg_reward = np.mean(rewards_list) if rewards_list else 0
        
        metrics = {
            'hit_rate': hit_rate,
            'hit_count': hit_count,
            'total_count': total_count,
            'avg_reward': avg_reward,
            'reward_std': np.std(rewards_list) if rewards_list else 0,
            'n_test_users': len(user_recommendations),
            'max_eval_interactions': max_eval_interactions
        }
        
        logger.info(f"Test Metrics: {metrics}")
        return metrics
    
    def get_recommendations_for_user(self, user_id: int, 
                                     top_k: int = 10) -> List[Dict]:
        """
        Get top recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
        
        Returns:
            List of recommendation dicts with movie info
        """
        movie_indices = self.agent.recommend(user_id, top_k=top_k)
        recommendations = []
        
        for idx, movie_idx in enumerate(movie_indices):
            movie_id = self.idx_to_movie.get(movie_idx)
            if movie_id and movie_id in self.movies_df['movieId'].values:
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                recommendations.append({
                    'rank': idx + 1,
                    'movie_id': movie_id,
                    'movie_idx': movie_idx,
                    'title': movie_info['title'],
                    'genres': movie_info['genres']
                })
        
        return recommendations
    
    def save_agent(self, filepath: str):
        """Save trained agent"""
        self.agent.save(filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training progression"""
        if not self.training_metrics['cumulative_reward']:
            logger.warning("No training metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Cumulative reward
        axes[0, 0].plot(self.training_metrics['cumulative_reward'], linewidth=2)
        axes[0, 0].set_title('Cumulative Reward Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average reward
        axes[0, 1].plot(self.training_metrics['avg_reward'], linewidth=2, color='orange')
        axes[0, 1].set_title('Average Reward Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top movies by frequency
        top_movies = sorted(
            self.training_metrics['movie_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        movie_titles = [
            self.movies_df[self.movies_df['movieId'] == self.idx_to_movie[idx]]['title'].values[0]
            for idx, _ in top_movies
        ]
        counts = [count for _, count in top_movies]
        
        axes[1, 0].barh(range(len(movie_titles)), counts, color='steelblue')
        axes[1, 0].set_yticks(range(len(movie_titles)))
        axes[1, 0].set_yticklabels(movie_titles, fontsize=8)
        axes[1, 0].set_title('Top 20 Most Recommended Movies', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Recommendation Count')
        axes[1, 0].invert_yaxis()
        
        # Reward distribution
        all_rewards = []
        for rewards in self.training_metrics['reward_by_movie'].values():
            all_rewards.extend(rewards)
        
        axes[1, 1].hist(all_rewards, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Distribution of Rewards', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Reward Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")
        
        return fig
    
    def generate_report(self, output_path: str = "training_report.txt"):
        """Generate training report"""
        stats = self.agent.get_stats()
        movie_mean_rewards = [
            float(np.mean(rewards))
            for rewards in self.training_metrics['reward_by_movie'].values()
            if len(rewards) > 0
        ]
        avg_reward_per_movie = float(np.mean(movie_mean_rewards)) if movie_mean_rewards else 0.0
        
        report = f"""
================================================================================
RL RECOMMENDER TRAINING REPORT
================================================================================
Algorithm: {self.algorithm}
Training Date: {pd.Timestamp.now()}

TRAINING STATISTICS
-------------------
Total Interactions: {stats['total_interactions']}
Total Reward: {stats['total_reward']:.2f}
Average Reward per Interaction: {stats['avg_reward']:.4f}
Number of Users: {stats['n_users']}
Number of Movies: {stats['n_movies']}

MOVIE STATISTICS
----------------
Movies with interactions: {len(self.training_metrics['movie_counts'])}
Average reward per movie: {avg_reward_per_movie:.4f}

TOP 10 MOVIES BY ENGAGEMENT
---------------------------
"""
        movie_avg_rewards = {}
        for movie_idx, rewards in self.training_metrics['reward_by_movie'].items():
            if rewards:
                movie_avg_rewards[movie_idx] = np.mean(rewards)
        
        top_movies = sorted(movie_avg_rewards.items(), key=lambda x: x[1], reverse=True)[:10]
        for rank, (movie_idx, avg_reward) in enumerate(top_movies, 1):
            movie_id = self.idx_to_movie.get(movie_idx)
            movie_title = "Unknown"
            if movie_id and movie_id in self.movies_df['movieId'].values:
                movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title'].values[0]
            
            report += f"{rank}. {movie_title} (avg_reward: {avg_reward:.2f})\n"
        
        report += "\n" + "="*80 + "\n"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")
        return report


def main():
    """Main training pipeline"""
    # Setup
    data_dir = "data"
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Create trainer
    trainer = RecommenderTrainer(data_dir, algorithm="thompson")
    
    # Train on full dataset
    trainer.train_on_ratings(n_samples=None, shuffle=False)
    
    # Evaluate
    test_metrics = trainer.evaluate_on_test_set(test_frac=0.2)
    
    # Visualize
    trainer.plot_training_curves(save_path=output_dir / "training_curves.png")
    
    # Generate report
    trainer.generate_report(output_path=output_dir / "training_report.txt")
    
    # Save agent
    trainer.save_agent(str(output_dir / "mab_agent_thompson.pkl"))
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
