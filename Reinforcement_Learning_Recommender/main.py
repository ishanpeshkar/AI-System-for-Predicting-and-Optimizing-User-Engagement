"""
Main Execution Script
====================

Complete pipeline for training, evaluation, and deployment of RL Recommender
"""

import sys
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary output directories"""
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    return output_dir, results_dir


def train_and_evaluate(algorithm: str = "thompson", n_samples: int = None):
    """
    Full training and evaluation pipeline
    
    Args:
        algorithm: Bandit algorithm to use
        n_samples: Number of samples for training
    """
    from train_mab import RecommenderTrainer
    
    logger.info("=" * 80)
    logger.info("RL RECOMMENDER TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Training samples: {n_samples if n_samples else 'All'}")
    
    output_dir, results_dir = setup_directories()
    
    # Create trainer
    trainer = RecommenderTrainer(data_dir="data", algorithm=algorithm)
    
    # Train
    logger.info("\n[1/4] Training on ratings data...")
    trainer.train_on_ratings(n_samples=n_samples, shuffle=False)
    
    # Evaluate
    logger.info("\n[2/4] Evaluating on test set...")
    test_metrics = trainer.evaluate_on_test_set(test_frac=0.2)
    
    logger.info(f"Test Metrics: {test_metrics}")
    
    # Visualize
    logger.info("\n[3/4] Generating training visualizations...")
    trainer.plot_training_curves(save_path=output_dir / "training_curves.png")
    
    # Generate report
    logger.info("\n[4/4] Generating training report...")
    report = trainer.generate_report(output_path=results_dir / "training_report.txt")
    
    # Save agent
    agent_path = output_dir / f"mab_agent_{algorithm}.pkl"
    trainer.save_agent(str(agent_path))
    
    logger.info(f"\nAgent saved to: {agent_path}")
    
    # Save test metrics
    metrics_path = results_dir / f"test_metrics_{algorithm}.json"
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"Test metrics saved to: {metrics_path}")
    logger.info("\nTraining pipeline completed successfully!")
    
    return trainer, agent_path, test_metrics


def test_recommendations(agent_path: str, n_test_users: int = 10):
    """
    Test recommendation generation
    
    Args:
        agent_path: Path to trained agent
        n_test_users: Number of test users
    """
    from prediction_eval import load_predictor, EvaluationMetrics
    import pandas as pd
    
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATION TESTING")
    logger.info("=" * 80)
    
    output_dir, results_dir = setup_directories()
    
    # Load predictor
    predictor = load_predictor(str(agent_path), "data")
    
    # Load user data
    ratings_df = pd.read_csv("data/ratings.csv")
    unique_users = ratings_df['userId'].unique()
    test_users = np.random.choice(unique_users, size=min(n_test_users, len(unique_users)),
                                  replace=False)
    
    logger.info(f"Testing on {len(test_users)} users")
    
    # Generate recommendations for test users
    test_results = []
    
    for user_id in test_users:
        recommendations = predictor.recommend_for_user(user_id, top_k=10)
        
        logger.info(f"\nUser {user_id}:")
        logger.info(f"  Top 5 Recommendations:")
        for rec in recommendations[:5]:
            logger.info(f"    {rec['rank']}. {rec['title']} ({rec['genres']})")
        
        test_results.append({
            'user_id': user_id,
            'recommendations': recommendations
        })
    
    # Save results
    results_path = results_dir / "sample_recommendations.json"
    serializable_results = []
    for result in test_results:
        serializable_recommendations = []
        for rec in result['recommendations']:
            serializable_recommendations.append({
                'rank': int(rec.get('rank', 0)),
                'movieId': int(rec.get('movieId', 0)),
                'movie_idx': int(rec.get('movie_idx', 0)),
                'title': rec.get('title', ''),
                'mean_rating': float(rec.get('mean_rating', 0.0)),
                'times_recommended': int(rec.get('times_recommended', 0))
            })
        serializable_results.append({
            'user_id': int(result['user_id']),
            'recommendations': serializable_recommendations
        })

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    
    return test_results


def generate_analysis_report(trainer_obj, agent_path: str):
    """
    Generate comprehensive analysis report
    
    Args:
        trainer_obj: RecommenderTrainer instance
        agent_path: Path to agent
    """
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING ANALYSIS REPORT")
    logger.info("=" * 80)
    
    output_dir, results_dir = setup_directories()
    
    # Collect statistics
    stats = trainer_obj.agent.get_stats()
    
    report = f"""
================================================================================
RL RECOMMENDER SYSTEM - COMPREHENSIVE ANALYSIS REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Algorithm: {trainer_obj.algorithm}

1. SYSTEM ARCHITECTURE
---------------------
The Multi-Armed Bandit recommender system models the recommendation problem as:
  - State (S): User context/history
  - Action (A): Movie to recommend
  - Reward (R): User engagement signal (rating)
  - Goal: Maximize cumulative reward

Algorithm: {trainer_obj.algorithm}
  Thompson Sampling: Maintains Beta distributions per movie, samples optimistically
  UCB: Upper Confidence Bound balances exploration-exploitation
  Contextual: Uses user state vectors from Member 3's LSTM model

2. TRAINING RESULTS
------------------
Total Interactions: {stats['total_interactions']:,}
Total Reward: {stats['total_reward']:.2f}
Average Reward: {stats['avg_reward']:.4f}
Unique Users: {stats['n_users']:,}
Unique Movies: {stats['n_movies']}

3. REWARD STRUCTURE
------------------
Rating >= 4.0  → Reward = +2.0 (High Engagement)
Rating < 4.0   → Reward = +1.0 (Moderate Engagement)
Not Watched    → Reward = -1.0 (No Engagement)

4. TOP PERFORMING MOVIES
------------------------
"""
    
    # Get top movies
    movie_avg_rewards = {}
    for movie_idx, rewards in trainer_obj.training_metrics['reward_by_movie'].items():
        if rewards:
            movie_avg_rewards[movie_idx] = np.mean(rewards)
    
    top_movies = sorted(movie_avg_rewards.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for rank, (movie_idx, avg_reward) in enumerate(top_movies, 1):
        movie_id = trainer_obj.idx_to_movie.get(movie_idx)
        movie_info = trainer_obj.movies_df[trainer_obj.movies_df['movieId'] == movie_id]
        if len(movie_info) > 0:
            title = movie_info.iloc[0]['title']
            count = trainer_obj.training_metrics['movie_counts'].get(movie_idx, 0)
            report += f"{rank}. {title}\n"
            report += f"   Avg Reward: {avg_reward:.3f}, Recommendations: {count}\n"
    
    report += f"""

5. INTEGRATION WITH OTHER COMPONENTS
------------------------------------
- Member 1 (Graph Model): Provides static node embeddings
- Member 2 (Baseline): Random Forest reference baseline
- Member 3 (LSTM): Provides 128-dim state vectors for contextual recommendations
- Member 4 (This system): Uses states to optimize recommendations via RL

6. KEY FEATURES
---------------
✓ Multi-Armed Bandit framework
✓ Thompson Sampling for exploration-exploitation
✓ Contextual recommendations using LSTM states
✓ Online learning capability
✓ Diversity-aware recommendations
✓ Real-time user feedback integration

7. PERFORMANCE CHARACTERISTICS
------------------------------
- Hit Rate: Measures if recommended movie is in user's future interactions
- Coverage: Percentage of catalog that can be recommended
- Diversity: Recommendation variety based on genres
- Novelty: Tendency to recommend lesser-known items
- Serendipity: Surprising but relevant recommendations

8. RECOMMENDATIONS FOR DEPLOYMENT
---------------------------------
1. Continuously collect user feedback to update arm statistics
2. Periodically re-sample arms to discover emerging preferences
3. Balance between exploitation (best known movies) and exploration
4. Monitor recommendation diversity to avoid filter bubbles
5. Use contextual information from LSTM for personalization

================================================================================
"""
    
    # Save report
    report_path = results_dir / "comprehensive_analysis.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Analysis report saved to: {report_path}")
    
    return report


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='RL Recommender System - Training and Evaluation'
    )
    parser.add_argument(
        '--algorithm',
        choices=['thompson', 'ucb', 'contextual'],
        default='thompson',
        help='Bandit algorithm to use'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of training samples (None = all)'
    )
    parser.add_argument(
        '--test-users',
        type=int,
        default=10,
        help='Number of users to test recommendations for'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training if model exists'
    )
    
    args = parser.parse_args()
    
    output_dir, results_dir = setup_directories()
    agent_path = output_dir / f"mab_agent_{args.algorithm}.pkl"
    
    # Train or load
    if not args.skip_training or not agent_path.exists():
        trainer, agent_path, test_metrics = train_and_evaluate(
            algorithm=args.algorithm,
            n_samples=args.samples
        )
    else:
        from train_mab import RecommenderTrainer
        logger.info(f"Loading existing agent from {agent_path}")
        trainer = RecommenderTrainer(data_dir="data", algorithm=args.algorithm)
        # Load agent
        import pickle
        with open(agent_path, 'rb') as f:
            trainer.agent = pickle.load(f)
    
    # Test recommendations
    test_recommendations(str(agent_path), n_test_users=args.test_users)
    
    # Generate analysis report
    generate_analysis_report(trainer, str(agent_path))
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL PIPELINES COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Model saved: models/mab_agent_{args.algorithm}.pkl")
    logger.info(f"Results saved: results/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
