"""
Multi-Armed Bandit Recommendation System
=========================================
Member 4: Reinforcement Learning Recommender

This system uses Multi-Armed Bandit algorithms to optimize movie recommendations
by learning user behavior patterns and maximizing long-term engagement rewards.

Core Algorithm: Thompson Sampling & Upper Confidence Bound (UCB)
"""

import numpy as np
import pandas as pd
import json
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pickle
import os
from datetime import datetime


class MovieBandit:
    """
    Multi-Armed Bandit for Movie Recommendations
    
    Each movie is an "arm" in the bandit problem.
    We maintain reward statistics for each user-movie pair and use Thompson Sampling
    to decide which movie to recommend next.
    """
    
    def __init__(self, n_arms: int, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize the Thompson Sampling Bandit
        
        Args:
            n_arms: Number of movies (arms)
            alpha: Beta distribution alpha parameter (success count)
            beta: Beta distribution beta parameter (failure count)
        """
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * alpha
        self.beta = np.ones(n_arms) * beta
        self.rewards_history = defaultdict(list)
        
    def sample(self) -> int:
        """Thompson Sampling - sample from Beta distribution for each arm"""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        """Update bandit with observed reward"""
        # Normalize reward to binary (success/failure)
        success = 1 if reward > 0 else 0
        failure = 1 if reward <= 0 else 0
        
        self.alpha[arm] += success
        self.beta[arm] += failure
        self.rewards_history[arm].append(reward)
    
    def get_arm_stats(self, arm: int) -> Dict:
        """Get statistics for an arm"""
        if len(self.rewards_history[arm]) == 0:
            return {"mean_reward": 0, "count": 0, "std": 0}
        
        rewards = np.array(self.rewards_history[arm])
        return {
            "mean_reward": np.mean(rewards),
            "count": len(rewards),
            "std": np.std(rewards) if len(rewards) > 1 else 0,
            "alpha": self.alpha[arm],
            "beta": self.beta[arm]
        }


class UCBBandit:
    """Upper Confidence Bound (UCB) Algorithm for Bandits"""
    
    def __init__(self, n_arms: int, c: float = 1.0):
        """
        Initialize UCB Bandit
        
        Args:
            n_arms: Number of arms (movies)
            c: Exploration constant
        """
        self.n_arms = n_arms
        self.c = c
        self.arm_counts = np.zeros(n_arms)
        self.arm_rewards = np.zeros(n_arms)
        self.total_plays = 0
        self.rewards_history = defaultdict(list)
    
    def select_arm(self) -> int:
        """Select arm using UCB algorithm"""
        ucb_values = np.zeros(self.n_arms)
        
        for arm in range(self.n_arms):
            if self.arm_counts[arm] == 0:
                # Unvisited arms get highest priority
                ucb_values[arm] = float('inf')
            else:
                exploitation = self.arm_rewards[arm] / self.arm_counts[arm]
                exploration = np.sqrt(np.log(self.total_plays) / self.arm_counts[arm])
                ucb_values[arm] = exploitation + self.c * exploration
        
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """Update UCB with observed reward"""
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward
        self.total_plays += 1
        self.rewards_history[arm].append(reward)
    
    def get_arm_stats(self, arm: int) -> Dict:
        """Get statistics for an arm"""
        if self.arm_counts[arm] == 0:
            return {"mean_reward": 0, "count": 0, "ucb": float('inf')}
        
        mean = self.arm_rewards[arm] / self.arm_counts[arm]
        ucb = mean + self.c * np.sqrt(np.log(self.total_plays) / self.arm_counts[arm])
        return {
            "mean_reward": mean,
            "count": int(self.arm_counts[arm]),
            "ucb": ucb
        }


class ContextualBandit:
    """
    Contextual Multi-Armed Bandit
    
    Takes user context (128-dim state from LSTM) into account when making decisions.
    """
    
    def __init__(self, n_arms: int, context_dim: int = 128, learning_rate: float = 0.01):
        """
        Initialize Contextual Bandit
        
        Args:
            n_arms: Number of movies (arms)
            context_dim: Dimension of user context vector
            learning_rate: Learning rate for weight updates
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.learning_rate = learning_rate
        
        # Initialize weights for each arm
        self.weights = np.random.randn(n_arms, context_dim) * 0.01
        self.arm_counts = np.zeros(n_arms)
        self.rewards_history = defaultdict(list)
    
    def select_arm(self, context: np.ndarray) -> int:
        """Select arm based on context"""
        # Compute predicted rewards for each arm given context
        scores = np.dot(self.weights, context)
        return np.argmax(scores)
    
    def update(self, arm: int, context: np.ndarray, reward: float):
        """Linear UCB update"""
        # Simple gradient update
        prediction = np.dot(self.weights[arm], context)
        error = reward - prediction
        self.weights[arm] += self.learning_rate * error * context
        
        self.arm_counts[arm] += 1
        self.rewards_history[arm].append(reward)
    
    def get_predictions(self, context: np.ndarray) -> np.ndarray:
        """Get predicted rewards for all arms given context"""
        return np.dot(self.weights, context)


class RLRecommenderAgent:
    """
    Main Reinforcement Learning Recommender Agent
    Orchestrates the full recommendation pipeline
    """
    
    def __init__(self, n_movies: int, state_dim: int = 128, 
                 algorithm: str = "thompson"):
        """
        Initialize RL Agent
        
        Args:
            n_movies: Total number of movies
            state_dim: Dimension of state vector from LSTM
            algorithm: 'thompson', 'ucb', or 'contextual'
        """
        self.n_movies = n_movies
        self.state_dim = state_dim
        self.algorithm = algorithm
        
        if algorithm == "thompson":
            self.bandit = MovieBandit(n_movies)
        elif algorithm == "ucb":
            self.bandit = UCBBandit(n_movies)
        elif algorithm == "contextual":
            self.bandit = ContextualBandit(n_movies, context_dim=state_dim)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.user_history = defaultdict(list)  # user_id -> [(movie_id, reward)]
        self.recommendation_history = []
        self.total_reward = 0
        self.total_interactions = 0
    
    def recommend(self, user_id: int, user_state: Optional[np.ndarray] = None,
                  top_k: int = 1) -> List[int]:
        """
        Recommend movies for a user
        
        Args:
            user_id: User ID
            user_state: Optional 128-dim state vector from LSTM
            top_k: Return top k recommendations
        
        Returns:
            List of recommended movie indices
        """
        top_k = max(1, min(top_k, self.n_movies))

        if self.algorithm == "contextual" and user_state is not None:
            scores = self.bandit.get_predictions(user_state)
        elif self.algorithm == "thompson":
            denom = self.bandit.alpha + self.bandit.beta
            scores = np.divide(
                self.bandit.alpha,
                np.where(denom == 0, 1.0, denom),
                dtype=float
            )
        elif self.algorithm == "ucb":
            counts = self.bandit.arm_counts
            rewards = self.bandit.arm_rewards
            means = np.divide(rewards, np.where(counts == 0, 1.0, counts), dtype=float)
            if self.bandit.total_plays <= 1:
                exploration = np.ones_like(means)
            else:
                exploration = np.sqrt(
                    np.log(self.bandit.total_plays) / np.where(counts == 0, 1.0, counts)
                )
            scores = means + self.bandit.c * exploration
            scores[counts == 0] = np.inf
        else:
            scores = np.zeros(self.n_movies, dtype=float)

        scores = np.array(scores)
        if top_k == 1:
            recommendations = [int(np.argmax(scores))]
        else:
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            recommendations = top_idx[np.argsort(scores[top_idx])[::-1]].astype(int).tolist()

        return recommendations
    
    def observe_reward(self, user_id: int, movie_id: int, rating: float,
                      user_state: Optional[np.ndarray] = None):
        """
        Observe reward from user interaction
        
        Args:
            user_id: User ID
            movie_id: Recommended movie ID
            rating: User rating (1-5 scale)
            user_state: Optional state vector for contextual update
        """
        # Convert rating to reward
        if rating >= 4:
            reward = 2.0  # High engagement
        elif rating >= 1:
            reward = 1.0  # Moderate engagement
        else:
            reward = -1.0  # No engagement / skip
        
        # Update bandit
        if self.algorithm == "contextual" and user_state is not None:
            self.bandit.update(movie_id, user_state, reward)
        else:
            self.bandit.update(movie_id, reward)
        
        # Track history
        self.user_history[user_id].append((movie_id, reward))
        self.total_reward += reward
        self.total_interactions += 1
        
        return reward
    
    def get_stats(self) -> Dict:
        """Get overall agent statistics"""
        return {
            "algorithm": self.algorithm,
            "total_interactions": self.total_interactions,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(1, self.total_interactions),
            "n_users": len(self.user_history),
            "n_movies": self.n_movies
        }
    
    def save(self, filepath: str):
        """Save agent to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str):
        """Load agent from disk"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def compute_reward(rating: float) -> float:
    """Convert MovieLens rating to reward signal"""
    if rating >= 4:
        return 2.0  # High engagement
    elif rating >= 1:
        return 1.0  # Moderate engagement
    else:
        return -1.0  # No engagement


def compute_engagement_score(ratings: List[float]) -> float:
    """Compute engagement score from a list of ratings"""
    if not ratings:
        return 0.0
    return np.mean([compute_reward(r) for r in ratings])


if __name__ == "__main__":
    print("Multi-Armed Bandit Recommender System Initialized")
    print("Available algorithms: thompson, ucb, contextual")
