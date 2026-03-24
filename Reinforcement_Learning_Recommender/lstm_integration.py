"""
LSTM State Integration Module
=============================

Integrates with Member 3's Sequential Deep Learning Model
Extracts state vectors and uses them for contextual recommendations
"""

import numpy as np
import torch
import json
import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import logging

logger = logging.getLogger(__name__)


class LSTMStateIntegrator:
    """
    Bridges between Member 3's LSTM model output and RL Agent
    
    The LSTM model produces 128-dim state vectors from user viewing history.
    These states are passed to the contextual bandit for better recommendations.
    """
    
    def __init__(self, lstm_model_path: str, movie2idx_path: str,
                 embed_dim: int = 64, hidden_dim: int = 128,
                 vocab_size: int = 9725):
        """
        Initialize integrator
        
        Args:
            lstm_model_path: Path to trained LSTM model
            movie2idx_path: Path to movie2idx.json mapping
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension (state dimension)
            vocab_size: Total vocabulary size
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.state_dim = hidden_dim
        
        # Load movie mapping
        with open(movie2idx_path, 'r') as f:
            self.movie2idx = json.load(f)
        
        self.idx2movie = {int(v): int(k) for k, v in self.movie2idx.items()}
        
        # Load LSTM model
        self.model = self._load_model(lstm_model_path)
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"LSTM Model loaded from {lstm_model_path}")
        logger.info(f"State dimension: {self.state_dim}")
    
    def _load_model(self, model_path: str):
        """Load LSTM model architecture and weights"""
        try:
            source_model_path = Path(__file__).parent.parent / "Sequential Deep Learning Model" / "src" / "model.py"
            spec = importlib.util.spec_from_file_location("member3_model", source_model_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load model definition from {source_model_path}")

            member3_model = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(member3_model)
            AttentionLSTM = member3_model.AttentionLSTM
            
            model = AttentionLSTM(
                vocab_size=self.vocab_size,
                embedding_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.vocab_size
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            # Return mock model for testing
            logger.warning("Returning mock LSTM model")
            return MockLSTMModel(self.vocab_size, self.embed_dim, 
                                self.hidden_dim, self.state_dim)
    
    def movieId_to_idx(self, movie_id: Union[int, str]) -> Optional[int]:
        """Convert movieId to vocabulary index"""
        key = str(movie_id)
        return self.movie2idx.get(key)
    
    def idx_to_movieId(self, idx: int) -> Optional[int]:
        """Convert vocabulary index to movieId"""
        return self.idx2movie.get(idx)
    
    def pad_sequence(self, movie_indices: List[int], 
                    length: int = 10) -> np.ndarray:
        """
        Pad movie sequence to fixed length
        
        Args:
            movie_indices: List of movie indices
            length: Target sequence length
        
        Returns:
            Padded sequence as numpy array
        """
        sequence = np.zeros(length, dtype=np.int64)
        
        if len(movie_indices) >= length:
            sequence = np.array(movie_indices[-length:], dtype=np.int64)
        else:
            sequence[length - len(movie_indices):] = movie_indices
        
        return sequence
    
    def extract_state(self, movie_sequence: List[int],
                     return_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract 128-dim state vector from movie sequence
        
        Args:
            movie_sequence: List of movie indices or movieIds
            return_tensor: If True, return torch tensor; else numpy array
        
        Returns:
            State vector of shape (128,) or (hidden_dim,)
        """
        # Convert movieIds to indices if needed
        indices = []
        for movie in movie_sequence:
            # Check if it's possibly a movieId (larger numbers)
            if movie > 100:  # Heuristic: movieIds are usually >100
                idx = self.movieId_to_idx(movie)
                indices.append(idx if idx is not None else 0)
            else:
                indices.append(movie)
        
        # Pad sequence
        padded = self.pad_sequence(indices, length=10)
        
        # Convert to tensor
        input_tensor = torch.LongTensor(padded).unsqueeze(0).to(self.device)
        
        # Forward pass to get hidden state
        with torch.no_grad():
            # Get last hidden state
            if hasattr(self.model, 'lstm'):
                _, (hidden, _) = self.model.lstm(
                    self.model.embedding(input_tensor)
                )
                state = hidden[-1]  # Last layer, last sequence position
            else:
                # Mock model
                state = torch.randn(1, self.state_dim, device=self.device)
        
        if return_tensor:
            return state.squeeze(0)
        else:
            return state.squeeze(0).cpu().numpy()
    
    def extract_states_batch(self, movie_sequences: List[List[int]]) -> np.ndarray:
        """
        Extract states for batch of sequences
        
        Args:
            movie_sequences: List of movie index sequences
        
        Returns:
            Array of shape (batch_size, state_dim)
        """
        states = []
        for seq in movie_sequences:
            state = self.extract_state(seq, return_tensor=False)
            states.append(state)
        
        return np.array(states)
    
    def get_state_statistics(self, states: np.ndarray) -> Dict:
        """Get statistics about state vectors"""
        return {
            'mean': np.mean(states, axis=0).tolist()[:5],  # First 5 dims
            'std': np.std(states, axis=0).tolist()[:5],
            'min': np.min(states),
            'max': np.max(states),
            'shape': states.shape
        }


class MockLSTMModel(torch.nn.Module):
    """
    Mock LSTM model for testing when real model unavailable
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, output_dim: int = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or vocab_size
        
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        """Forward pass through mock model"""
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return output, (hidden, cell)


class ContextualRecommendationEngine:
    """
    Full contextual recommendation engine combining LSTM states and RL agent
    """
    
    def __init__(self, rl_agent, lstm_integrator: LSTMStateIntegrator):
        """
        Initialize the engine
        
        Args:
            rl_agent: Trained RLRecommenderAgent
            lstm_integrator: LSTMStateIntegrator instance
        """
        self.rl_agent = rl_agent
        self.lstm_integrator = lstm_integrator
    
    def recommend_contextual(self, user_id: int, 
                           movie_history: List[int],
                           top_k: int = 10) -> List[Dict]:
        """
        Generate contextual recommendations based on user history
        
        Args:
            user_id: User ID
            movie_history: List of recently watched movie indices
            top_k: Number of recommendations
        
        Returns:
            List of recommendations with scores
        """
        # Extract state from history
        user_state = self.lstm_integrator.extract_state(movie_history)
        
        # Get contextual recommendations
        recommendations = self.rl_agent.recommend(
            user_id,
            user_state=user_state,
            top_k=top_k
        )
        
        return [
            {
                'rank': i + 1,
                'movie_idx': rec,
                'is_contextual': True
            }
            for i, rec in enumerate(recommendations)
        ]
    
    def explain_recommendation(self, user_id: int,
                             movie_history: List[int],
                             recommended_movie_idx: int) -> Dict:
        """
        Explain why a movie was recommended
        
        Args:
            user_id: User ID
            movie_history: User's watch history
            recommended_movie_idx: Recommended movie index
        
        Returns:
            Explanation dictionary
        """
        # Get state
        user_state = self.lstm_integrator.extract_state(movie_history)
        
        # Get bandit stats
        stats = self.rl_agent.bandit.get_arm_stats(recommended_movie_idx)
        
        # Compute state similarity to recommendation
        if isinstance(self.rl_agent.bandit, type(self.rl_agent.bandit)):
            if hasattr(self.rl_agent.bandit, 'weights'):
                score = np.dot(
                    self.rl_agent.bandit.weights[recommended_movie_idx],
                    user_state
                )
            else:
                score = stats.get('mean_reward', 0)
        else:
            score = stats.get('mean_reward', 0)
        
        return {
            'user_id': user_id,
            'recommended_movie_idx': recommended_movie_idx,
            'predicted_score': float(score),
            'mean_reward': float(stats.get('mean_reward', 0)),
            'recommendation_count': int(stats.get('count', 0)),
            'state_magnitude': float(np.linalg.norm(user_state)),
            'explanation': f"This movie has mean reward of {stats.get('mean_reward', 0):.3f} "
                          f"and aligns well with your viewing context"
        }


def create_contextual_engine(agent_path: str, lstm_model_path: str,
                            movie2idx_path: str) -> ContextualRecommendationEngine:
    """
    Factory function to create a full contextual engine
    
    Args:
        agent_path: Path to trained RL agent
        lstm_model_path: Path to LSTM model
        movie2idx_path: Path to movie2idx mapping
    
    Returns:
        ContextualRecommendationEngine instance
    """
    import pickle
    
    # Load agent
    with open(agent_path, 'rb') as f:
        agent = pickle.load(f)
    
    # Create integrator
    integrator = LSTMStateIntegrator(lstm_model_path, movie2idx_path)
    
    # Create engine
    engine = ContextualRecommendationEngine(agent, integrator)
    
    return engine


if __name__ == "__main__":
    print("LSTM Integration Module Ready")
