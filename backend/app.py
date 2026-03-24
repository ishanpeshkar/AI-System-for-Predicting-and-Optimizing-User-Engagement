from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator


RepoRoot = Path(__file__).resolve().parents[1]

BaselineDir = RepoRoot / "Feature Engineering + Baseline Model"
BaselineDatasetDir = BaselineDir / "ml-latest-small"
BaselineModelPath = BaselineDir / "rf_model.pkl"

GraphDir = RepoRoot / "GraphModel"
GraphEmbeddingsPath = GraphDir / "movie_embeddings.txt"

SequentialDir = RepoRoot / "Sequential Deep Learning Model"
SequentialDataDir = SequentialDir / "data"
SequentialModelWeights = SequentialDir / "models" / "lstm_model.pt"

RLDir = RepoRoot / "Reinforcement_Learning_Recommender"
RLAgentPath = RLDir / "models" / "mab_agent_thompson.pkl"
RLDataDir = RLDir / "data"


# Allow importing the sequential and RL modules from their folders.
sys.path.insert(0, str(SequentialDir / "src"))
sys.path.insert(0, str(RLDir))

from model import AttentionLSTM  # type: ignore
from prediction_eval import load_predictor  # type: ignore


class RecommendationRequest(BaseModel):
    userId: int = Field(..., ge=1)
    topK: int = Field(10, ge=1, le=50)
    algorithm: Literal[
        "RL (Bandit)",
        "Baseline (RF)",
        "Graph (Node2Vec)",
        "Sequential (LSTM)",
    ] = "RL (Bandit)"


class Recommendation(BaseModel):
    rank: int
    movieId: int
    title: str
    genres: Optional[str] = None
    score: Optional[float] = None


class RecommendationResponse(BaseModel):
    userId: int
    algorithm: str
    recommendations: List[Recommendation]


class UserIntelligence(BaseModel):
    userId: int
    dominantGenre: str
    recentTrend: str
    engagementLevel: str
    sequenceInsight: str
    confidenceScore: float


class EngagementEvent(BaseModel):
    userId: int
    recommendationId: int
    movieTitle: str
    clicked: bool
    watched: bool
    rating: Optional[float] = None
    timestamp: str


class FeedbackRequest(BaseModel):
    userId: int
    movieId: int
    movieTitle: str
    feedbackType: str  # 'liked' | 'watched' | 'disliked'


def _safe_mean(x: List[float]) -> float:
    return float(np.mean(x)) if x else 0.0


def _load_movies_map(movies_df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for _, row in movies_df.iterrows():
        mid = int(row["movieId"])
        out[mid] = {"title": str(row["title"]), "genres": str(row.get("genres", ""))}
    return out


@lru_cache(maxsize=1)
def _parse_graph_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      all_movie_ids: (N,)
      mat: (N, D) float32
      mat_norm: (N,) float32
    """
    movie_vecs: Dict[int, np.ndarray] = {}

    with open(GraphEmbeddingsPath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = parts[0]
            if not label.startswith("movie_") or not label.endswith(".0"):
                continue
            # label is like: movie_<movieId>.0
            mid = int(float(label[len("movie_") : -2]))
            vals = list(map(float, parts[1:]))
            movie_vecs[mid] = np.asarray(vals, dtype=np.float32)

    all_movie_ids = np.asarray(list(movie_vecs.keys()), dtype=np.int64)
    mat = np.stack([movie_vecs[mid] for mid in all_movie_ids], axis=0)
    mat_norm = np.linalg.norm(mat, axis=1).astype(np.float32)
    return all_movie_ids, mat, mat_norm


def create_app() -> FastAPI:
    app = FastAPI(title="ML Recommendation API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    movies_baseline = pd.read_csv(BaselineDatasetDir / "movies.csv")
    ratings_baseline = pd.read_csv(BaselineDatasetDir / "ratings.csv")
    ratings_df = pd.read_csv(RLDataDir / "ratings.csv")
    movies_df = pd.read_csv(RLDataDir / "movies.csv")
    movies_graph = pd.read_csv(GraphDir / "movies.csv")
    ratings_graph = pd.read_csv(GraphDir / "ratings.csv")
    movies_seq = pd.read_csv(SequentialDataDir / "movies.csv")
    ratings_seq = pd.read_csv(SequentialDataDir / "ratings.csv")

    # Baseline precompute
    rf_model: BaseEstimator = torch.load  # type: ignore[attr-defined]
    rf_model = None  # silence type checker; overwritten below
    import joblib

    rf_model = joblib.load(BaselineModelPath)
    expected_cols: List[str] = list(getattr(rf_model, "feature_names_in_", []))

    # Precompute movie features (excluding user-specific columns).
    movie_stats = (
        ratings_baseline.groupby("movieId")
        .agg(movie_avg_rating=("rating", "mean"), movie_rating_count=("rating", "count"))
        .reset_index()
    )
    movies_with_stats = movies_baseline.merge(movie_stats, on="movieId", how="left")
    movies_with_stats = movies_with_stats.fillna(
        {"movie_avg_rating": float(ratings_baseline["rating"].mean()), "movie_rating_count": 1}
    )

    # Genres one-hot
    genre_dummies = movies_with_stats["genres"].str.get_dummies(sep="|")

    # Year extraction from title "Title (YYYY)"
    movies_with_stats["year"] = (
        movies_with_stats["title"].str.extract(r"\((\d{4})\)")[0].astype(float)
    )
    movies_with_stats["year"] = movies_with_stats["year"].fillna(
        movies_with_stats["year"].median()
    ).astype(int)

    base_movie_features = movies_with_stats[
        ["movieId", "movie_avg_rating", "movie_rating_count", "year"]
    ].copy()
    base_movie_features = base_movie_features.join(genre_dummies)

    movie_id_list = base_movie_features["movieId"].astype(int).values
    movies_map_baseline = _load_movies_map(movies_baseline)

    # Sequential precompute (mapping + model)
    with open(SequentialDataDir / "movie2idx.json", "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    movie2idx: Dict[int, int] = {int(k): int(v) for k, v in raw_mapping.items()}
    idx2movie: Dict[int, int] = {v: k for k, v in movie2idx.items()}

    seq_vocab_size = 9725
    seq_embed_dim = 64
    seq_hidden_dim = 128
    seq_output_dim = seq_vocab_size
    seq_device = torch.device("cpu")
    seq_model = AttentionLSTM(seq_vocab_size, seq_embed_dim, seq_hidden_dim, seq_output_dim).to(seq_device)
    seq_state_dict = torch.load(SequentialModelWeights, map_location=seq_device)
    seq_model.load_state_dict(seq_state_dict)
    seq_model.eval()

    movies_map_seq = _load_movies_map(movies_seq)

    # RL predictor
    rl_predictor = load_predictor(str(RLAgentPath), str(RLDataDir))

    movies_map_graph = _load_movies_map(movies_graph)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    def _recommend_baseline(user_id: int, top_k: int) -> List[Recommendation]:
        user_ratings = ratings_baseline[ratings_baseline["userId"] == user_id]
        if user_ratings.empty:
            user_avg = float(ratings_baseline["rating"].mean())
            user_count = 1
            watched: Set[int] = set()
        else:
            user_avg = float(user_ratings["rating"].mean())
            user_count = int(user_ratings["rating"].count())
            watched = set(user_ratings["movieId"].astype(int).unique().tolist())

        X = base_movie_features.copy()
        X["user_avg_rating"] = user_avg
        X["user_rating_count"] = user_count

        if expected_cols:
            X = X.reindex(columns=expected_cols, fill_value=0)

        # Probability of liking (class 1)
        probs = rf_model.predict_proba(X)[:, 1]

        order = np.argsort(-probs)
        recs: List[Recommendation] = []
        for idx in order:
            mid = int(movie_id_list[idx])
            if mid in watched:
                continue
            meta = movies_map_baseline.get(mid, {})
            recs.append(
                Recommendation(
                    rank=len(recs) + 1,
                    movieId=mid,
                    title=meta.get("title", ""),
                    genres=meta.get("genres", ""),
                    score=float(probs[idx]),
                )
            )
            
            if len(recs) >= top_k:
                break
        return recs

    def _recommend_graph(user_id: int, top_k: int) -> List[Recommendation]:
        all_movie_ids, mat, mat_norm = _parse_graph_embeddings()
        ratings_u = ratings_graph[ratings_graph["userId"] == user_id].sort_values("timestamp")

        watched = set(ratings_u["movieId"].astype(int).unique().tolist())
        last_movies = ratings_u.tail(10)["movieId"].astype(int).tolist()

        if last_movies:
            # Build user vector as mean embedding of recently watched movies.
            movie_to_vec = {int(mid): mat[i] for i, mid in enumerate(all_movie_ids)}
            emb_list = [movie_to_vec[mid] for mid in last_movies if mid in movie_to_vec]
            user_vec = np.mean(emb_list, axis=0) if emb_list else np.mean(mat, axis=0)
        else:
            user_vec = np.mean(mat, axis=0)

        denom = (mat_norm * float(np.linalg.norm(user_vec))) + 1e-9
        sims = (mat @ user_vec) / denom

        order = np.argsort(-sims)
        recs: List[Recommendation] = []
        for idx in order:
            mid = int(all_movie_ids[idx])
            if mid in watched:
                continue
            meta = movies_map_graph.get(mid, {})
            recs.append(
                Recommendation(
                    rank=len(recs) + 1,
                    movieId=mid,
                    title=meta.get("title", ""),
                    genres=meta.get("genres", ""),
                    score=float(sims[idx]),
                )
            )
            if len(recs) >= top_k:
                break
        return recs

    def _recommend_sequential(user_id: int, top_k: int) -> List[Recommendation]:
        ratings_u = ratings_seq[ratings_seq["userId"] == user_id].sort_values("timestamp")
        if ratings_u.empty:
            seq = [0] * 10
            watched_idxs: Set[int] = set()
        else:
            last_movies = ratings_u.tail(10)["movieId"].astype(int).tolist()
            watched_idxs = {movie2idx.get(mid, 0) for mid in last_movies}
            seq_idxs = [movie2idx.get(mid, 0) for mid in last_movies]
            if len(seq_idxs) < 10:
                seq = [0] * (10 - len(seq_idxs)) + seq_idxs
            else:
                seq = seq_idxs[-10:]

        seq_tensor = torch.LongTensor([seq]).to(seq_device)
        with torch.no_grad():
            logits = seq_model(seq_tensor)[0]  # [vocab]
            probs = torch.softmax(logits, dim=0)

        # Pick more candidates than needed so we can filter watched items.
        k = top_k + len(watched_idxs) + 10
        k = min(int(k), int(probs.shape[0]))
        _, indices = torch.topk(probs, k=k)

        recs: List[Recommendation] = []
        for idx in indices.tolist():
            if idx == 0:
                continue
            if idx in watched_idxs:
                continue
            mid = idx2movie.get(idx)
            if mid is None:
                continue
            meta = movies_map_seq.get(int(mid), {})
            recs.append(
                Recommendation(
                    rank=len(recs) + 1,
                    movieId=int(mid),
                    title=meta.get("title", ""),
                    genres=meta.get("genres", ""),
                    score=float(probs[idx].item()),
                )
            )
            if len(recs) >= top_k:
                break
        return recs

    def _recommend_rl(user_id: int, top_k: int) -> List[Recommendation]:
        rec_dicts = rl_predictor.recommend_for_user(user_id=user_id, top_k=top_k)
        recs: List[Recommendation] = []
        for rec in rec_dicts:
            recs.append(
                Recommendation(
                    rank=int(rec.get("rank", len(recs) + 1)),
                    movieId=int(rec.get("movieId")),
                    title=str(rec.get("title", "")),
                    genres=str(rec.get("genres", "")) if rec.get("genres") is not None else None,
                    score=float(rec.get("mean_rating", 0.0)) if rec.get("mean_rating") is not None else None,
                )
            )
        return recs[:top_k]

    def rerank_by_user_genre(
        recommendations: List[Recommendation],
        user_id: int,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        top_k: int = 10,
    ) -> List[Recommendation]:
        """
        Rerank bandit recommendations by boosting movies whose
        primary genre matches the user's most-watched genre
        in their last 20 ratings.
        """
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if user_ratings.empty:
            return recommendations

        recent = user_ratings.sort_values("timestamp").tail(20)
        genre_count: Dict[str, int] = {}
        for movie_id in recent["movieId"].tolist():
            match = movies_df[movies_df["movieId"] == movie_id]
            if not match.empty:
                primary = str(match.iloc[0]["genres"]).split("|")[0]
                genre_count[primary] = genre_count.get(primary, 0) + 1

        if not genre_count:
            return recommendations

        top_genre = max(genre_count, key=genre_count.get)

        reranked: List[Recommendation] = []
        for rec in recommendations:
            score = float(rec.score or 0)
            genres = rec.genres or ""
            split_genres = genres.split("|") if genres else []
            primary_genre = split_genres[0] if split_genres else ""

            if primary_genre == top_genre:
                score = score * 1.35  # 35% boost for primary-genre match
            elif top_genre in genres:
                score = score * 1.15  # 15% boost when top genre appears as secondary
            else:
                score = score * 0.9  # slight penalty for non-matching genres
            reranked.append(
                Recommendation(
                    rank=rec.rank,
                    movieId=rec.movieId,
                    title=rec.title,
                    genres=rec.genres,
                    score=score,
                )
            )

        reranked.sort(key=lambda x: float(x.score or 0), reverse=True)

        final: List[Recommendation] = []
        for index, rec in enumerate(reranked[:top_k], start=1):
            final.append(
                Recommendation(
                    rank=index,
                    movieId=rec.movieId,
                    title=rec.title,
                    genres=rec.genres,
                    score=rec.score,
                )
            )
        return final

    @app.post("/api/recommendations", response_model=RecommendationResponse)
    def recommend(req: RecommendationRequest) -> RecommendationResponse:
        # Map frontend algorithm names to internal names
        algo_map = {
            "RL (Bandit)": "rl",
            "Baseline (RF)": "baseline",
            "Graph (Node2Vec)": "graph",
            "Sequential (LSTM)": "sequential",
        }
        algo_internal = algo_map.get(req.algorithm, "rl")

        if algo_internal == "baseline":
            recs = _recommend_baseline(req.userId, req.topK)
        elif algo_internal == "graph":
            recs = _recommend_graph(req.userId, req.topK)
        elif algo_internal == "sequential":
            recs = _recommend_sequential(req.userId, req.topK)
        else:
            recs = _recommend_rl(req.userId, req.topK)
            recs = rerank_by_user_genre(recs, req.userId, ratings_df, movies_df, req.topK)

        return RecommendationResponse(
            userId=req.userId,
            algorithm=req.algorithm,
            recommendations=recs,
        )

    @app.get("/api/user-intelligence/{user_id}", response_model=UserIntelligence)
    def get_user_intelligence(user_id: int) -> UserIntelligence:
        """
        Analyze user behavior and return personalized intelligence.
        This endpoint provides insights into user preferences, patterns, and confidence scores.
        """
        # Analyze user's watch history
        user_ratings = ratings_baseline[ratings_baseline["userId"] == user_id].sort_values("timestamp")

        if user_ratings.empty:
            return UserIntelligence(
                userId=user_id,
                dominantGenre="Mixed",
                recentTrend="Varied",
                engagementLevel="Low",
                sequenceInsight="User has no watch history. Starting fresh recommendations.",
                confidenceScore=0.71,
            )

        # Get watched movies and their genres
        watched_movies = user_ratings.tail(20)["movieId"].unique()
        movie_data = movies_baseline[movies_baseline["movieId"].isin(watched_movies)]

        # Extract dominant genre
        all_genres: List[str] = []
        for genres_str in movie_data["genres"].dropna():
            all_genres.extend(genres_str.split("|"))

        genre_counts = {}
        for genre in all_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        dominant_genre = max(genre_counts, key=genre_counts.get) if genre_counts else "Drama"

        # Recent trend
        recent_movies = user_ratings.tail(5)
        recent_genres_list: List[str] = []
        for _, row in recent_movies.iterrows():
            movie = movies_baseline[movies_baseline["movieId"] == row["movieId"]]
            if not movie.empty:
                genres = movie.iloc[0]["genres"]
                if isinstance(genres, str):
                    recent_genres_list.extend(genres.split("|")[:2])

        recent_trend = " / ".join(set(recent_genres_list[:3])) if recent_genres_list else dominant_genre

        # Engagement level
        avg_rating = float(user_ratings["rating"].mean())
        if avg_rating >= 4.0:
            engagement = "High"
            confidence = 0.82
        elif avg_rating >= 3.0:
            engagement = "Medium"
            confidence = 0.76
        else:
            engagement = "Low"
            confidence = 0.71

        # Sequence insight
        if len(user_ratings) <= 5:
            insight = f"User is new. Recently watched {dominant_genre} content."
        elif avg_rating >= 4.0:
            insight = f"User is transitioning from {recent_genres_list[0] if recent_genres_list else 'Unknown'} → {dominant_genre}.\nRecent 5 movies show strong {dominant_genre} theme."
        else:
            insight = f"User shows consistent preference for {dominant_genre} and\n{recent_genres_list[1] if len(recent_genres_list) > 1 else 'similar'} content."

        return UserIntelligence(
            userId=user_id,
            dominantGenre=dominant_genre,
            recentTrend=recent_trend,
            engagementLevel=engagement,
            sequenceInsight=insight,
            confidenceScore=confidence,
        )

    @app.get("/api/user-profile/{user_id}")
    def get_user_profile(user_id: int) -> Dict[str, Any]:
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if user_ratings.empty:
            return {
                "found": False,
                "total_ratings": 0,
                "avg_rating": 0,
                "watch_history": [],
                "top_genres": [],
                "dominant_genre": "Unknown",
            }

        sorted_ratings = user_ratings.sort_values("timestamp")
        movie_ids = sorted_ratings["movieId"].tolist()
        recent_ids = movie_ids[-10:]
        recent_movies: List[Dict[str, Any]] = []
        for mid in recent_ids:
            match = movies_df[movies_df["movieId"] == mid]
            if not match.empty:
                recent_movies.append(
                    {
                        "movieId": int(mid),
                        "title": str(match.iloc[0]["title"]),
                        "genres": str(match.iloc[0]["genres"]),
                    }
                )

        all_genres: List[str] = []
        for mid in movie_ids:
            match = movies_df[movies_df["movieId"] == mid]
            if not match.empty:
                all_genres.extend(str(match.iloc[0]["genres"]).split("|"))

        genre_counts: Dict[str, int] = {}
        for genre in all_genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "found": True,
            "total_ratings": len(user_ratings),
            "avg_rating": round(float(user_ratings["rating"].mean()), 1),
            "watch_history": recent_movies,
            "top_genres": [g[0] for g in top_genres],
            "dominant_genre": top_genres[0][0] if top_genres else "Unknown",
        }

    @app.post("/api/feedback")
    def submit_feedback(request: FeedbackRequest) -> Dict[str, Any]:
        reward_map = {"liked": 2.0, "watched": 1.0, "disliked": -1.0}
        reward = reward_map.get(request.feedbackType, 0.0)
        entry = {
            "user_id": request.userId,
            "movie_id": request.movieId,
            "movie_title": request.movieTitle,
            "feedback_type": request.feedbackType,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        }

        feedback_dir = RLDir / "data"
        feedback_dir.mkdir(exist_ok=True)
        feedback_file = feedback_dir / "feedback_store.json"

        all_feedback: List[Dict[str, Any]] = []
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, "r", encoding="utf-8") as file:
                    all_feedback = json.load(file)
            except json.JSONDecodeError:
                all_feedback = []

        all_feedback.append(entry)
        with open(feedback_file, "w", encoding="utf-8") as file:
            json.dump(all_feedback, file, indent=2)

        bandit_file = RLDir / "models" / "bandit_model.json"
        if os.path.exists(bandit_file):
            with open(bandit_file, "r", encoding="utf-8") as file:
                bandit = json.load(file)
            key = str(request.movieId)
            if key not in bandit:
                bandit[key] = {"alpha": 1.0, "beta": 1.0, "total_feedback": 0}

            if reward > 0:
                bandit[key]["alpha"] += reward
            else:
                bandit[key]["beta"] += abs(reward)

            bandit[key]["total_feedback"] = bandit[key].get("total_feedback", 0) + 1

            with open(bandit_file, "w", encoding="utf-8") as file:
                json.dump(bandit, file, indent=2)

        return {
            "success": True,
            "reward_applied": reward,
            "total_feedback": len(all_feedback),
        }

    @app.get("/api/feedback/stats")
    def get_feedback_stats() -> Dict[str, Any]:
        feedback_file = RLDir / "data" / "feedback_store.json"
        if not os.path.exists(feedback_file):
            return {
                "total_feedback": 0,
                "positive_rate": 0,
                "liked": 0,
                "watched": 0,
                "disliked": 0,
                "model_updates": 0,
                "recent": [],
            }

        with open(feedback_file, "r", encoding="utf-8") as file:
            feedback = json.load(file)

        if not feedback:
            return {
                "total_feedback": 0,
                "positive_rate": 0,
                "liked": 0,
                "watched": 0,
                "disliked": 0,
                "model_updates": 0,
                "recent": [],
            }

        liked = sum(1 for item in feedback if item.get("feedback_type") == "liked")
        watched = sum(1 for item in feedback if item.get("feedback_type") == "watched")
        disliked = sum(1 for item in feedback if item.get("feedback_type") == "disliked")
        positive = liked + watched

        return {
            "total_feedback": len(feedback),
            "positive_rate": round((positive / len(feedback)) * 100, 1),
            "liked": liked,
            "watched": watched,
            "disliked": disliked,
            "model_updates": len(feedback),
            "recent": feedback[-5:][::-1],
        }

    @app.post("/api/engagement-track")
    def track_engagement(event: EngagementEvent) -> Dict[str, str]:
        """
        Track user engagement with recommendations.
        Used to collect data on which recommendations users interact with.
        """
        import logging

        logger = logging.getLogger("engagement")
        logger.info(
            f"User {event.userId} - Movie: {event.movieTitle} - Clicked: {event.clicked} - "
            f"Watched: {event.watched} - Rating: {event.rating}"
        )

        return {"status": "tracked", "message": f"Engagement recorded for user {event.userId}"}

    return app


app = create_app()

