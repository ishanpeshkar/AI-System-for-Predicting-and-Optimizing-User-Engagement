# Frontend Integration Guide

This document is for the member integrating the Reinforcement Learning recommender with the frontend.

## What this model provides
The model recommends movies for a user based on learned reward patterns.

### Inputs
Minimum input:
- `userId`

Optional input for future contextual mode:
- `recentMovieIds` or `recentMovieIdxSequence` from the user's last watched movies

### Output
A recommendation list like:

```json
[
  {
    "rank": 1,
    "movieId": 592,
    "movie_idx": 507,
    "title": "Batman (1989)",
    "mean_rating": 2.0,
    "times_recommended": 14
  }
]
```

## Recommended integration architecture
The frontend should **not** load the `.pkl` model directly.
Use a backend service/API layer.

### Flow
1. Frontend sends `userId` to backend.
2. Backend loads the trained model.
3. Backend generates recommendations.
4. Backend returns JSON to frontend.
5. Frontend displays movie cards/list.

## Backend contract suggestion
### Endpoint
`POST /api/recommendations`

### Request body
```json
{
  "userId": 123,
  "topK": 10
}
```

### Response body
```json
{
  "userId": 123,
  "algorithm": "thompson",
  "recommendations": [
    {
      "rank": 1,
      "movieId": 592,
      "movie_idx": 507,
      "title": "Batman (1989)",
      "mean_rating": 2.0,
      "times_recommended": 14
    }
  ]
}
```

## Python backend example
Use the existing utilities in this folder.

```python
from prediction_eval import load_predictor

predictor = load_predictor("models/mab_agent_thompson.pkl", "data")


def get_recommendations(user_id: int, top_k: int = 10):
    recs = predictor.recommend_for_user(user_id=user_id, top_k=top_k)
    return {
        "userId": int(user_id),
        "algorithm": "thompson",
        "recommendations": recs
    }
```

## If contextual mode is used later
If Member 3's LSTM output is available, backend can switch to contextual recommendations.

Needed inputs:
- user watch history
- mapping through `movie2idx.json`
- LSTM state extraction through `lstm_integration.py`

The frontend API can stay the same if backend handles the extra processing internally.

## What the frontend should display
For each recommendation:
- movie title
- rank
- optional score (`mean_rating`)
- optional metadata from `movies.csv`

## Important implementation notes
- Keep model loading on server startup, not per request.
- Return only JSON-safe Python types.
- If `userId` is new, still return top learned movies from the trained model.
- If movie posters/details are needed, join `movieId` with your movie metadata layer.

## Files the integrator needs
- `models/mab_agent_thompson.pkl`
- `prediction_eval.py`
- `mab_recommender.py`
- `data/movies.csv`

## Current verified command
```bash
python main.py --algorithm thompson --samples 5000
```

This has already been run successfully in this folder.
