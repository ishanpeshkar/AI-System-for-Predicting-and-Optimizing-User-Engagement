# CinePulse Backend — FastAPI + 4 ML Models

**Recommendation engine with Graph, RF, LSTM, and RL models**

---

## ⚡ **Quick Start**

```bash
# 1. Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start server
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# API running at http://127.0.0.1:8000
# Docs at http://127.0.0.1:8000/docs
```

---

## 📁 **File Structure**

```
backend/
├── app.py                      # FastAPI application + endpoints
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment variables
└── __init__.py
```

---

## 🔧 **Configuration**

Create `.env` file from `.env.example`:

```env
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
BACKEND_RELOAD=true
CORS_ORIGINS=["http://localhost:5173", "http://127.0.0.1:5173"]
LOG_LEVEL=INFO
USE_GPU=false
MODEL_CACHE_SIZE=1000
```

Load with `config.py`:
```python
from config import settings
print(settings.backend_host)
print(settings.parsed_cors_origins)
```

---

## 🚀 **API Endpoints**

### **1. Health Check**
```
GET /health
Response: { "status": "ok" }
```

### **2. Recommendations** (Main endpoint)
```
POST /api/recommendations

Request:
{
  "userId": 1,
  "topK": 5,
  "algorithm": "RL (Bandit)" | "Baseline (RF)" | "Graph (Node2Vec)" | "Sequential (LSTM)"
}

Response:
{
  "userId": 1,
  "algorithm": "RL (Bandit)",
  "recommendations": [
    {
      "rank": 1,
      "movieId": 356,
      "title": "Forrest Gump (1994)",
      "genres": "Drama|Romance",
      "score": 0.923
    },
    ...
  ]
}
```

### **3. User Intelligence** (Engagement analytics)
```
GET /api/user-intelligence/{user_id}

Response:
{
  "userId": 1,
  "dominantGenre": "Sci-Fi",
  "recentTrend": "Space / Futuristic",
  "engagementLevel": "High",
  "sequenceInsight": "User is transitioning from Action → Sci-Fi...",
  "confidenceScore": 0.82
}
```

### **4. Engagement Tracking**
```
POST /api/engagement-track

Request:
{
  "userId": 1,
  "recommendationId": 356,
  "movieTitle": "Forrest Gump (1994)",
  "clicked": true,
  "watched": false,
  "rating": null,
  "timestamp": "2026-03-22T10:30:00Z"
}

Response:
{
  "status": "tracked",
  "message": "Engagement recorded for user 1"
}
```

---

## 🧠 **ML Models**

### **M1 - Graph (Node2Vec)**
- **Method**: Cosine similarity in 64-dim embedding space
- **Input**: User's recent 10 movies → project to embedding space
- **Output**: Top-K movies by structural similarity
- **File**: `GraphModel/movie_embeddings.txt`
- **Metric**: Embeddings 64-dim

### **M2 - Baseline (Random Forest)**
- **Method**: Engagement probability via RF classifier
- **Features**: Movie stats (rating, count, year, genres) + user stats (avg rating, count)
- **Output**: Top-K movies by P(engagement) score
- **File**: `Feature Engineering + Baseline Model/rf_model.pkl`
- **Metric**: AUC 0.74 (single model)

### **M3 - Sequential (LSTM)**
- **Method**: Attention LSTM on last 10 watched movies
- **Architecture**: 64-dim embeddings → 128-hidden LSTM → softmax over vocab
- **Output**: Top-K movies by next-watch probability
- **File**: `Sequential Deep Learning Model/models/lstm_model.pt`
- **Metric**: HR@10 8.14%

### **M4 - RL (Bandit)**
- **Method**: Thompson Sampling multi-armed bandit
- **Learning**: Balances exploration (discover) vs exploitation (confirm)
- **Output**: Top-K movies by estimated long-term reward
- **File**: `Reinforcement_Learning_Recommender/models/mab_agent_thompson.pkl`
- **Metric**: Regret -61% vs naive

---

## 📊 **Data Flow**

```
Request: { userId, topK, algorithm }
    ↓
[Algorithm Selection]
    ├─→ Graph:      Graph embeddings + cosine sim
    ├─→ Baseline:   RF predict_proba
    ├─→ LSTM:       Sequential forward pass
    └─→ RL:         Bandit policy
    ↓
[Filter]
    └─→ Remove already-watched movies
    └─→ Keep top K unwatched
    ↓
Response: [
  { rank, movieId, title, genres, score },
  { rank, movieId, title, genres, score },
  ...
]
```

---

## 🔌 **Integration Points**

### **Frontend → Backend**
1. Frontend sends `algorithm` in full name format
2. Backend maps to internal name (e.g., "RL (Bandit)" → "rl")
3. Calls corresponding `_recommend_*()` function
4. Returns ranked movies with scores

### **New Endpoints**
- **User Intelligence**: Analyzes watch history → genre/trend/engagement
- **Engagement Tracking**: Logs user interactions for feedback loop

---

## 🧪 **Testing**

### **Curl Examples**
```bash
# Health check
curl http://127.0.0.1:8000/health

# Get recommendations
curl -X POST http://127.0.0.1:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"userId": 1, "topK": 5, "algorithm": "RL (Bandit)"}'

# Get user intelligence
curl http://127.0.0.1:8000/api/user-intelligence/1

# Track engagement
curl -X POST http://127.0.0.1:8000/api/engagement-track \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 1,
    "recommendationId": 356,
    "movieTitle": "Forrest Gump",
    "clicked": true,
    "watched": false,
    "rating": null,
    "timestamp": "2026-03-22T10:30:00Z"
  }'
```

### **Interactive Docs**
Visit http://127.0.0.1:8000/docs (Swagger UI)
- Try endpoints interactively
- See request/response schemas
- Auto-generated from Pydantic models

---

## 📦 **Dependencies**

```
fastapi==0.109.0           # Web framework
uvicorn[standard]==0.27.0  # ASGI server
pydantic==2.5.3            # Data validation
numpy==1.24.3              # Numerical computing
pandas==2.0.3              # Data frames
scikit-learn==1.3.2        # RF model + preprocessing
torch==2.2.1               # LSTM model
joblib==1.3.2              # Model persistence
python-dotenv==1.0.0       # .env support
```

---

## 🐛 **Troubleshooting**

**ImportError: No module named 'sklearn'**
```bash
pip install scikit-learn
```

**ImportError: No module named 'torch'**
```bash
# CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU version (if CUDA available)
pip install torch
```

**Model files not found**
- Check paths in `app.py` line 20–32
- Ensure `/GraphModel`, `/Sequential Deep Learning Model`, etc. exist
- Verify `.pkl`, `.pt`, `.txt` files are present

**CORS errors from frontend**
- Update `.env` CORS_ORIGINS with frontend URL
- Ensure backend is running at `127.0.0.1:8000`
- Restart backend after .env changes

**Port already in use**
```bash
# Use different port
uvicorn app:app --host 127.0.0.1 --port 8001
```

---

## 🔒 **Security Considerations**

### **Current (Development)**
- CORS allows all origins: `allow_origins=["*"]`
- No authentication required
- No rate limiting

### **Production Checklist**
- [ ] Add API key authentication (FastAPI security)
- [ ] Restrict CORS origins to frontend domain
- [ ] Add rate limiting (slowapi library)
- [ ] Add request validation & sanitization
- [ ] Use HTTPS/SSL
- [ ] Add logging & monitoring
- [ ] Set up error tracking (Sentry)
- [ ] Add database for engagement events

---

## 📈 **Performance Tips**

1. **Caching**: Use `@lru_cache` for graph embeddings (already done)
2. **Async**: FastAPI handles concurrency automatically
3. **GPU**: Set `USE_GPU=true` in .env if CUDA available
4. **Batch**: Process multiple users' requests in batch

---

## 🚀 **Deployment**

### **Local**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### **Docker**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Cloud (AWS/Heroku/Azure)**
- Deploy Docker image to container service
- Set environment variables
- Ensure model files are included
- Monitor inference latency

---

## 📊 **Model Details**

### **Graph Embeddings**
- Node2Vec on movie similarity graph
- 64-dimensional vectors
- File: ~100MB

### **Random Forest**
- 1000 trees, max_depth=20
- Trained on MovieLens features
- File: ~50MB

### **LSTM**
- AttentionLSTM architecture
- 64-dim embeddings, 128-hidden
- Trained on 610 users, 100K+ sequences
- File: ~100MB

### **RL Agent**
- Thompson Sampling with Beta conjugate prior
- Learned over 100K+ user interactions
- File: ~10MB

---

## 🔄 **Request/Response Examples**

### **Success**
```json
{
  "userId": 1,
  "algorithm": "RL (Bandit)",
  "recommendations": [
    {
      "rank": 1,
      "movieId": 356,
      "title": "Forrest Gump (1994)",
      "genres": "Drama|Romance",
      "score": 0.923
    }
  ]
}
```

### **Error** (Invalid user)
```json
{
  "detail": "User 99999 has no watch history"
}
```

---

## 📚 **Code Structure**

```python
# app.py organization:

1. Imports & Path setup
2. Constants: model paths, directories
3. Pydantic models: Request/Response schemas
4. Helper functions: _safe_mean, _load_movies_map, _parse_graph_embeddings
5. create_app() factory function:
   - Load all data & models
   - Define endpoints
   - Return configured app
6. Recommendation functions: _recommend_baseline, _recommend_graph, _recommend_sequential, _recommend_rl
7. Route handlers: /health, /api/recommendations, /api/user-intelligence, /api/engagement-track
8. Instantiate: app = create_app()
```

---

## 🎓 **Learning Resources**

- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Uvicorn](https://www.uvicorn.org)
- [Pydantic](https://docs.pydantic.dev)
- [PyTorch](https://pytorch.org)
- [Scikit-learn](https://scikit-learn.org)

---

## 🤝 **Contributing**

To add a new recommendation algorithm:

1. Create `_recommend_mynewmodel(user_id, top_k)` function
2. Load model in `create_app()` factory
3. Add to algorithm map in `/api/recommendations`
4. Add Pydantic response model
5. Test with curl / Swagger UI

---

Ready to run? 🚀

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app:app --reload
```

Then visit http://127.0.0.1:8000/docs to test endpoints!
