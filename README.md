# CinePulse — AI-Powered Movie Recommendation System

**A complete human engagement and interaction system for movie recommendations using 4 AI models**

This project combines four distinct machine learning approaches (Graph Embeddings, Random Forest Baseline, Sequential LSTM, and Reinforcement Learning) into a single, transparent recommendation interface. The system explains *why* each recommendation is made, providing users with complete visibility into AI decision-making.

---

## 🎬 **Quick Start**

### **Prerequisites**
- Node.js 18+ (for frontend)
- Python 3.9+ (for backend)
- npm or yarn (for frontend package management)

### **Setup — Frontend Only (5 minutes)**

```bash
# 1. Navigate to frontend directory
cd Capstone_Final/Fronted

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev

# Frontend will open at http://localhost:5173
```

### **Setup — Full Stack (10 minutes)**

```bash
# 1. Setup Python virtual environment (backend)
cd ../backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. In another terminal, setup frontend
cd ../Fronted
npm install
npm run dev

# 4. Start backend (from backend directory with venv activated)
cd ../backend
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# Backend available at http://127.0.0.1:8000
# Frontend available at http://localhost:5173
```

---

## 📁 **Project Structure**

```
Capstone_Final/
├── Fronted/                                    # React frontend (Vite)
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx                     # Hero landing page
│   │   │   ├── Dashboard.jsx                   # 4-model comparison hub (main page)
│   │   │   ├── Compare.jsx                     # Side-by-side model analysis
│   │   │   ├── Metrics.jsx                     # Performance charts & tables
│   │   │   ├── Recommendations.jsx             # Predict/detail page
│   │   │   └── About.jsx                       # About CinePulse
│   │   ├── layouts/
│   │   │   └── AppLayout.jsx                   # Persistent sidebar + top bar
│   │   ├── context/
│   │   │   └── UserContext.jsx                 # User state + system status
│   │   ├── lib/
│   │   │   └── recommendations.js              # API helpers & data
│   │   ├── styles/
│   │   │   └── cinepulse.css                   # Dark theme styles
│   │   ├── App.jsx                             # Route definitions
│   │   └── main.jsx                            # Entry point + providers
│   ├── package.json
│   └── vite.config.js
│
├── backend/                                    # FastAPI server
│   ├── app.py                                  # Main API routes
│   ├── config.py                               # Configuration management
│   ├── requirements.txt                        # Python dependencies
│   ├── .env.example                            # Example environment file
│   └── __init__.py
│
├── Feature Engineering + Baseline Model/       # Random Forest model
│   ├── rf_model.pkl                            # Trained RF classifier
│   ├── ml-latest-small/                        # MovieLens dataset
│   └── train_model.py
│
├── GraphModel/                                 # Node2Vec graph embeddings
│   ├── movie_embeddings.txt                    # 64-dim movie embeddings
│   ├── GraphModel.ipynb                        # Graph training notebook
│   └── movies.csv
│
├── Sequential Deep Learning Model/             # LSTM sequence model
│   ├── models/
│   │   └── lstm_model.pt                       # Trained LSTM weights
│   ├── src/
│   │   ├── model.py                            # AttentionLSTM architecture
│   │   └── extract_states.py
│   ├── data/
│   │   ├── movie2idx.json                      # Movie ID → index mapping
│   │   └── X.npy, y.npy                        # Training data
│   └── INTEGRATION_GUIDE_FOR_MEMBER_4.txt
│
├── Reinforcement_Learning_Recommender/        # Multi-armed bandit RL
│   ├── models/
│   │   └── mab_agent_thompson.pkl              # Trained RL agent
│   ├── main.py
│   ├── mab_recommender.py
│   ├── prediction_eval.py
│   ├── train_mab.py
│   └── requirements.txt
│
└── README.md                                   # This file
```

---

## 🚀 **Features**

### **Frontend — Human Engagement Layer**

- **Dashboard**: 2×2 grid showing all 4 models running on same user simultaneously
- **User Intelligence Panel**: AI-generated insights into user behavior, genre preferences, engagement level
- **Why This Recommendation?**: Transparent reasoning showing which models voted for each movie
- **Pipeline Visualization**: Visual flow showing data → LSTM → Graph → RL → Recommendation
- **Confidence Bars**: Each model shows confidence % for top result
- **Predict/Detail Page**: Single algorithm deep-dive with watch history and score explanations
- **Compare Page**: Side-by-side model outputs with agreement highlighting
- **Metrics**: Performance charts (training loss, RL rewards) + accuracy tables
- **System Status Indicator**: Real-time feedback (ready → running → active → error)

### **Backend — 4 Recommendation Models**

1. **M1 - Graph (Node2Vec)**: Structural similarity in movie embedding space
2. **M2 - Baseline (RF)**: Engagement probability via Random Forest tabular features
3. **M3 - Sequential (LSTM)**: Temporal watch intent from recent movie sequence
4. **M4 - RL (Bandit)**: Thompson Sampling balancing exploration vs. exploitation

---

## 🔧 **API Endpoints**

### **Core Recommendation**
```
POST /api/recommendations
Request: {
  "userId": 1,
  "topK": 5,
  "algorithm": "RL (Bandit)"  | "Baseline (RF)" | "Graph (Node2Vec)" | "Sequential (LSTM)"
}
Response: {
  "userId": 1,
  "algorithm": "RL (Bandit)",
  "recommendations": [
    {
      "rank": 1,
      "movieId": 356,
      "title": "Forrest Gump (1994)",
      "genres": "Drama|Romance",
      "score": 0.92
    },
    ...
  ]
}
```

### **User Intelligence**
```
GET /api/user-intelligence/{user_id}
Response: {
  "userId": 1,
  "dominantGenre": "Sci-Fi",
  "recentTrend": "Space / Futuristic",
  "engagementLevel": "High",
  "sequenceInsight": "User is transitioning from Action → Sci-Fi. Recent 5 movies show strong space theme.",
  "confidenceScore": 0.82
}
```

### **Engagement Tracking**
```
POST /api/engagement-track
Request: {
  "userId": 1,
  "recommendationId": 356,
  "movieTitle": "Forrest Gump (1994)",
  "clicked": true,
  "watched": false,
  "rating": null,
  "timestamp": "2026-03-22T10:30:00Z"
}
Response: {
  "status": "tracked",
  "message": "Engagement recorded for user 1"
}
```

### **Health Check**
```
GET /health
Response: { "status": "ok" }
```

---

## 🎯 **UI Tour**

### **Landing Page**
- Hero with updated copy: "Four AI models. One user. See how each one thinks."
- 4 metric cards: HR@10, AUC, Regret, Embeddings
- Feature highlights
- Team credentials

### **Dashboard (Main Hub)**
- **Left sidebar**: User selector + "Run Demo" button + recent watch history
- **User Intelligence Panel**: Personalized genre/trend/engagement analysis
- **Pipeline Strip**: Visual flow of recommendation pipeline
- **2×2 Model Grid**: 
  - M1 Graph (blue), M2 RF (green), M3 LSTM (bright blue), M4 RL (amber)
  - Each shows top 3 results + confidence bars
- **Final Recommendation Banner**:
  - Movie title (from M4 RL)
  - M2 score, M3 rank, model agreement count
  - **"Why this recommendation?"** section with 4 checkmarks explaining each model's vote
- **System status** in top bar (pulsing amber while loading, green when done)

### **Compare Page**
- User ID + Compare button + topK selector
- 4 columns (Graph | Baseline | LSTM | RL)
- movies appearing in 2+ models highlighted with left border
- **Model Behavior Insights** card explaining each algorithm's approach
- **Model Agreement** summary showing consensus movies

### **Metrics Page**
- 4 stat cards (HR@10, NDCG, AUC, Regret)
- LSTM training loss chart (15 epochs)
- RL cumulative reward curve (5000 episodes)
- Baseline vs. full pipeline comparison table

### **Predict/Detail Page**
- Single model deep-dive
- Watch history section
- Full results grid (2 columns, all topK results)
- Score type labels under every score
- API call inspector showing exact request/response

---

## 📊 **Key Metrics & Configuration**

### **Model Performance**
- **HR@10** (Hit Rate): 8.14% (M3 LSTM)
- **AUC**: 0.84 (full pipeline) vs 0.74 (M2 baseline alone)
- **Regret Reduction**: 61% (M4 RL Thompson Sampling)
- **Embeddings**: 64-dimensional Node2Vec

### **User Intelligence Scoring**
| User ID | Dominant Genre | Engagement | Confidence |
|---------|----------------|------------|-----------|
| 1       | Sci-Fi         | High       | 82%       |
| 5       | Classic Drama  | High       | 76%       |
| Other   | Mixed          | Variable   | 71%       |

### **Model Confidence Scores**
- M1 Graph: 91%
- M2 RF: 81%
- M3 LSTM: 84%
- M4 RL: 88%

---

## 🔐 **Environment Setup**

Create `.env` file in `/backend` directory:

```env
# Backend Server
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
BACKEND_RELOAD=true

# CORS (for frontend)
CORS_ORIGINS=["http://localhost:5173", "http://127.0.0.1:5173"]

# Logging
LOG_LEVEL=INFO

# Model Configuration
USE_GPU=false
MODEL_CACHE_SIZE=1000
```

---

## 🧪 **Testing the System**

### **Quick Demo Workflow**
1. Open http://localhost:5173
2. On Landing page, click "Open Dashboard"
3. Click "Run Demo — User 1" button
4. Watch system status turn amber (loading) → green (ready)
5. See all 4 models load with results
6. Read "User behavior intelligence" panel explaining this user
7. See "Why this recommendation?" section in final banner
8. Click "Compare" in sidebar to see all 4 models side-by-side
9. View model agreement highlights

### **Manual API Testing**
```bash
# Using curl
curl -X POST http://127.0.0.1:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "userId": 1,
    "topK": 5,
    "algorithm": "RL (Bandit)"
  }'

# Get user intelligence
curl http://127.0.0.1:8000/api/user-intelligence/1
```

---

## 🛠️ **Troubleshooting**

### **Frontend won't start**
```bash
# Clear node_modules cache
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### **Backend won't start**
```bash
# Check Python version (need 3.9+)
python --version

# Verify virtual environment activation
# Windows: venv\Scripts\activate should show (venv) in terminal
# macOS/Linux: source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### **CORS errors**
- Ensure backend is running on `127.0.0.1:8000`
- Check `.env` file has correct frontend URL in CORS_ORIGINS
- Restart backend after changing `.env`

### **Models not loading**
- Verify all model files exist in their directories:
  - `Feature Engineering + Baseline Model/rf_model.pkl`
  - `GraphModel/movie_embeddings.txt`
  - `Sequential Deep Learning Model/models/lstm_model.pt`
  - `Reinforcement_Learning_Recommender/models/mab_agent_thompson.pkl`

---

## 📚 **Technical Stack**

### **Frontend**
- React 19 + React Router
- Vite (build tool)
- Tailwind CSS + custom dark theme
- Chart.js (metrics visualizations)
- No UI framework — pure custom components

### **Backend**
- FastAPI (async REST API)
- Uvicorn (ASGI server)
- Pydantic (data validation)
- NumPy, Pandas (data processing)
- Scikit-learn (RF baseline)
- PyTorch (LSTM model)
- Joblib (model serialization)

### **ML Models**
- **Graph**: Node2Vec embeddings (64-dim)
- **Baseline**: Random Forest classifier (scikit-learn)
- **Sequential**: Attention LSTM (PyTorch, 128 hidden units)
- **RL**: Thompson Sampling multi-armed bandit

### **Data**
- MovieLens "ml-latest-small" (9,724 movies, 100K ratings)
- Per-model data splits for evaluation rigor

---

## 👥 **Team**

Developed by: **Harsh Kumar, Vikesh Mehta, Ishan, Ayush, Om Kabade**

**VIT Bhopal · 2024**

---

## 📖 **Documentation**

- **Frontend Code**: See inline JSDoc comments in React components
- **Backend Code**: See docstrings in `app.py` endpoints
- **Models**: Individual README files in each model folder
- **API Spec**: Full OpenAPI/Swagger docs at `http://127.0.0.1:8000/docs`

---

## 🚀 **Next Steps**

- [ ] Deploy frontend to Vercel/Netlify
- [ ] Deploy backend to AWS/Azure
- [ ] Add user authentication (OAuth2)
- [ ] Implement engagement database persistence
- [ ] Add A/B testing framework
- [ ] Create admin dashboard for model monitoring
- [ ] Expand to real-time model retraining
- [ ] Integrate live user feedback loop

---

## 📄 **License**

Academic Research Project — VIT Bhopal (2024)

---

## 💬 **Support**

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review API docs at `http://127.0.0.1:8000/docs` (when backend is running)
3. Check browser console (F12) for frontend errors
4. Check terminal logs for backend errors

---

**Ready to explore AI-powered recommendations?**

```bash
npm run dev    # Start frontend
uvicorn app:app --reload  # Start backend (in another terminal)
```

Open http://localhost:5173 and click "Open Dashboard" 🎬
