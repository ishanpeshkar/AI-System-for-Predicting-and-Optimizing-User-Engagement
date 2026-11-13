import { useEffect, useMemo, useState } from 'react'
import './Recommendations.css'
import { useUserContext } from '../context/UserContext'
import { getUserHistory, postRecommendations } from '../lib/recommendations'

const algorithms = [
  { value: 'RL (Bandit)', label: 'RL (Bandit)' },
  { value: 'Baseline (RF)', label: 'Baseline (RF)' },
  { value: 'Graph (Node2Vec)', label: 'Graph (Node2Vec)' },
  { value: 'Sequential (LSTM)', label: 'Sequential (LSTM)' },
]

const scoreTypeLabels = {
  'RL (Bandit)': 'bandit reward score',
  'Baseline (RF)': 'engagement probability (0-1)',
  'Graph (Node2Vec)': 'cosine similarity (0-1)',
  'Sequential (LSTM)': 'next-watch probability',
}

function RecommendationCard({ rec, scoreLabel }) {
  return (
    <article className="predict-card">
      <div className="predict-card-top">
        <div>
          <div className="predict-rank">Rank #{rec.rank}</div>
          <h3 className="predict-title">{rec.title}</h3>
        </div>
        {typeof rec.score === 'number' ? (
          <div className="predict-score-col">
            <div className="predict-score-label">Score</div>
            <div className="predict-score-value">
              {rec.score.toFixed(3)}
            </div>
            <div className="predict-score-type">{scoreLabel}</div>
          </div>
        ) : null}
      </div>

      {rec.genres ? (
        <p className="predict-genres">{rec.genres}</p>
      ) : null}
    </article>
  )
}

export default function Recommendations() {
  const { setLastUserId } = useUserContext()
  const [userId, setUserId] = useState(1)
  const [topK, setTopK] = useState(10)
  const [algorithm, setAlgorithm] = useState('RL (Bandit)')

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [recommendations, setRecommendations] = useState([])

  const algorithmLabel = useMemo(() => {
    return algorithms.find((a) => a.value === algorithm)?.label ?? algorithm
  }, [algorithm])

  async function fetchRecommendations() {
    const normalizedUserId = Number(userId) || 1
    const normalizedTopK = Number(topK) || 10
    setLastUserId(normalizedUserId)
    setLoading(true)
    setError('')

    try {
      const data = await postRecommendations({
        userId: normalizedUserId,
        topK: normalizedTopK,
        algorithm,
      })
      setRecommendations(data)
    } catch (e) {
      setError(e?.message ?? 'Unknown error')
      setRecommendations([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRecommendations()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const watchHistory = getUserHistory(Number(userId) || 1)
  const scoreLabel = scoreTypeLabels[algorithmLabel] ?? 'model score'

  return (
    <div className="predict-page">
      <section className="predict-controls cp-page-card">
        <div className="predict-controls-grid">
          <label className="predict-field">
            <span>User ID</span>
            <input
              className="cp-input"
              type="number"
              min={1}
              value={userId}
              onChange={(e) => setUserId(e.target.value === '' ? '' : Number(e.target.value))}
            />
          </label>

          <label className="predict-field">
            <span>Algorithm</span>
            <select
              className="cp-select"
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
            >
              {algorithms.map((a) => (
                <option key={a.value} value={a.value}>
                  {a.label}
                </option>
              ))}
            </select>
          </label>

          <label className="predict-field">
            <span>Top K</span>
            <input
              className="cp-input"
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(e.target.value === '' ? '' : Number(e.target.value))}
            />
          </label>

          <div className="predict-actions">
            <button
              className="cp-button-primary"
              onClick={fetchRecommendations}
              disabled={loading}
            >
              {loading ? 'Loading...' : 'Get Recommendations'}
            </button>
          </div>
        </div>

        {error ? <div className="predict-error">{error}</div> : null}
      </section>

      <section className="cp-page-card" style={{ padding: 14 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>Watch history</h3>
        <div className="predict-watch-history">
          {watchHistory.map((movie) => (
            <span key={movie} className="predict-watch-chip">
              {movie}
            </span>
          ))}
        </div>
      </section>

      <section className="predict-results">
        <div className="predict-results-head">
          <div>
            <h2>Recommendations</h2>
            <p>Ranked results returned from the backend API.</p>
          </div>
          <div className="predict-api-meta">
            API: <span>POST /api/recommendations</span>
          </div>
        </div>

        {!loading && recommendations.length === 0 ? (
          <div className="predict-empty">
            No results yet. Enter parameters above and click "Get Recommendations".
          </div>
        ) : null}

        {loading ? (
          <div className="predict-empty">Running the selected model and generating a ranked list...</div>
        ) : null}

        {!loading ? (
          <div className="predict-grid">
            {recommendations.map((rec) => (
              <RecommendationCard
                key={`${rec.rank}-${rec.movieId}`}
                rec={rec}
                scoreLabel={scoreLabel}
              />
            ))}
          </div>
        ) : null}

        <div className="predict-api-call">
          <div className="predict-api-line">POST /api/recommendations</div>
          <pre>{`{
  "userId": ${Number(userId) || 1},
  "topK": ${Number(topK) || 10},
  "algorithm": "${algorithmLabel}"
}`}</pre>
        </div>
      </section>
    </div>
  )
}

