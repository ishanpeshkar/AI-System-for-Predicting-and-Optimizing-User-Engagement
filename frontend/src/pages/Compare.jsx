import { useMemo, useState } from 'react'
import { useUserContext } from '../context/UserContext'
import { MODEL_DEFINITIONS, formatScore, postRecommendations } from '../lib/recommendations'

const COLUMN_ORDER = ['Graph (Node2Vec)', 'Baseline (RF)', 'Sequential (LSTM)', 'RL (Bandit)']

const metricBadges = {
  graph: 'Embeddings 64-dim',
  baseline: 'AUC 0.74',
  lstm: 'HR@10 8.14%',
  rl: 'Regret -61%',
}

export default function Compare() {
  const { setLastUserId } = useUserContext()
  const [userId, setUserId] = useState(1)
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState({ graph: [], baseline: [], lstm: [], rl: [] })

  async function handleCompare() {
    const normalizedUserId = Number(userId) || 1
    const normalizedTopK = Number(topK) || 5
    setLastUserId(normalizedUserId)
    setLoading(true)
    setError('')

    try {
      const [graph, baseline, lstm, rl] = await Promise.all(
        COLUMN_ORDER.map((algorithm) =>
          postRecommendations({ userId: normalizedUserId, topK: normalizedTopK, algorithm }),
        ),
      )

      setResults({ graph, baseline, lstm, rl })
    } catch (err) {
      setError(err?.message ?? 'Unable to compare models')
      setResults({ graph: [], baseline: [], lstm: [], rl: [] })
    } finally {
      setLoading(false)
    }
  }

  const agreement = useMemo(() => {
    const counts = new Map()

    Object.values(results).forEach((rows) => {
      const unique = new Set(rows.map((row) => row.title))
      unique.forEach((title) => {
        counts.set(title, (counts.get(title) ?? 0) + 1)
      })
    })

    const rows = Array.from(counts.entries())
      .filter(([, count]) => count >= 2)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([title, count]) => ({ title, count }))

    return { counts, rows }
  }, [results])

  const topInsight = useMemo(() => {
    const rlTop = results.rl[0]?.title
    const lstmTop = results.lstm[0]?.title
    if (!rlTop || !lstmTop) {
      return 'Run compare to generate an insight from cross-model behavior.'
    }
    if (rlTop === lstmTop) {
      return 'RL agrees with LSTM on top recommendation -> strong sequential signal detected.'
    }
    return 'RL diverges from LSTM at rank 1 -> policy optimization is trading short-term sequence fit for long-term reward.'
  }, [results])

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <section className="cp-page-card" style={{ padding: 14, display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'end' }}>
          <label style={{ display: 'grid', gap: 6 }}>
            <span style={{ fontSize: 12, color: '#8b949e' }}>User ID</span>
            <input
              className="cp-input"
              type="number"
              value={userId}
              onChange={(event) => setUserId(Number(event.target.value))}
            />
          </label>
          <button className="cp-button-primary" onClick={handleCompare}>
            {loading ? 'Loading...' : 'Compare'}
          </button>
        </div>

        <label style={{ display: 'grid', gap: 6 }}>
          <span style={{ fontSize: 12, color: '#8b949e' }}>Top K</span>
          <select className="cp-select" value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
            {[3, 5, 8, 10].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
      </section>

      {error ? (
        <section className="cp-page-card" style={{ padding: 12, borderColor: '#a3716a', color: '#fdaeb7' }}>
          {error}
        </section>
      ) : null}

      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 10 }}>
        {MODEL_DEFINITIONS.map((model) => (
          <article key={model.key} className="cp-page-card" style={{ padding: 12 }}>
            <header style={{ borderBottom: '1px solid #30363d', paddingBottom: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span
                  style={{
                    fontSize: 11,
                    borderRadius: 8,
                    border: '1px solid #30363d',
                    padding: '2px 7px',
                    color: model.accent,
                    background: '#101722',
                  }}
                >
                  {model.badge}
                </span>
                <strong style={{ fontSize: 13 }}>{model.name}</strong>
              </div>
              <p style={{ margin: '6px 0 0', fontSize: 11, color: '#8b949e' }}>{model.scoreLabel}</p>
              <span
                style={{
                  marginTop: 6,
                  display: 'inline-block',
                  fontSize: 10,
                  color: '#6e7681',
                  border: '1px solid #30363d',
                  padding: '2px 6px',
                  borderRadius: 999,
                }}
              >
                {metricBadges[model.key]}
              </span>
            </header>

            <div style={{ marginTop: 8, display: 'grid' }}>
              {(loading ? [] : results[model.key]).map((item, index) => {
                const agreementCount = agreement.counts.get(item.title) ?? 0
                const highlighted = agreementCount >= 2

                return (
                  <div
                    key={`${model.key}-${item.movieId ?? item.title}-${index}`}
                    style={{
                      padding: '8px 8px',
                      borderBottom: '1px solid #222b36',
                      borderLeft: highlighted ? '2px solid #6366f1' : '2px solid transparent',
                      background: highlighted ? '#1a1a2e' : 'transparent',
                    }}
                  >
                    <div style={{ display: 'grid', gridTemplateColumns: '16px 1fr auto', gap: 8 }}>
                      <span style={{ fontSize: 11, color: '#6e7681' }}>{index + 1}</span>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 500 }}>{item.title}</div>
                        <div style={{ marginTop: 3, fontSize: 11, color: '#8b949e' }}>{item.genres}</div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ color: model.accent, fontSize: 12 }}>{formatScore(item.score)}</div>
                        <div style={{ marginTop: 2, fontSize: 10, color: '#6e7681' }}>{model.scoreLabel}</div>
                      </div>
                    </div>
                  </div>
                )
              })}

              {loading ? (
                <div className="cp-pulse" style={{ padding: 10, color: '#8b949e', fontSize: 12 }}>
                  Loading model outputs...
                </div>
              ) : null}
            </div>
          </article>
        ))}
      </section>

      <section className="cp-page-card" style={{ padding: 16 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>Model behavior insights</h3>
        <div style={{ marginTop: 10, background: '#111823', border: '1px solid #30363d', borderRadius: 10, padding: '10px 12px' }}>
          <p style={{ margin: 0, fontSize: 11, color: '#8b949e', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            Insight
          </p>
          <p style={{ margin: '6px 0 0', fontSize: 12, color: '#e6edf3' }}>{topInsight}</p>
        </div>
        <p style={{ margin: '10px 0 0', fontSize: 11, color: '#8b949e', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
          Understanding each algorithm
        </p>
        <div style={{ marginTop: 12, display: 'grid', gap: 0 }}>
          {[
            {
              badge: 'M1 Graph',
              color: '#79c0ff',
              bg: '#1c2333',
              text: 'Recommends based on structural similarity — users with overlapping watch graphs',
            },
            {
              badge: 'M2 RF',
              color: '#3fb950',
              bg: '#1c2820',
              text: 'Prefers high-engagement movies — relies on tabular features like genre match and popularity',
            },
            {
              badge: 'M3 LSTM',
              color: '#58a6ff',
              bg: '#102a4c',
              text: 'Captures recent behavioral trends — sensitive to the order and recency of watched movies',
            },
            {
              badge: 'M4 RL',
              color: '#d2a534',
              bg: '#3d2e0e',
              text: 'Balances exploration vs exploitation — learns which strategies produce long-term engagement',
            },
          ].map((row, idx) => (
            <div
              key={row.badge}
              style={{
                display: 'flex',
                gap: 10,
                padding: '8px 0',
                borderBottom: idx < 3 ? '1px solid #21262d' : 'none',
              }}
            >
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  borderRadius: 8,
                  border: '1px solid #30363d',
                  padding: '3px 7px',
                  background: row.bg,
                  color: row.color,
                  whiteSpace: 'nowrap',
                }}
              >
                {row.badge}
              </span>
              <span style={{ fontSize: 12, color: '#8b949e', lineHeight: 1.5 }}>{row.text}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="cp-page-card" style={{ padding: 14 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>Model agreement</h3>
        <div style={{ marginTop: 10, display: 'grid', gap: 8 }}>
          {agreement.rows.map((item, index) => (
            <div
              key={item.title}
              style={{
                display: 'grid',
                gridTemplateColumns: '1fr auto 160px',
                gap: 12,
                alignItems: 'center',
              }}
            >
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span>{item.title}</span>
                {index === 0 ? (
                  <span
                    style={{
                      fontSize: 10,
                      background: '#1c2d1c',
                      border: '1px solid #3fb950',
                      color: '#3fb950',
                      borderRadius: 999,
                      padding: '2px 6px',
                    }}
                  >
                    consensus
                  </span>
                ) : null}
              </div>
              <span style={{ fontSize: 12, color: '#8b949e' }}>{item.count} / 4 models</span>
              <div style={{ height: 8, borderRadius: 999, background: '#111823', border: '1px solid #30363d', overflow: 'hidden' }}>
                <div style={{ width: `${(item.count / 4) * 100}%`, height: '100%', background: '#6366f1' }} />
              </div>
            </div>
          ))}

          {!loading && agreement.rows.length === 0 ? (
            <p style={{ margin: 0, fontSize: 12, color: '#8b949e' }}>Run compare to see consensus between models.</p>
          ) : null}
        </div>
      </section>
    </div>
  )
}
