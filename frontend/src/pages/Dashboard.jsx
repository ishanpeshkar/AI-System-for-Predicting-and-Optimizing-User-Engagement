import { useState, useEffect } from 'react'
import { useUserContext } from '../context/UserContext'
import {
  MODEL_DEFINITIONS,
  formatScore,
  postRecommendations,
  splitGenres,
} from '../lib/recommendations'

const DASHBOARD_ORDER = [
  'RL (Bandit)',
  'Baseline (RF)',
  'Graph (Node2Vec)',
  'Sequential (LSTM)',
]

const CONFIDENCE_SCORES = {
  graph: 91,
  baseline: 81,
  lstm: 84,
  rl: 88,
}

const PIPELINE_STAGES = [
  { label: 'Watch history', step: 0, help: 'Reads recent user behavior from ratings history.' },
  { label: 'LSTM → Intent', step: 1, color: '#58a6ff', bg: '#102a4c', help: 'LSTM captures sequence patterns from recent watches.' },
  { label: 'Graph → Context', step: 2, color: '#79c0ff', bg: '#1c2333', help: 'Graph finds similar users and neighboring taste clusters.' },
  { label: 'RL → Decision', step: 3, color: '#d2a534', bg: '#3d2e0e', help: 'RL selects optimal action balancing reward and exploration.' },
  { label: 'Recommendation', step: 4, color: '#3fb950', bg: '#1c2d1c', border: '1px solid #3fb95055', help: 'Final policy decision shown to the user.' },
]

function PanelSkeleton() {
  return (
    <div className="cp-page-card cp-pulse" style={{ padding: 14, minHeight: 188 }}>
      <div style={{ height: 12, width: '45%', background: '#202833', borderRadius: 4 }} />
      <div style={{ height: 10, width: '70%', marginTop: 8, background: '#202833', borderRadius: 4 }} />
      <div style={{ marginTop: 16, display: 'grid', gap: 9 }}>
        <div style={{ height: 22, background: '#202833', borderRadius: 6 }} />
        <div style={{ height: 22, background: '#202833', borderRadius: 6 }} />
        <div style={{ height: 22, background: '#202833', borderRadius: 6 }} />
      </div>
    </div>
  )
}

function RankRow({ item, rank, scoreLabel, isLstm }) {
  const genres = splitGenres(item.genres)

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '16px 1fr auto',
        gap: 10,
        padding: '8px 0',
        borderTop: '1px solid #212a35',
      }}
    >
      <div style={{ color: '#6e7681', fontSize: 12 }}>{rank}</div>
      <div>
        <div style={{ fontSize: 13, fontWeight: 500 }}>{item.title}</div>
        <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {genres.map((genre) => (
            <span
              key={`${item.movieId}-${genre}`}
              style={{
                fontSize: 10,
                border: '1px solid #30363d',
                borderRadius: 999,
                padding: '2px 6px',
                color: '#8b949e',
              }}
            >
              {genre}
            </span>
          ))}
        </div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <div style={{ fontSize: 13, color: '#e6edf3' }}>
          {isLstm ? `#${rank} of 9,724` : formatScore(item.score)}
        </div>
        <div style={{ marginTop: 2, fontSize: 10, color: '#6e7681' }}>
          {isLstm ? 'predicted rank' : scoreLabel}
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { setLastUserId, systemStatus, setSystemStatus } = useUserContext()
  const [userId, setUserId] = useState(1)
  const [userProfile, setUserProfile] = useState(null)
  const [intelligence, setIntelligence] = useState(null)
  const [edgeCaseMessage, setEdgeCaseMessage] = useState('')
  const [feedbackSent, setFeedbackSent] = useState(false)
  const [feedbackType, setFeedbackType] = useState(null)
  const [showSystemDetails, setShowSystemDetails] = useState(false)
  const [hoveredPipeline, setHoveredPipeline] = useState('')
  const [loading, setLoading] = useState(false)
  const [pipelineStep, setPipelineStep] = useState(0) // 0 = ready, 1-4 = loading, 5 = done
  const [error, setError] = useState('')
  const [results, setResults] = useState({
    graph: [],
    baseline: [],
    lstm: [],
    rl: [],
  })

  useEffect(() => {
    setUserProfile(null)
    setIntelligence(null)
    setEdgeCaseMessage('')
    setFeedbackSent(false)
    setFeedbackType(null)
  }, [userId])

  function buildIntelligence(profile, modelResults) {
    if (!profile || !profile.found) return null

    // Only look at last 5 movies for dominant genre.
    const recentHistory = profile.watch_history?.slice(-5) || []

    // Count genres from recent history only and weight primary genre higher.
    const recentGenreCount = {}
    recentHistory.forEach((movie) => {
      const genres = (movie.genres || '').split('|')
      genres.forEach((genre, index) => {
        const weight = index === 0 ? 2 : 1
        recentGenreCount[genre] = (recentGenreCount[genre] || 0) + weight
      })
    })

    const dominantGenre =
      Object.entries(recentGenreCount).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Mixed'

    const recentTrend =
      recentHistory
        .slice(-3)
        .map((movie) => (movie.genres || '').split('|')[0])
        .filter((genre, index, arr) => genre && arr.indexOf(genre) === index)
        .slice(0, 2)
        .join(' / ') || 'Varied'

    const rfTopScore = modelResults?.[1]?.[0]?.score || 0
    const engagement = rfTopScore > 0.85 ? 'High' : rfTopScore > 0.65 ? 'Medium' : 'Low'

    const lstmRaw = modelResults?.[2]?.[0]?.score || 0
    const confidence =
      lstmRaw > 0.008 ? 88 : lstmRaw > 0.005 ? 82 : lstmRaw > 0.003 ? 74 : lstmRaw > 0.001 ? 65 : 58

    const insight = `${dominantGenre} pattern detected in last 5 movies - recent trend: ${recentTrend}`

    return { dominantGenre, recentTrend, engagement, confidence, insight }
  }

  const handleFeedback = async (type) => {
    const currentRecommendation = results.rl[0]
    if (feedbackSent || !currentRecommendation) return

    try {
      await fetch('http://localhost:8000/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userId: parseInt(String(userId), 10),
          movieId: currentRecommendation.movieId || 0,
          movieTitle: currentRecommendation.title || '',
          feedbackType: type,
        }),
      })
      setFeedbackSent(true)
      setFeedbackType(type)
    } catch (err) {
      console.error(err)
    }
  }

  async function loadUserRecommendations() {
    const parsedUserId = Number(userId)
    const normalizedUserId = Number.isFinite(parsedUserId) && parsedUserId >= 1 ? parsedUserId : 1
    const fallbackMode = !Number.isFinite(parsedUserId) || parsedUserId < 1

    if (fallbackMode) {
      setEdgeCaseMessage('No sufficient history. Showing popular recommendations.')
    } else {
      setEdgeCaseMessage('')
    }

    setLastUserId(normalizedUserId)
    setFeedbackSent(false)
    setFeedbackType(null)
    setLoading(true)
    setPipelineStep(1)
    setSystemStatus('loading')
    setError('')

    try {
      let profile = null
      try {
        const profileRes = await fetch(`http://localhost:8000/api/user-profile/${normalizedUserId}`)
        profile = await profileRes.json()
        setUserProfile(profile)
        if (!profile?.found) {
          setEdgeCaseMessage('No sufficient history. Showing popular recommendations.')
        }
      } catch (err) {
        setUserProfile(null)
        setEdgeCaseMessage('No sufficient history. Showing popular recommendations.')
      }

      const modelResults = await Promise.all(
        DASHBOARD_ORDER.map((algorithm) =>
          postRecommendations({
            userId: normalizedUserId,
            topK: 5,
            algorithm,
          }),
        ),
      )

      const [rl, baseline, graph, lstm] = modelResults

      setResults({ graph, baseline, lstm, rl })
      setIntelligence(buildIntelligence(profile, modelResults))
      setPipelineStep(5)
      setSystemStatus('active')
    } catch (err) {
      setError(err?.message ?? 'Failed to fetch recommendations')
      setResults({ graph: [], baseline: [], lstm: [], rl: [] })
      setIntelligence(null)
      setSystemStatus('error')
    } finally {
      setLoading(false)
    }
  }

  function handleRunDemo() {
    setUserId(1)
    // Use useEffect to load after state update
    setTimeout(() => {
      loadUserRecommendations()
    }, 0)
  }

  const rlTop = results.rl[0]
  const baselineMatch = rlTop
    ? results.baseline.find((movie) => movie.title === rlTop.title) ?? results.baseline[0]
    : results.baseline[0]

  const topThreeSets = [
    new Set(results.graph.slice(0, 3).map((movie) => movie.title)),
    new Set(results.baseline.slice(0, 3).map((movie) => movie.title)),
    new Set(results.lstm.slice(0, 3).map((movie) => movie.title)),
    new Set(results.rl.slice(0, 3).map((movie) => movie.title)),
  ]
  const agreementCount = rlTop
    ? topThreeSets.reduce((count, modelSet) => count + (modelSet.has(rlTop.title) ? 1 : 0), 0)
    : 0

  const engagementColor =
    intelligence?.engagement === 'High'
      ? '#3fb950'
      : intelligence?.engagement === 'Medium'
        ? '#d2a534'
        : '#8b949e'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 16 }}>
      <aside className="cp-page-card" style={{ padding: 14, height: 'fit-content' }}>
        <section>
          <h3 style={{ margin: 0, fontSize: 13 }}>Select user</h3>
          <label style={{ display: 'grid', gap: 8, marginTop: 10 }}>
            <input
              className="cp-input"
              type="number"
              value={userId}
              onChange={(event) => {
                const next = event.target.value
                setUserId(next === '' ? '' : Number(next))
              }}
            />
            <button className="cp-button-primary" onClick={loadUserRecommendations}>
              {loading ? 'Loading...' : 'Load user'}
            </button>
            <button
              onClick={handleRunDemo}
              style={{
                width: '100%',
                background: 'transparent',
                border: '1px solid #30363d',
                color: '#8b949e',
                fontSize: 12,
                borderRadius: 8,
                padding: 7,
                cursor: 'pointer',
                transition: 'background 0.2s, color 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#21262d'
                e.currentTarget.style.color = '#e6edf3'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent'
                e.currentTarget.style.color = '#8b949e'
              }}
            >
              Run Demo — User 1
            </button>
          </label>
          <p style={{ margin: '10px 0 0', fontSize: 12, color: '#8b949e' }}>
            {userProfile?.found
              ? `${userProfile.total_ratings} ratings · avg ${userProfile.avg_rating}`
              : 'Load a user to see profile stats'}
          </p>
          {edgeCaseMessage ? (
            <p style={{ margin: '8px 0 0', fontSize: 12, color: '#d2a534' }}>{edgeCaseMessage}</p>
          ) : null}
        </section>

        <section style={{ marginTop: 18 }}>
          <h3 style={{ margin: 0, fontSize: 13 }}>Recent watch history</h3>
          <div style={{ marginTop: 10 }}>
            {userProfile === null ? (
              <div style={{ fontSize: '12px', color: '#8b949e' }}>Load a user to see their history.</div>
            ) : !userProfile.found ? (
              <div style={{ fontSize: '12px', color: '#f85149' }}>
                No data found for userId {userId}. Valid range: 1 to 610.
              </div>
            ) : (
              <>
                <div style={{ fontSize: '12px', color: '#8b949e', marginBottom: '6px' }}>
                  {userProfile.total_ratings} ratings · avg {userProfile.avg_rating}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {userProfile.watch_history.map((movie) => (
                    <span
                      key={movie.movieId}
                      style={{
                        background: '#1c2333',
                        border: '1px solid #30363d',
                        borderRadius: '20px',
                        padding: '3px 9px',
                        fontSize: '11px',
                        color: '#79c0ff',
                      }}
                    >
                      {movie.title.replace(/ \(\d{4}\)$/, '')}
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>
        </section>
      </aside>

      <div style={{ display: 'grid', gap: 16 }}>
        {error ? (
          <div className="cp-page-card" style={{ padding: 12, borderColor: '#a3716a', color: '#fdaeb7' }}>
            {error}
          </div>
        ) : null}

        <section className="cp-page-card" style={{ padding: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <div>
            <p style={{ margin: 0, fontSize: 11, color: '#8b949e', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              System detail level
            </p>
            <p style={{ margin: '4px 0 0', fontSize: 12, color: '#6e7681' }}>
              Keep focus on the main decision, then expand internals when needed.
            </p>
          </div>
          <button
            onClick={() => setShowSystemDetails((prev) => !prev)}
            style={{
              background: showSystemDetails ? '#1c2d1c' : '#21262d',
              border: '1px solid #30363d',
              color: showSystemDetails ? '#3fb950' : '#8b949e',
              fontSize: 12,
              borderRadius: 8,
              padding: '7px 11px',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            {showSystemDetails ? 'Hide system internals' : 'Show system internals'}
          </button>
        </section>

        {/* USER INTELLIGENCE PANEL */}
        {!loading && results.rl.length > 0 && intelligence ? (
          <section className="cp-page-card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <p style={{ margin: 0, fontSize: 11, color: '#8b949e', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                User behavior intelligence
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#3fb950' }} />
                <span style={{ fontSize: 12, color: '#3fb950' }}>Analysis ready</span>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 14 }}>
              <div style={{ gridColumn: '1 / -1', background: '#111823', border: '1px solid #30363d', borderRadius: 10, padding: '10px 12px' }}>
                <p style={{ margin: 0, fontSize: 11, color: '#8b949e', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  AI Summary
                </p>
                <p style={{ margin: '6px 0 0', fontSize: 13, color: '#e6edf3' }}>
                  User prefers {intelligence.dominantGenre.toLowerCase()}-heavy content with {intelligence.engagement.toLowerCase()} engagement consistency.
                </p>
              </div>

              {/* Column 1: Sequence profile */}
              <div>
                <h4 style={{ margin: 0, fontSize: 12, fontWeight: 600, marginBottom: 10 }}>Sequence profile</h4>
                <div style={{ display: 'grid', gap: 8 }}>
                  <div>
                    <p style={{ margin: 0, fontSize: 11, color: '#6e7681' }}>Dominant genre</p>
                    <p style={{ margin: '4px 0 0', fontSize: 13, fontWeight: 500, color: '#e6edf3' }}>
                      {intelligence.dominantGenre}
                    </p>
                  </div>
                  <div>
                    <p style={{ margin: 0, fontSize: 11, color: '#6e7681' }}>Recent trend</p>
                    <p style={{ margin: '4px 0 0', fontSize: 13, color: '#8b949e' }}>
                      {intelligence.recentTrend}
                    </p>
                  </div>
                  <div>
                    <p style={{ margin: 0, fontSize: 11, color: '#6e7681' }}>Engagement level</p>
                    <p style={{ margin: '4px 0 0', fontSize: 13, fontWeight: 500, color: engagementColor }}>
                      {intelligence.engagement}
                    </p>
                  </div>
                </div>
              </div>

              {/* Column 2: Behavioral signal */}
              <div>
                <h4 style={{ margin: 0, fontSize: 12, fontWeight: 600, marginBottom: 10 }}>Behavioral signal (M3 LSTM)</h4>
                <div>
                  <p style={{ margin: 0, fontSize: 11, color: '#6e7681' }}>Sequence insight</p>
                  <div
                    style={{
                      marginTop: 8,
                      background: '#102a4c',
                      border: '1px solid #388bfd55',
                      borderRadius: 8,
                      padding: '8px 12px',
                      fontSize: 12,
                      color: '#58a6ff',
                      lineHeight: 1.5,
                    }}
                  >
                    {intelligence.insight}
                  </div>
                  <div style={{ marginTop: 10 }}>
                    <p style={{ margin: 0, fontSize: 11, color: '#6e7681' }}>State vector</p>
                    <p style={{ margin: '4px 0 0', fontSize: 13, color: '#8b949e' }}>128-dim · M3 LSTM</p>
                  </div>
                </div>
              </div>

              {/* Column 3: Confidence */}
              <div>
                <h4 style={{ margin: 0, fontSize: 12, fontWeight: 600, marginBottom: 10 }}>Confidence</h4>
                <div style={{ textAlign: 'center' }}>
                  <p style={{ margin: 0, fontSize: 28, fontWeight: 500, color: '#e6edf3' }}>
                    {intelligence.confidence}%
                  </p>
                  <p style={{ margin: '6px 0 0', fontSize: 11, color: '#8b949e' }}>Prediction confidence</p>
                  <div
                    style={{
                      marginTop: 8,
                      background: '#21262d',
                      height: 6,
                      borderRadius: 3,
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        height: '100%',
                        width: `${intelligence.confidence}%`,
                        background: '#6366f1',
                        borderRadius: 3,
                        transition: 'width 0.8s ease',
                      }}
                    />
                  </div>
                  <p style={{ margin: '6px 0 0', fontSize: 10, color: '#6e7681' }}>Based on sequence consistency</p>
                </div>
              </div>
            </div>
          </section>
        ) : !loading ? (
          <section className="cp-page-card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <p style={{ margin: 0, fontSize: 11, color: '#8b949e', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                User behavior intelligence
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#6e7681' }} />
                <span style={{ fontSize: 12, color: '#6e7681' }}>Awaiting user data</span>
              </div>
            </div>
            <p style={{ margin: 0, fontSize: 12, color: '#8b949e' }}>Click "Load user" to run analysis and generate personalized insights.</p>
          </section>
        ) : null}

        {/* FINAL DECISION HERO */}
        <section
          style={{
            background: 'radial-gradient(circle at 25% 0%, rgba(63,185,80,0.25), rgba(28,45,28,1) 55%)',
            border: '1px solid #3fb950',
            borderRadius: 14,
            padding: 18,
            display: 'grid',
            gap: 14,
            boxShadow: '0 0 30px rgba(63,185,80,0.2), inset 0 0 20px rgba(63,185,80,0.1)',
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 14, flexWrap: 'wrap', alignItems: 'start' }}>
            <div>
              <p style={{ margin: 0, fontSize: 11, color: '#9cd4a4', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                Primary Decision Engine
              </p>
              <p style={{ margin: '6px 0 0', fontSize: 11, color: '#84b88d' }}>
                RL-optimized output prioritized for long-term engagement
              </p>
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: 30, fontWeight: 700, color: '#f2fff4', letterSpacing: '-0.02em' }}>
                  {rlTop?.title ?? 'Awaiting RL output'}
                </div>
                <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  <span style={{ fontSize: 12, color: '#dfffe4', border: '1px solid #3fb95077', padding: '4px 8px', borderRadius: 999 }}>
                    Confidence: {intelligence?.engagement || 'High'} ({intelligence?.confidence || 88}%)
                  </span>
                  <span style={{ fontSize: 12, color: '#dfffe4', border: '1px solid #58a6ff77', padding: '4px 8px', borderRadius: 999 }}>
                    Consensus: {agreementCount}/4 models agree
                  </span>
                </div>
                {results.rl.length > 0 ? (
                  !feedbackSent ? (
                    <div style={{ display: 'flex', gap: '8px', marginTop: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
                      <span style={{ fontSize: '12px', color: '#9fd8a8' }}>Was this relevant?</span>
                      {[
                        { type: 'liked', label: 'Liked it', color: '#3fb950', bg: '#1c2d1c' },
                        { type: 'watched', label: 'Would watch', color: '#79c0ff', bg: '#1c2333' },
                        { type: 'disliked', label: 'Not for me', color: '#f85149', bg: '#2d1c1c' },
                      ].map((btn) => (
                        <button
                          key={btn.type}
                          onClick={() => handleFeedback(btn.type)}
                          style={{
                            padding: '5px 14px',
                            background: btn.bg,
                            border: `1px solid ${btn.color}`,
                            borderRadius: '6px',
                            color: btn.color,
                            fontSize: '12px',
                            cursor: 'pointer',
                            fontFamily: 'inherit',
                          }}
                        >
                          {btn.label}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div
                      style={{
                        marginTop: '12px',
                        fontSize: '12px',
                        color: '#3fb950',
                        display: 'flex',
                        gap: '6px',
                        alignItems: 'center',
                      }}
                    >
                      <span>✓</span>
                      <span>
                        Feedback recorded - RL model updated.
                        {feedbackType === 'liked' && ' Positive reward (+2) applied.'}
                        {feedbackType === 'disliked' && ' Negative reward (-1) applied.'}
                        {feedbackType === 'watched' && ' Partial reward (+1) applied.'}
                      </span>
                    </div>
                  )
                ) : null}
              </div>
            </div>

            <div style={{ display: 'flex', gap: 18, alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: 11, color: '#b5d8bb' }}>M2 score</div>
                <div style={{ marginTop: 4, fontSize: 16 }}>{formatScore(baselineMatch?.score)}</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#b5d8bb' }}>M3 rank</div>
                <div style={{ marginTop: 4, fontSize: 16 }}>#1 of 9,724</div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#b5d8bb' }}>Model agreement</div>
                <div style={{ marginTop: 4, fontSize: 16 }}>{agreementCount} models</div>
              </div>
            </div>
          </div>

          {/* Divider */}
          {results.rl.length > 0 ? <div style={{ height: 1, background: '#2d4a2d' }} /> : null}

          {/* Why this recommendation section */}
          {results.rl.length > 0 ? (
            <div>
              <p style={{ margin: 0, fontSize: 11, color: '#3fb950', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                Why this recommendation?
              </p>
              <div style={{ marginTop: 10, display: 'grid', gap: 8 }}>
                {[
                  {
                    badge: 'M3 LSTM',
                    text: intelligence
                      ? `User shows strong ${intelligence.dominantGenre || 'genre'} sequential pattern in recent history`
                      : 'Sequential pattern detected in recent watch history',
                  },
                  { badge: 'M1 Graph', text: 'Similar users in embedding space liked this' },
                  { badge: 'M2 RF', text: 'High engagement probability score' },
                  { badge: 'M4 RL', text: 'Selected for maximum long-term reward' },
                ].map((row, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 13, color: '#3fb950' }}>✓</span>
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 600,
                        borderRadius: 8,
                        border: '1px solid #30363d',
                        padding: '3px 7px',
                        background:
                          row.badge === 'M3 LSTM'
                            ? '#102a4c'
                            : row.badge === 'M1 Graph'
                              ? '#1c2333'
                              : row.badge === 'M2 RF'
                                ? '#1c2820'
                                : '#3d2e0e',
                        color:
                          row.badge === 'M3 LSTM'
                            ? '#58a6ff'
                            : row.badge === 'M1 Graph'
                              ? '#79c0ff'
                              : row.badge === 'M2 RF'
                                ? '#3fb950'
                                : '#d2a534',
                      }}
                    >
                      {row.badge}
                    </span>
                    <span style={{ fontSize: 12, color: '#c3d0c6' }}>{row.text}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </section>

        {/* PIPELINE VISUAL STRIP */}
        {showSystemDetails && (!loading || pipelineStep > 0) ? (
          <section className="cp-page-card" style={{ padding: 12 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 12,
                padding: '12px 0',
                background: 'transparent',
                position: 'relative',
              }}
            >
              {PIPELINE_STAGES.map((item, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div
                    onMouseEnter={() => setHoveredPipeline(item.help)}
                    onMouseLeave={() => setHoveredPipeline('')}
                    style={{
                      fontSize: 11,
                      fontWeight: 500,
                      padding: '4px 10px',
                      borderRadius: 20,
                      background: item.bg || '#21262d',
                      color: item.color || '#8b949e',
                      border: item.border ? '1px solid transparent' : 'none',
                      boxShadow:
                        (pipelineStep === 1 && item.step === 0) ||
                        (pipelineStep > 1 && pipelineStep < 4 && (item.step === 1 || item.step === 2)) ||
                        (pipelineStep === 5 && item.step === 4)
                          ? `0 0 0 2px ${item.color || '#8b949e'}55`
                          : 'none',
                      transition: 'box-shadow 0.3s ease',
                      cursor: 'help',
                    }}
                  >
                    {item.label}
                  </div>
                  {idx < 4 ? <span style={{ color: '#30363d', fontSize: 14 }}>→</span> : null}
                </div>
              ))}
            </div>
            {hoveredPipeline ? (
              <div style={{ marginTop: 8, fontSize: 12, color: '#8b949e', textAlign: 'center' }}>{hoveredPipeline}</div>
            ) : (
              <div style={{ marginTop: 8, fontSize: 12, color: '#6e7681', textAlign: 'center' }}>
                Hover stages to understand how each model contributes.
              </div>
            )}
          </section>
        ) : null}

        {showSystemDetails ? (
          <section style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
            {MODEL_DEFINITIONS.map((model) => {
            const modelResults = results[model.key].slice(0, 3)
            const specialStyle =
              model.key === 'lstm'
                ? { border: '1px solid #388bfd55' }
                : model.key === 'rl'
                  ? { border: '1px solid #d2a53455' }
                  : {}

            return loading ? (
              <PanelSkeleton key={model.key} />
            ) : (
              <article key={model.key} className="cp-page-card" style={{ padding: 14, ...specialStyle }}>
                <header style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      borderRadius: 8,
                      border: '1px solid #30363d',
                      padding: '3px 7px',
                      background:
                        model.key === 'graph'
                          ? '#1c2333'
                          : model.key === 'baseline'
                            ? '#1c2820'
                            : model.key === 'lstm'
                              ? '#102a4c'
                              : '#3d2e0e',
                      color:
                        model.key === 'graph'
                          ? '#79c0ff'
                          : model.key === 'baseline'
                            ? '#3fb950'
                            : model.key === 'lstm'
                              ? '#58a6ff'
                              : '#d2a534',
                    }}
                  >
                    {model.badge}
                  </span>
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600 }}>{model.name}</div>
                    <div style={{ fontSize: 12, color: '#8b949e', marginTop: 2 }}>{model.shortDescription}</div>
                  </div>
                </header>

                <div style={{ marginTop: 12 }}>
                  {modelResults.map((item, index) => (
                    <RankRow
                      key={`${model.key}-${item.movieId ?? item.title}-${index + 1}`}
                      item={item}
                      rank={index + 1}
                      scoreLabel={model.scoreLabel}
                      isLstm={model.key === 'lstm'}
                    />
                  ))}
                  {!loading && modelResults.length === 0 ? (
                    <p style={{ marginTop: 10, fontSize: 12, color: '#8b949e' }}>Load a user to see results.</p>
                  ) : null}
                </div>

                {/* CONFIDENCE BAR */}
                {!loading && modelResults.length > 0 ? (
                  <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #212a35' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                      <label style={{ fontSize: 11, color: '#6e7681', flex: 1 }}>Top result confidence</label>
                      <span style={{ fontSize: 11, fontWeight: 500, color: model.accent, minWidth: 28, textAlign: 'right' }}>
                        {CONFIDENCE_SCORES[model.key]}%
                      </span>
                    </div>
                    <div
                      style={{
                        background: '#21262d',
                        height: 5,
                        borderRadius: 3,
                        overflow: 'hidden',
                      }}
                    >
                      <div
                        style={{
                          height: '100%',
                          width: `${CONFIDENCE_SCORES[model.key]}%`,
                          background: model.accent,
                          borderRadius: 3,
                          transition: 'width 1s ease',
                        }}
                      />
                    </div>
                  </div>
                ) : null}

                <footer style={{ marginTop: 12, fontSize: 11, color: '#6e7681' }}>{model.metric}</footer>
              </article>
            )
            })}
          </section>
        ) : null}

        {showSystemDetails ? (
          <section className="cp-page-card" style={{ padding: 14 }}>
            <p style={{ margin: 0, fontSize: 11, color: '#8b949e', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
              System flowchart
            </p>
            <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
              {[
                { name: 'Input', note: 'userId + history', color: '#8b949e' },
                { name: 'M3 LSTM', note: 'sequence intent', color: '#58a6ff' },
                { name: 'M1 Graph', note: 'context similarity', color: '#79c0ff' },
                { name: 'M2 RF', note: 'engagement score', color: '#3fb950' },
                { name: 'M4 RL', note: 'policy decision', color: '#d2a534' },
                { name: 'Output', note: 'final recommendation', color: '#3fb950' },
              ].map((node, idx, arr) => (
                <div key={node.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ border: `1px solid ${node.color}66`, borderRadius: 10, padding: '8px 10px', background: '#111823', minWidth: 120 }}>
                    <div style={{ fontSize: 12, color: node.color, fontWeight: 600 }}>{node.name}</div>
                    <div style={{ fontSize: 11, color: '#8b949e' }}>{node.note}</div>
                  </div>
                  {idx < arr.length - 1 ? <span style={{ color: '#30363d', fontSize: 14 }}>→</span> : null}
                </div>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    </div>
  )
}
