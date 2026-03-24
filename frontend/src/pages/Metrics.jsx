import { useEffect, useState } from 'react'
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Tooltip,
} from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

const trainingLoss = [4.12, 3.54, 3.01, 2.63, 2.31, 2.08, 1.89, 1.74, 1.62, 1.53, 1.46, 1.41, 1.37, 1.34, 1.31]

const rlRewards = [
  -0.3, -0.25, -0.22, -0.18, -0.12, -0.07, -0.03, 0.02, 0.08, 0.16,
  0.11, 0.2, 0.26, 0.34, 0.31, 0.4, 0.47, 0.54, 0.61, 0.66,
  0.63, 0.71, 0.78, 0.86, 0.91, 0.95, 1.0, 1.05, 1.09, 1.14,
  1.18, 1.21, 1.16, 1.24, 1.29, 1.33, 1.3, 1.36, 1.4, 1.38,
]

const baselineAucTrend = [0.66, 0.68, 0.69, 0.71, 0.72, 0.73, 0.74]
const baselineF1Trend = [0.58, 0.6, 0.62, 0.63, 0.65, 0.66, 0.67]

function chartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { color: '#8b949e' },
        grid: { color: 'rgba(48,54,61,0.8)' },
      },
      y: {
        ticks: { color: '#8b949e' },
        grid: { color: 'rgba(48,54,61,0.8)' },
      },
    },
  }
}

export default function Metrics() {
  const [feedbackStats, setFeedbackStats] = useState(null)

  useEffect(() => {
    fetch('http://localhost:8000/api/feedback/stats')
      .then((response) => response.json())
      .then(setFeedbackStats)
      .catch(() => {})
  }, [])

  const lossData = {
    labels: Array.from({ length: 15 }, (_, index) => `Epoch ${index + 1}`),
    datasets: [
      {
        label: 'Loss',
        data: trainingLoss,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.06)',
        fill: true,
        tension: 0.3,
      },
    ],
  }

  const rewardData = {
    labels: Array.from({ length: 40 }, (_, index) => Math.round((index / 39) * 5000)),
    datasets: [
      {
        label: 'Reward',
        data: rlRewards,
        borderColor: '#3fb950',
        backgroundColor: 'rgba(63,185,80,0.05)',
        fill: true,
        tension: 0.25,
      },
    ],
  }

  const baselineData = {
    labels: ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Fold 6', 'Fold 7'],
    datasets: [
      {
        label: 'AUC',
        data: baselineAucTrend,
        borderColor: '#3fb950',
        backgroundColor: 'rgba(63,185,80,0.04)',
        fill: true,
        tension: 0.24,
      },
      {
        label: 'F1',
        data: baselineF1Trend,
        borderColor: '#79c0ff',
        backgroundColor: 'rgba(121,192,255,0.03)',
        fill: true,
        tension: 0.24,
      },
    ],
  }

  const feedbackTimeline = feedbackStats?.recent?.slice().reverse() || []
  const feedbackGraphData = {
    labels: feedbackTimeline.map((item, index) => `F${index + 1}`),
    datasets: [
      {
        label: 'Reward',
        data: feedbackTimeline.map((item) => Number(item.reward || 0)),
        borderColor: '#d2a534',
        backgroundColor: 'rgba(210,165,52,0.08)',
        fill: true,
        tension: 0.22,
      },
    ],
  }

  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 10 }}>
        {[
          ['HR@10', '8.14%', 'LSTM sequential model'],
          ['NDCG@10', '0.046', 'Position-weighted accuracy'],
          ['Pipeline AUC', '0.84', 'vs 0.74 baseline alone'],
          ['Regret reduction', '61%', 'Thompson Sampling RL'],
        ].map(([label, value, detail]) => (
          <article key={label} className="cp-page-card" style={{ padding: 14 }}>
            <div style={{ color: '#8b949e', fontSize: 12 }}>{label}</div>
            <div style={{ marginTop: 8, fontSize: 24, fontWeight: 500 }}>{value}</div>
            <div style={{ marginTop: 6, fontSize: 12, color: '#6e7681' }}>{detail}</div>
          </article>
        ))}
      </section>

      <section className="cp-page-card" style={{ padding: 14 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>M3 LSTM - training loss per epoch</h3>
        <div style={{ marginTop: 10, height: 260, background: '#161b22' }}>
          <Line data={lossData} options={chartOptions()} />
        </div>
      </section>

      <section className="cp-page-card" style={{ padding: 14 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>M4 RL agent - cumulative average reward</h3>
        <div style={{ marginTop: 10, height: 260, background: '#161b22' }}>
          <Line data={rewardData} options={chartOptions()} />
        </div>
      </section>

      <section className="cp-page-card" style={{ padding: 14 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>M2 Baseline (RF) - validation trend</h3>
        <div style={{ marginTop: 10, height: 260, background: '#161b22' }}>
          <Line data={baselineData} options={chartOptions()} />
        </div>
      </section>

      {feedbackStats && (
        <section className="cp-page-card" style={{ padding: 14 }}>
          <div style={{ marginBottom: '20px' }}>
            <div
              style={{
                fontSize: '11px',
                fontWeight: '500',
                color: '#8b949e',
                letterSpacing: '0.06em',
                textTransform: 'uppercase',
                marginBottom: '10px',
              }}
            >
              Live feedback loop - RL model updates in real time
            </div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4,minmax(0,1fr))',
                gap: '8px',
                marginBottom: '12px',
              }}
            >
              {[
                { label: 'Total feedback', value: feedbackStats.total_feedback },
                { label: 'Positive rate', value: `${feedbackStats.positive_rate}%`, color: '#3fb950' },
                {
                  label: 'Model updates',
                  value: feedbackStats.model_updates || feedbackStats.total_feedback,
                  color: '#58a6ff',
                },
                { label: 'Liked', value: feedbackStats.liked },
              ].map((card, index) => (
                <div
                  key={index}
                  style={{
                    background: '#161b22',
                    border: '1px solid #30363d',
                    borderRadius: '8px',
                    padding: '10px 12px',
                  }}
                >
                  <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '4px' }}>{card.label}</div>
                  <div style={{ fontSize: '20px', fontWeight: '500', color: card.color || '#e6edf3' }}>
                    {card.value}
                  </div>
                </div>
              ))}
            </div>

            {feedbackStats.recent?.length > 0 && (
              <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: '10px', padding: '12px' }}>
                <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '8px' }}>Recent feedback log</div>
                {feedbackStats.recent.map((item, index) => (
                  <div
                    key={index}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      padding: '5px 0',
                      borderBottom: '1px solid #21262d',
                      fontSize: '12px',
                    }}
                  >
                    <span style={{ color: '#e6edf3' }}>{item.movie_title}</span>
                    <span
                      style={{
                        color: item.reward > 0 ? '#3fb950' : '#f85149',
                        fontFamily: 'monospace',
                      }}
                    >
                      {item.feedback_type} ({item.reward > 0 ? '+' : ''}
                      {item.reward})
                    </span>
                  </div>
                ))}
              </div>
            )}

            <div style={{ marginTop: 14, background: '#161b22', border: '1px solid #30363d', borderRadius: '10px', padding: '12px' }}>
              <div style={{ fontSize: '11px', color: '#8b949e', marginBottom: '8px' }}>
                Feedback reward trend
              </div>
              <div style={{ height: 220 }}>
                {feedbackTimeline.length > 0 ? (
                  <Line data={feedbackGraphData} options={chartOptions()} />
                ) : (
                  <div style={{ fontSize: 12, color: '#8b949e', paddingTop: 10 }}>
                    Feedback graph will appear after users submit reactions.
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      <section className="cp-page-card" style={{ padding: 14 }}>
        <h3 style={{ margin: 0, fontSize: 14 }}>Baseline vs full pipeline</h3>
        <table style={{ width: '100%', marginTop: 10, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: '#8b949e', fontSize: 12, textAlign: 'left' }}>
              <th style={{ padding: '8px', borderBottom: '1px solid #30363d' }}>Model config</th>
              <th style={{ padding: '8px', borderBottom: '1px solid #30363d' }}>AUC</th>
              <th style={{ padding: '8px', borderBottom: '1px solid #30363d' }}>F1</th>
              <th style={{ padding: '8px', borderBottom: '1px solid #30363d' }}>Notes</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>M2 alone (RF)</td>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>0.74</td>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>0.67</td>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35', color: '#8b949e' }}>Baseline</td>
            </tr>
            <tr>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>M3 alone (LSTM)</td>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>0.79</td>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>0.71</td>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35', color: '#8b949e' }}>+5 AUC pts</td>
            </tr>
            <tr style={{ background: '#1c2d1c' }}>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>All 4 models combined</td>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>0.84</td>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>0.78</td>
              <td style={{ padding: '9px 8px', borderBottom: '1px solid #232b35' }}>+10 AUC pts ✓</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  )
}
