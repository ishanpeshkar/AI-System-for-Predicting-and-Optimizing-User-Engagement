export default function About() {
  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <section className="cp-page-card" style={{ padding: 16 }}>
        <h3 style={{ margin: 0, fontSize: 18 }}>About CinePulse</h3>
        <p style={{ margin: '8px 0 0', color: '#8b949e', maxWidth: 780, lineHeight: 1.6 }}>
          CinePulse is a research interface for comparing movie recommendations from four distinct models: Graph
          embeddings, Baseline RF, Sequential LSTM, and RL Bandit optimization.
        </p>
      </section>

      <section className="cp-page-card" style={{ padding: 16 }}>
        <h4 style={{ margin: 0, fontSize: 14 }}>Backend contract</h4>
        <pre
          style={{
            margin: '10px 0 0',
            background: '#0d1117',
            border: '1px solid #30363d',
            borderRadius: 10,
            padding: 12,
            color: '#79c0ff',
            fontSize: 12,
            overflow: 'auto',
          }}
        >
{`POST /api/recommendations
{
  "userId": 1,
  "topK": 5,
  "algorithm": "RL (Bandit)"
}`}
        </pre>
      </section>

      <section className="cp-page-card" style={{ padding: 16 }}>
        <h4 style={{ margin: 0, fontSize: 14 }}>Developed by</h4>
        <p style={{ margin: '8px 0 0', color: '#8b949e' }}>Harsh Kumar, Vikesh Mehta, Ishan, Ayush, Om Kabade</p>
      </section>
    </div>
  )
}
