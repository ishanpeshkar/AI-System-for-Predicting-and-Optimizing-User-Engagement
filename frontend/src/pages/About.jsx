export default function About() {
  const members = [
    {
      initial: "H",
      name: "Harsh Kumar",
      reg: "22BCE11405",
      module: "M5 — Backend Integration & Dashboard",
      tech: "FastAPI · React 18 · Vite",
      metric: "6 pages · 4 API endpoints",
      color: "#E05A5A",
      bg: "#2d1c1c",
      desc:
        "Built the complete FastAPI backend and React dashboard from scratch. Implemented parallel model inference with Promise.all, the User Intelligence Panel, explainability layer, live feedback loop, and all 6 application pages.",
    },
    {
      initial: "A",
      name: "Ayush Raj",
      reg: "22BCE10636",
      module: "M1 — Graph Representation Learning",
      tech: "Node2Vec · networkx · scikit-learn",
      metric: "Cosine similarity: 0.91",
      color: "#7B5EA7",
      bg: "#1e1630",
      desc:
        "Constructed the bipartite user-movie graph from MovieLens interactions and applied Node2Vec random walks to generate 64-dimensional embedding vectors for all 610 users and 9,724 movies.",
    },
    {
      initial: "O",
      name: "Om Milind Kabade",
      reg: "22BCE10890",
      module: "M2 — Feature Engineering & Baseline ML",
      tech: "Random Forest · XGBoost · pandas",
      metric: "AUC: 0.74",
      color: "#2DBD8F",
      bg: "#0e2318",
      desc:
        "Engineered user, movie, and interaction features from raw data and trained a Random Forest classifier to predict engagement probability (0–1) for any user-movie pair. Establishes the baseline all other models are compared against.",
    },
    {
      initial: "V",
      name: "Vikesh Mehta",
      reg: "22BCE11631",
      module: "M4 — Reinforcement Learning Agent",
      tech: "Thompson Sampling · Bandit · JSON",
      metric: "Regret reduction: 61%",
      color: "#F5A623",
      bg: "#2a1e05",
      desc:
        "Implemented the Multi-Armed Bandit agent using Thompson Sampling. Maintains Beta(α, β) distributions per movie arm and updates them in real time from user feedback — no retraining required. Avg reward improved from −0.3 to +1.42 over 5,000 episodes.",
    },
    {
      initial: "I",
      name: "Ishan Peshkar",
      reg: "22BCE10502",
      module: "M3 — Sequential Deep Learning (LSTM)",
      tech: "PyTorch · Attention-LSTM · Adam",
      metric: "HR@10: 8.14%  ·  NDCG@10: 0.046",
      color: "#4F8EF7",
      bg: "#0d1e3d",
      desc:
        "Built the Attention-LSTM model that captures temporal user behavior. Processes the last 10 watched movies as an ordered sequence and outputs a probability distribution over all 9,724 movies. Exports 128-dim hidden state vector to the RL agent via extract_states.py.",
    },
  ];

  const pipeline = [
    { who: "Ayush (M1)", out: "64-dim embedding vectors", color: "#7B5EA7" },
    { who: "Om (M2)", out: "Engagement probability 0–1", color: "#2DBD8F" },
    { who: "Ishan (M3)", out: "128-dim hidden state vector", color: "#4F8EF7" },
    { who: "Harsh (M4)", out: "Final recommendation + reward", color: "#F5A623" },
    { who: "Vikesh (M5)", out: "Dashboard + live feedback loop", color: "#E05A5A" },
  ];

  const metrics = [
    { label: "HR@10", value: "8.14%", sub: "LSTM sequential model", color: "#4F8EF7" },
    { label: "Pipeline AUC", value: "0.84", sub: "vs 0.74 baseline alone", color: "#00C2CB" },
    { label: "Regret ↓", value: "61%", sub: "RL Thompson Sampling", color: "#F5A623" },
    { label: "Embeddings", value: "64-dim", sub: "Node2Vec graph", color: "#7B5EA7" },
  ];

  return (
    <div style={{ display: "grid", gap: 14 }}>

      {/* ── Project overview ── */}
      <section className="cp-page-card" style={{ padding: 20 }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div style={{ flex: 1, minWidth: 260 }}>
            <h3 style={{ margin: 0, fontSize: 20, color: "#e6edf3" }}>CinePulse</h3>
            <p style={{ margin: "6px 0 0", color: "#8b949e", lineHeight: 1.65, maxWidth: 680, fontSize: 13 }}>
              An AI system for predicting and optimizing user engagement through multi-model signal fusion.
              Four independent models analyze the same user simultaneously — graph embeddings, baseline ML,
              sequential deep learning, and reinforcement learning — and the RL agent fuses all signals into
              a single explainable recommendation that improves with every user interaction.
            </p>
            <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
              {["VIT Bhopal University", "DSN4095 — Capstone Phase II", "March 2026"].map((tag) => (
                <span key={tag} style={{ fontSize: 11, padding: "2px 10px", borderRadius: 20, background: "#161b22", border: "1px solid #30363d", color: "#8b949e" }}>
                  {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Metric pills */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {metrics.map((m) => (
              <div key={m.label} style={{ background: "#0d1117", border: `1px solid ${m.color}40`, borderRadius: 10, padding: "8px 14px", minWidth: 120 }}>
                <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 2 }}>{m.label}</div>
                <div style={{ fontSize: 20, fontWeight: 600, color: m.color }}>{m.value}</div>
                <div style={{ fontSize: 10, color: "#6e7681", marginTop: 1 }}>{m.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Team members ── */}
      <section className="cp-page-card" style={{ padding: 20 }}>
        <h4 style={{ margin: "0 0 14px", fontSize: 14, color: "#e6edf3" }}>The team</h4>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 10 }}>
          {members.map((m) => (
            <div key={m.reg} style={{ background: m.bg, border: `1px solid ${m.color}55`, borderRadius: 10, padding: 14 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                <div style={{ width: 36, height: 36, borderRadius: "50%", background: m.color, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700, color: "#fff", flexShrink: 0 }}>
                  {m.initial}
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#e6edf3" }}>{m.name}</div>
                  <div style={{ fontSize: 10, color: "#6e7681" }}>{m.reg}</div>
                </div>
              </div>
              <div style={{ fontSize: 11, fontWeight: 600, color: m.color, marginBottom: 4 }}>{m.module}</div>
              <p style={{ fontSize: 11, color: "#8b949e", lineHeight: 1.55, margin: "0 0 8px" }}>{m.desc}</p>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 20, background: "#161b22", color: "#8b949e", border: "1px solid #30363d" }}>{m.tech}</span>
                <span style={{ fontSize: 10, padding: "2px 7px", borderRadius: 20, background: "#161b22", color: m.color, border: `1px solid ${m.color}40` }}>{m.metric}</span>
              </div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 12, fontSize: 11, color: "#6e7681" }}>
          Supervisor: Mr. Chour Singh Rajpoot · Department of Computer Science and Engineering
        </div>
      </section>

      {/* ── Pipeline integration map ── */}
      <section className="cp-page-card" style={{ padding: 20 }}>
        <h4 style={{ margin: "0 0 4px", fontSize: 14, color: "#e6edf3" }}>Pipeline integration</h4>
        <p style={{ margin: "0 0 14px", fontSize: 12, color: "#8b949e" }}>
          Each member's output feeds directly into the next stage. All models share{" "}
          <code style={{ fontSize: 11, background: "#0d1117", padding: "1px 5px", borderRadius: 4, color: "#79c0ff" }}>movie2idx.json</code>{" "}
          as a common vocabulary.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {pipeline.map((p, i) => (
            <div key={p.who}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 28, height: 28, borderRadius: "50%", background: p.color, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: "#fff", flexShrink: 0 }}>
                  {i + 1}
                </div>
                <div style={{ background: "#161b22", border: `1px solid ${p.color}50`, borderRadius: 8, padding: "7px 14px", flex: 1, display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 6 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: p.color }}>{p.who}</span>
                  <span style={{ fontSize: 11, color: "#8b949e" }}>→ outputs: <span style={{ color: "#e6edf3" }}>{p.out}</span></span>
                </div>
              </div>
              {i < pipeline.length - 1 && (
                <div style={{ width: 1, height: 8, background: "#30363d", marginLeft: 13 }} />
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ── Backend contract ── */}
      <section className="cp-page-card" style={{ padding: 20 }}>
        <h4 style={{ margin: "0 0 4px", fontSize: 14, color: "#e6edf3" }}>Backend contract</h4>
        <p style={{ margin: "0 0 12px", fontSize: 12, color: "#8b949e" }}>
          All models are served through a single FastAPI backend. The frontend never loads model files — all inference runs server-side.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 10 }}>
          {[
            {
              method: "POST", route: "/api/recommendations",
              body: `{ "userId": 1, "topK": 5,\n  "algorithm": "RL (Bandit)" }`,
              resp: "Ranked movie list with scores",
            },
            {
              method: "GET", route: "/api/user-profile/{userId}",
              body: "— (userId in path)",
              resp: "Watch history, stats, dominant genre",
            },
            {
              method: "POST", route: "/api/feedback",
              body: `{ "userId": 1, "movieId": 527,\n  "feedbackType": "liked" }`,
              resp: "Updates bandit Beta distribution",
            },
            {
              method: "GET", route: "/api/feedback/stats",
              body: "— (no body)",
              resp: "Total feedback, positive rate, recent log",
            },
          ].map((ep) => (
            <div key={ep.route} style={{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 10, padding: 12 }}>
              <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 4, background: ep.method === "POST" ? "#1c2d1c" : "#102a4c", color: ep.method === "POST" ? "#3fb950" : "#58a6ff" }}>
                  {ep.method}
                </span>
                <code style={{ fontSize: 11, color: "#79c0ff" }}>{ep.route}</code>
              </div>
              <pre style={{ margin: 0, fontSize: 11, color: "#8b949e", whiteSpace: "pre-wrap", lineHeight: 1.5 }}>{ep.body}</pre>
              <div style={{ marginTop: 6, fontSize: 11, color: "#6e7681" }}>↳ {ep.resp}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Tech stack ── */}
      <section className="cp-page-card" style={{ padding: 20 }}>
        <h4 style={{ margin: "0 0 12px", fontSize: 14, color: "#e6edf3" }}>Tech stack</h4>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10 }}>
          {[
            { layer: "ML / DL", items: ["PyTorch 2.0", "scikit-learn", "XGBoost", "node2vec", "networkx"], color: "#4F8EF7" },
            { layer: "Backend", items: ["FastAPI", "Uvicorn", "pandas", "numpy"], color: "#2DBD8F" },
            { layer: "Frontend", items: ["React 18", "Vite", "React Router DOM v6", "Chart.js"], color: "#E05A5A" },
            { layer: "Dataset", items: ["MovieLens ml-latest-small", "610 users · 9,724 movies", "100,836 ratings"], color: "#F5A623" },
          ].map((s) => (
            <div key={s.layer} style={{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 10, padding: 12 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: s.color, marginBottom: 8 }}>{s.layer}</div>
              {s.items.map((it) => (
                <div key={it} style={{ fontSize: 11, color: "#8b949e", padding: "3px 0", borderBottom: "1px solid #21262d" }}>
                  {it}
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}