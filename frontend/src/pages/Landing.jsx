import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

/* ─── Scroll-reveal hook ─────────────────────────────────────────── */
function useReveal(threshold = 0.12) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); obs.disconnect(); } },
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return [ref, visible];
}

function Reveal({ children, delay = 0, y = 20, className = "" }) {
  const [ref, visible] = useReveal();
  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "none" : `translateY(${y}px)`,
        transition: `opacity 0.6s cubic-bezier(.16,1,.3,1) ${delay}ms, transform 0.6s cubic-bezier(.16,1,.3,1) ${delay}ms`,
      }}
    >
      {children}
    </div>
  );
}

/* ─── Small helpers ──────────────────────────────────────────────── */
function SectionLabel({ children }) {
  return (
    <p style={{
      fontSize: 11, fontWeight: 600, letterSpacing: "0.1em",
      textTransform: "uppercase", color: "var(--accent)",
      marginBottom: 12, fontFamily: "var(--font-mono)",
    }}>
      {children}
    </p>
  );
}

function Divider() {
  return (
    <div style={{
      height: 1, background: "var(--border)",
      maxWidth: 1152, margin: "0 auto",
    }} />
  );
}

function Stat({ title, value, subtitle }) {
  return (
    <div style={{
      padding: "14px 20px", borderRadius: 12,
      border: "1px solid var(--border)", background: "var(--surface)",
      textAlign: "left", minWidth: 180,
      flex: "1 1 180px",
    }}>
      <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 6, letterSpacing: "0.08em", textTransform: "uppercase" }}>
        {title}
      </div>
      <div style={{ fontSize: 20, fontWeight: 700, color: "var(--text-primary)", fontFamily: "var(--font-mono)", lineHeight: 1.1 }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 6 }}>
        {subtitle}
      </div>
    </div>
  );
}

/* ─── Feature card ───────────────────────────────────────────────── */
function FeatureCard({ icon, title, description, delay }) {
  const [hov, setHov] = useState(false);
  return (
    <Reveal delay={delay}>
      <div
        onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
        style={{
          background: hov ? "var(--surface-hover)" : "var(--surface)",
          border: `1px solid ${hov ? "var(--border-strong)" : "var(--border)"}`,
          borderRadius: 16, padding: "22px 20px", height: "100%",
          transition: "background 0.2s, border-color 0.2s", cursor: "default",
        }}
      >
        <div style={{
          width: 36, height: 36, borderRadius: 10,
          background: "var(--accent-subtle)", border: "1px solid var(--accent-border)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 16, marginBottom: 14,
        }}>
          {icon}
        </div>
        <h3 style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", marginBottom: 6 }}>{title}</h3>
        <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.65, margin: 0 }}>{description}</p>
      </div>
    </Reveal>
  );
}

/* ─── Model card ─────────────────────────────────────────────────── */
const ACCENTS = ["#6366f1", "#8b5cf6", "#3b82f6", "#10b981"];

function ModelCard({ label, title, description, index }) {
  const color = ACCENTS[index];
  const [hov, setHov] = useState(false);
  return (
    <Reveal delay={index * 70}>
      <div
        onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
        style={{
          borderRadius: 16,
          border: `1px solid ${hov ? color + "55" : "var(--border)"}`,
          background: hov ? `${color}09` : "var(--surface)",
          padding: "20px", height: "100%",
          transition: "border-color 0.2s, background 0.2s",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
          <span style={{ width: 7, height: 7, borderRadius: "50%", background: color, boxShadow: `0 0 8px ${color}88`, flexShrink: 0 }} />
          <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color, fontFamily: "var(--font-mono)" }}>
            {label}
          </span>
        </div>
        <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 6 }}>{title}</div>
        <p style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.65, margin: 0 }}>{description}</p>
      </div>
    </Reveal>
  );
}

/* ─── FAQ accordion ──────────────────────────────────────────────── */
function FAQ({ q, a, delay }) {
  const [open, setOpen] = useState(false);
  return (
    <Reveal delay={delay}>
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          borderRadius: 14,
          border: "1px solid var(--border)",
          background: open ? "var(--surface-hover)" : "var(--surface)",
          overflow: "hidden", cursor: "pointer",
          transition: "background 0.2s",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, padding: "16px 18px" }}>
          <span style={{ fontSize: 13.5, fontWeight: 500, color: "var(--text-primary)" }}>{q}</span>
          <span style={{ color: "var(--text-muted)", fontSize: 18, lineHeight: 1, transform: open ? "rotate(45deg)" : "none", transition: "transform 0.25s", flexShrink: 0 }}>
            +
          </span>
        </div>
        <div style={{ maxHeight: open ? 120 : 0, opacity: open ? 1 : 0, overflow: "hidden", transition: "max-height 0.3s cubic-bezier(.16,1,.3,1), opacity 0.25s" }}>
          <p style={{ margin: 0, padding: "0 18px 16px", fontSize: 13, color: "var(--text-muted)", lineHeight: 1.65 }}>{a}</p>
        </div>
      </div>
    </Reveal>
  );
}

/* ─── Code panel ─────────────────────────────────────────────────── */
function CodePanel({ label, children }) {
  return (
    <div style={{ borderRadius: 0, background: "var(--surface-deep)", overflow: "hidden" }}>
      <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--border)", fontSize: 10, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-subtle)", fontFamily: "var(--font-mono)" }}>
        {label}
      </div>
      <div style={{ padding: "14px 16px", fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.75 }}>
        {children}
      </div>
    </div>
  );
}

/* ─── Landing ────────────────────────────────────────────────────── */
export default function Landing() {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const h = () => setScrolled(window.scrollY > 24);
    window.addEventListener("scroll", h, { passive: true });
    return () => window.removeEventListener("scroll", h);
  }, []);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
        :root{
          --bg:#09090b;
          --surface:rgba(255,255,255,0.035);
          --surface-hover:rgba(255,255,255,0.055);
          --surface-deep:rgba(0,0,0,0.3);
          --border:rgba(255,255,255,0.08);
          --border-strong:rgba(255,255,255,0.15);
          --text-primary:#f4f4f5;
          --text-secondary:#a1a1aa;
          --text-muted:#71717a;
          --text-subtle:#3f3f46;
          --accent:#818cf8;
          --accent-strong:#6366f1;
          --accent-subtle:rgba(99,102,241,0.1);
          --accent-border:rgba(99,102,241,0.25);
          --font-sans:'Geist',system-ui,sans-serif;
          --font-mono:'Geist Mono',monospace;
        }
        html{scroll-behavior:smooth;}
        body{background:var(--bg);color:var(--text-primary);font-family:var(--font-sans);-webkit-font-smoothing:antialiased;}
        ::selection{background:rgba(99,102,241,0.3);}
        @keyframes hero-up{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:none;}}
        @keyframes hero-scale{from{opacity:0;transform:scale(0.97);}to{opacity:1;transform:none;}}
        .h0{animation:hero-up .7s cubic-bezier(.16,1,.3,1) 0ms both;}
        .h1{animation:hero-up .7s cubic-bezier(.16,1,.3,1) 80ms both;}
        .h2{animation:hero-up .7s cubic-bezier(.16,1,.3,1) 160ms both;}
        .h3{animation:hero-up .7s cubic-bezier(.16,1,.3,1) 240ms both;}
        .h4{animation:hero-up .7s cubic-bezier(.16,1,.3,1) 320ms both;}
        .hi{animation:hero-scale .8s cubic-bezier(.16,1,.3,1) 100ms both;}
        .btn-primary{
          display:inline-flex;align-items:center;gap:6px;
          background:var(--accent-strong);color:#fff;
          border:none;border-radius:10px;padding:11px 22px;
          font-size:14px;font-weight:600;font-family:var(--font-sans);
          cursor:pointer;letter-spacing:-0.01em;
          transition:background .18s,box-shadow .18s,transform .15s;
        }
        .btn-primary:hover{background:#4f46e5;box-shadow:0 0 0 4px rgba(99,102,241,.2);transform:translateY(-1px);}
        .btn-primary:active{transform:none;}
        .btn-ghost{
          display:inline-flex;align-items:center;gap:6px;
          background:transparent;color:var(--text-muted);
          border:1px solid var(--border);border-radius:10px;padding:10px 18px;
          font-size:13px;font-weight:500;font-family:var(--font-sans);
          cursor:pointer;transition:border-color .18s,color .18s,background .18s;
        }
        .btn-ghost:hover{border-color:var(--border-strong);color:var(--text-primary);background:var(--surface);}
        .nav-link{
          font-size:13px;font-weight:500;color:var(--text-muted);
          text-decoration:none;padding:6px 10px;border-radius:7px;
          transition:color .15s;
        }
        .nav-link:hover{color:var(--text-primary);}
        .wrap{max-width:1152px;margin:0 auto;padding:0 clamp(16px,4vw,48px);}
        .section{padding:80px clamp(16px,4vw,48px);}
        .grid-hero{display:grid;grid-template-columns:1fr 1fr;gap:56px;align-items:center;}
        .grid-3{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;}
        .grid-4{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px;}
        .grid-2{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;}
        .api-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1px;background:var(--border);}
        @media(max-width:768px){
          .grid-hero{grid-template-columns:1fr;}
          .img-col{order:-1;}
          .hide-mobile{display:none!important;}
        }
      `}</style>

      <div style={{ background: "var(--bg)", minHeight: "100vh" }}>

        {/* ── HEADER ── */}
        <header style={{
          position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
          borderBottom: `1px solid ${scrolled ? "var(--border)" : "transparent"}`,
          background: scrolled ? "rgba(9,9,11,0.85)" : "transparent",
          backdropFilter: scrolled ? "blur(20px)" : "none",
          transition: "background .3s,border-color .3s",
        }}>
          <div className="wrap" style={{ height: 60, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
            {/* Logo */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{
                width: 32, height: 32, borderRadius: 9,
                background: "var(--accent-strong)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#fff",
              }}>CP</div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>CinePulse</div>
                <div style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", letterSpacing: "0.04em" }}>
                  Four models, one interface
                </div>
              </div>
            </div>
            {/* Nav */}
            <nav style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {["Models", "Team", "API", "FAQ"].map(item => (
                <a key={item} className="nav-link hide-mobile" href={`#${item.toLowerCase()}`}>{item}</a>
              ))}
              <button className="btn-ghost" onClick={() => navigate('/dashboard')} style={{ marginLeft: 8 }}>Launch app →</button>
            </nav>
          </div>
        </header>

        {/* ── HERO ── */}
        <section style={{ padding: "120px clamp(16px,4vw,48px) 80px", maxWidth: 1152, margin: "0 auto" }}>
          <div className="grid-hero">
            {/* Left */}
            <div>
              <div className="h0" style={{ marginBottom: 20 }}>
                <span style={{
                  display: "inline-flex", alignItems: "center", gap: 7,
                  fontSize: 11, fontWeight: 500, letterSpacing: "0.07em", textTransform: "uppercase",
                  color: "var(--accent)", background: "var(--accent-subtle)",
                  border: "1px solid var(--accent-border)", borderRadius: 999, padding: "4px 12px",
                }}>
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)", display: "inline-block" }} />
                  ML-powered · 4 algorithms
                </span>
              </div>

              <h1 className="h1" style={{ fontSize: "clamp(36px,5vw,52px)", fontWeight: 700, letterSpacing: "-0.04em", lineHeight: 1.08, color: "var(--text-primary)" }}>
                Four AI models.
                <br />
                <span style={{ color: "var(--accent)" }}>One decision engine. See how each one thinks.</span>
              </h1>

              <p className="h2" style={{ marginTop: 18, fontSize: 15, color: "var(--text-secondary)", lineHeight: 1.7, maxWidth: 440 }}>
                CinePulse is a research system that runs four different ML algorithms on the same user simultaneously — graph embeddings, baseline ML, sequential deep learning, and reinforcement learning — and shows you exactly how each one reasons.
              </p>

              <div className="h3" style={{ marginTop: 28, display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10 }}>
                <button className="btn-primary" onClick={() => navigate('/dashboard')}>Open Dashboard →</button>
                <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Simple, fast, and professional.</span>
              </div>

              <div className="h4" style={{ marginTop: 32, display: "flex", gap: 10, flexWrap: "wrap" }}>
                <Stat title="HR@10" value="8.14%" subtitle="LSTM sequential model" />
                <Stat title="AUC" value="0.84" subtitle="Full 4-model pipeline" />
                <Stat title="Regret ↓" value="61%" subtitle="RL Thompson Sampling" />
                <Stat title="Embeddings" value="64-dim" subtitle="Node2Vec graph" />
              </div>
            </div>

            {/* Right */}
            <div className="hi img-col" style={{ position: "relative" }}>
              <div style={{
                position: "absolute", inset: "-20px", borderRadius: 36,
                background: "radial-gradient(ellipse,rgba(99,102,241,.16) 0%,transparent 65%)",
                pointerEvents: "none", filter: "blur(14px)",
              }} />
              <div style={{
                position: "relative", borderRadius: 24, overflow: "hidden",
                border: "1px solid var(--border)",
                boxShadow: "0 24px 72px rgba(0,0,0,.55)",
              }}>
                <img
                  src="https://images.unsplash.com/photo-1485846234645-a62644f84728?auto=format&fit=crop&w=1200&q=80"
                  alt="Cinema audience"
                  style={{ width: "100%", height: 400, objectFit: "cover", display: "block" }}
                  loading="lazy"
                />
                <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to top,rgba(9,9,11,.65) 0%,transparent 60%)" }} />
                {/* Chip */}
                <div style={{
                  position: "absolute", bottom: 14, left: 14, right: 14,
                  borderRadius: 12, padding: "10px 14px",
                  background: "rgba(9,9,11,.82)", backdropFilter: "blur(16px)",
                  border: "1px solid var(--border)",
                  display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10,
                }}>
                  <div>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", letterSpacing: "0.06em", textTransform: "uppercase" }}>
                      Top recommendation
                    </div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginTop: 3 }}>
                      Interstellar · Score 0.97
                    </div>
                  </div>
                  <div style={{
                    width: 32, height: 32, borderRadius: 8,
                    background: "var(--accent-strong)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 11, flexShrink: 0, color: "#fff",
                  }}>▶</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <Divider />

        {/* ── WHY DIFFERENT ── */}
        <section id="features" style={{ padding: "80px clamp(16px,4vw,48px)", maxWidth: 1152, margin: "0 auto" }}>
          <Reveal>
            <SectionLabel>Why it feels different</SectionLabel>
            <h2 style={{ fontSize: "clamp(24px,3.5vw,32px)", fontWeight: 700, letterSpacing: "-0.03em" }}>Designed for clarity.</h2>
            <p style={{ marginTop: 8, fontSize: 14, color: "var(--text-muted)", maxWidth: 480 }}>
              Predictable navigation, consistent UI, and real model output — not mocked data.
            </p>
          </Reveal>
          <div className="grid-3" style={{ marginTop: 36 }}>
            <FeatureCard delay={0}   icon="🎯" title="Personalized ranking"  description="Enter a userId and receive a ranked list of movies tailored to that viewer's watch history and engagement patterns." />
            <FeatureCard delay={80}  icon="🔀" title="Multiple algorithms"   description="Compare RL bandit, sequential LSTM, graph embeddings, and baseline RF — each with a distinct philosophy." />
            <FeatureCard delay={160} icon="🔗" title="Clean integration"     description="Frontend talks to the backend via a simple JSON contract. No model files in the browser, no heavy dependencies." />
          </div>
        </section>

        <Divider />

        {/* ── MODELS ── */}
        <section id="models" style={{ padding: "80px clamp(16px,4vw,48px)", maxWidth: 1152, margin: "0 auto" }}>
          <Reveal>
            <SectionLabel>Four models</SectionLabel>
            <h2 style={{ fontSize: "clamp(24px,3.5vw,32px)", fontWeight: 700, letterSpacing: "-0.03em" }}>Explore the algorithms.</h2>
            <p style={{ marginTop: 8, fontSize: 14, color: "var(--text-muted)", maxWidth: 520 }}>
              Each algorithm gives a different lens on the recommendation problem. Use the dropdown on the next page to compare outputs.
            </p>
          </Reveal>
          <div className="grid-4" style={{ marginTop: 36 }}>
            <ModelCard index={0} label="RF"    title="Baseline (RF)"      description="Predicts engagement likelihood using carefully engineered tabular features." />
            <ModelCard index={1} label="Graph" title="Graph (Embeddings)" description="Uses relationship embeddings to surface structurally similar items." />
            <ModelCard index={2} label="LSTM"  title="Sequential (LSTM)"  description="Captures temporal viewing intent from a user's recent watch history." />
            <ModelCard index={3} label="RL"    title="RL (Bandit)"        description="Balances exploration vs. exploitation learned from real feedback signals." />
          </div>
        </section>

        <Divider />

        {/* ── TEAM ── */}
        <section id="team" style={{ padding: "80px clamp(16px,4vw,48px)", maxWidth: 1152, margin: "0 auto" }}>
          <div className="grid-hero" style={{ gap: 48 }}>
            <div>
              <Reveal>
                <SectionLabel>The team</SectionLabel>
                <h2 style={{ fontSize: "clamp(24px,3.5vw,32px)", fontWeight: 700, letterSpacing: "-0.03em" }}>Built at VIT Bhopal.</h2>
                <p style={{ marginTop: 10, fontSize: 14, color: "var(--text-muted)", lineHeight: 1.7, maxWidth: 400 }}>
                  Developed by five students passionate about machine learning and thoughtful product design.
                </p>
              </Reveal>
              <div style={{ marginTop: 28, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                {["Harsh Kumar","Vikesh Mehta","Ishan","Ayush","Om Kabade"].map((name, i) => (
                  <Reveal key={name} delay={i * 55}>
                    <div
                      style={{
                        display: "flex", alignItems: "center", gap: 10,
                        padding: "12px 14px", borderRadius: 12,
                        border: "1px solid var(--border)", background: "var(--surface)",
                        transition: "border-color .2s,background .2s", cursor: "default",
                      }}
                      onMouseEnter={e => { e.currentTarget.style.borderColor = "var(--border-strong)"; e.currentTarget.style.background = "var(--surface-hover)"; }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.background = "var(--surface)"; }}
                    >
                      <div style={{
                        width: 30, height: 30, borderRadius: 8, flexShrink: 0,
                        background: `hsl(${230 + i * 20},65%,52%)`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 12, fontWeight: 700, color: "#fff",
                      }}>
                        {name[0]}
                      </div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)" }}>{name}</div>
                        <div style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)", letterSpacing: "0.05em" }}>Contributor</div>
                      </div>
                    </div>
                  </Reveal>
                ))}
              </div>
            </div>

            <Reveal delay={100}>
              <div style={{ borderRadius: 24, overflow: "hidden", border: "1px solid var(--border)", boxShadow: "0 20px 60px rgba(0,0,0,.4)" }}>
                <img
                  src="https://vitbhopal.ac.in/wp-content/uploads/2024/03/VITB-Background.jpeg"
                  alt="VIT Bhopal campus"
                  style={{ width: "100%", height: 340, objectFit: "cover", display: "block" }}
                  loading="lazy"
                />
              </div>
            </Reveal>
          </div>
        </section>

        <Divider />

        {/* ── API ── */}
        <section id="api" style={{ padding: "80px clamp(16px,4vw,48px)", maxWidth: 1152, margin: "0 auto" }}>
          <Reveal>
            <SectionLabel>Backend contract</SectionLabel>
            <h2 style={{ fontSize: "clamp(24px,3.5vw,32px)", fontWeight: 700, letterSpacing: "-0.03em" }}>One endpoint. Clean contract.</h2>
            <p style={{ marginTop: 8, fontSize: 14, color: "var(--text-muted)", maxWidth: 500 }}>
              CinePulse uses a single JSON endpoint. Models run on the server — the frontend stays lean.
            </p>
          </Reveal>

          <Reveal delay={80}>
            <div style={{ marginTop: 36, borderRadius: 20, border: "1px solid var(--border)", overflow: "hidden" }}>
              {/* Chrome */}
              <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", background: "var(--surface)", display: "flex", alignItems: "center", gap: 7 }}>
                <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#ef4444", opacity: .6 }} />
                <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#f59e0b", opacity: .6 }} />
                <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#10b981", opacity: .6 }} />
                <span style={{ marginLeft: 10, fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>api-spec.json</span>
              </div>
              <div className="api-grid">
                <CodePanel label="Method / Endpoint">
                  <span style={{ color: "#4ade80" }}>POST</span>{"  "}
                  <span style={{ color: "var(--accent)" }}>/api/recommendations</span>
                </CodePanel>
                <CodePanel label="Request Body">
                  <span style={{ color: "#f472b6" }}>{"{"}</span><br />
                  {"  "}userId<span style={{ color: "var(--text-subtle)" }}>,</span><br />
                  {"  "}topK<span style={{ color: "var(--text-subtle)" }}>,</span><br />
                  {"  "}algorithm<br />
                  <span style={{ color: "#f472b6" }}>{"}"}</span>
                </CodePanel>
                <CodePanel label="Response Shape">
                  <span style={{ color: "var(--text-muted)" }}>Array of ranked items</span><br />
                  <span style={{ color: "var(--accent)" }}>movieId</span>
                  <span style={{ color: "var(--text-subtle)" }}>, title, genres</span><br />
                  <span style={{ color: "var(--accent)" }}>score</span>
                  <span style={{ color: "var(--text-subtle)" }}> (optional)</span>
                </CodePanel>
              </div>
            </div>
          </Reveal>
        </section>

        <Divider />

        {/* ── FAQ ── */}
        <section id="faq" style={{ padding: "80px clamp(16px,4vw,48px)", maxWidth: 1152, margin: "0 auto" }}>
          <Reveal>
            <SectionLabel>FAQ</SectionLabel>
            <h2 style={{ fontSize: "clamp(24px,3.5vw,32px)", fontWeight: 700, letterSpacing: "-0.03em" }}>Common questions.</h2>
            <p style={{ marginTop: 8, fontSize: 14, color: "var(--text-muted)" }}>Short answers when testing recommendations.</p>
          </Reveal>
          <div className="grid-2" style={{ marginTop: 36 }}>
            <FAQ delay={0}   q="Do I need to run ML code in the browser?"  a="No. The browser only calls the backend API. All models run on the server — the UI just sends a request and renders the JSON response." />
            <FAQ delay={60}  q="What is a valid userId?"                    a="Any userId that exists in your dataset from ratings. The backend handles cold-start edge cases gracefully." />
            <FAQ delay={80}  q="Which algorithm should I use first?"        a="Try all four. RL bandit is engagement-driven; sequential LSTM emphasizes recent history. Comparing them side-by-side is the point." />
            <FAQ delay={100} q="How do I test quickly?"                     a="Start with userId: 1 and topK: 10, then click Get Recommendations. Results appear in seconds." />
          </div>
        </section>

        <Divider />

        {/* ── CTA STRIP ── */}
        <section style={{ padding: "80px clamp(16px,4vw,48px)", maxWidth: 1152, margin: "0 auto" }}>
          <Reveal>
            <div style={{
              borderRadius: 24, padding: "52px 48px",
              border: "1px solid var(--accent-border)",
              background: "linear-gradient(135deg,rgba(99,102,241,.1) 0%,rgba(139,92,246,.06) 100%)",
              display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 24,
            }}>
              <div>
                <h2 style={{ fontSize: "clamp(22px,3vw,30px)", fontWeight: 700, letterSpacing: "-0.03em" }}>Ready to explore your taste?</h2>
                <p style={{ marginTop: 8, fontSize: 14, color: "var(--text-muted)", maxWidth: 400 }}>
                  Pick a model, enter your ID, and get ranked movie recommendations in seconds.
                </p>
              </div>
              <button className="btn-primary" onClick={() => navigate('/dashboard')} style={{ fontSize: 15, padding: "13px 28px" }}>
                Open Dashboard →
              </button>
            </div>
          </Reveal>
        </section>

        {/* ── FOOTER ── */}
        <footer style={{ borderTop: "1px solid var(--border)" }}>
          <div className="wrap" style={{ height: 56, display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{
                width: 26, height: 26, borderRadius: 7,
                background: "var(--accent-strong)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 10, fontWeight: 700, fontFamily: "var(--font-mono)", color: "#fff",
              }}>CP</div>
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                © {new Date().getFullYear()} CinePulse — ML powered, UX-first.
              </span>
            </div>
            <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-subtle)" }}>
              POST /api/recommendations
            </span>
          </div>
        </footer>

      </div>
    </>
  );
}