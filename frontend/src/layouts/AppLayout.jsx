import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useUserContext } from '../context/UserContext'
import '../styles/cinepulse.css'

const navItems = [
  {
    label: 'Dashboard',
    path: '/dashboard',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="3" y="3" width="8" height="8" rx="2" />
        <rect x="13" y="3" width="8" height="5" rx="2" />
        <rect x="13" y="10" width="8" height="11" rx="2" />
        <rect x="3" y="13" width="8" height="8" rx="2" />
      </svg>
    ),
  },
  {
    label: 'Compare',
    path: '/compare',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M7 4v16M17 4v16M4 8h6M14 8h6M4 16h6M14 16h6" />
      </svg>
    ),
  },
  {
    label: 'Predict',
    path: '/predict',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 4h14v16H5z" />
        <path d="m10 9 5 3-5 3z" />
      </svg>
    ),
  },
  {
    label: 'Metrics',
    path: '/metrics',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 19V9M10 19V5M16 19v-7M22 19v-4" />
      </svg>
    ),
  },
  {
    label: 'About',
    path: '/about',
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="8" r="3" />
        <path d="M5 20c1.5-3 4-5 7-5s5.5 2 7 5" />
      </svg>
    ),
  },
]

const pageTitles = {
  '/dashboard': 'Dashboard',
  '/compare': 'Compare Models',
  '/predict': 'Predict',
  '/metrics': 'Metrics',
  '/about': 'About CinePulse',
}

export default function AppLayout() {
  const location = useLocation()
  const { lastUserId, systemStatus } = useUserContext()

  const statusConfig = {
    ready: { dot: '#8b949e', text: 'System ready' },
    loading: { dot: '#d2a534', text: 'Running models...', pulse: true },
    active: { dot: '#3fb950', text: 'All models active' },
    error: { dot: '#f85149', text: 'Backend unreachable' },
  }

  const status = statusConfig[systemStatus] || statusConfig.ready

  return (
    <div className="cp-shell">
      <aside className="cp-sidebar">
        <div>
          <div className="cp-brand">
            <div className="cp-logo">CP</div>
            <div>
              <h1>CinePulse</h1>
              <p>Four models, one interface</p>
            </div>
          </div>

          <nav className="cp-nav">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  isActive ? 'cp-nav-link cp-nav-link-active' : 'cp-nav-link'
                }
              >
                <span className="cp-nav-icon">{item.icon}</span>
                <span>{item.label}</span>
              </NavLink>
            ))}
          </nav>
        </div>

        <p className="cp-sidebar-footnote">VIT Bhopal · Capstone 2026</p>
      </aside>

      <div className="cp-main">
        <header className="cp-topbar">
          <h2>{pageTitles[location.pathname] ?? 'CinePulse'}</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: status.dot,
                  animation: status.pulse ? 'cpStatusPulse 1.2s ease-in-out infinite' : 'none',
                }}
              />
              <span style={{ fontSize: 12, color: status.dot }}>{status.text}</span>
            </div>
            <div className="cp-user-pill">userId: {lastUserId}</div>
          </div>
        </header>

        <section className="cp-content-scroll">
          <Outlet />
        </section>
      </div>
    </div>
  )
}
