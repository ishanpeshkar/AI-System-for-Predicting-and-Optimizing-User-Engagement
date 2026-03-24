import { Navigate, Route, Routes } from 'react-router-dom'
import Landing from './pages/Landing'
import PredictPage from './pages/Recommendations'
import Dashboard from './pages/Dashboard'
import Compare from './pages/Compare'
import Metrics from './pages/Metrics'
import About from './pages/About'
import AppLayout from './layouts/AppLayout'
import './App.css'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />

      <Route element={<AppLayout />}>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/metrics" element={<Metrics />} />
        <Route path="/about" element={<About />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
