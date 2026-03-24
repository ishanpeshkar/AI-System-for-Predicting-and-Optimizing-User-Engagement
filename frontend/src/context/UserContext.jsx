import { createContext, useContext, useMemo, useState } from 'react'

const UserContext = createContext(null)

export function UserProvider({ children }) {
  const [lastUserId, setLastUserId] = useState(() => {
    const saved = window.localStorage.getItem('cinepulse:lastUserId')
    return saved ? Number(saved) : 1
  })
  const [systemStatus, setSystemStatus] = useState('ready') // 'ready' | 'loading' | 'active' | 'error'

  function updateLastUserId(nextUserId) {
    const normalized = Number(nextUserId) || 1
    setLastUserId(normalized)
    window.localStorage.setItem('cinepulse:lastUserId', String(normalized))
  }

  const value = useMemo(
    () => ({ lastUserId, setLastUserId: updateLastUserId, systemStatus, setSystemStatus }),
    [lastUserId, systemStatus],
  )

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>
}

export function useUserContext() {
  const context = useContext(UserContext)
  if (!context) {
    throw new Error('useUserContext must be used inside UserProvider')
  }
  return context
}
