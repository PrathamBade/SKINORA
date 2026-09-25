/**
 * SKINORA — Auth Context
 *
 * Provides global authentication state (user, token) and actions
 * (login, register, logout) to all components via React Context.
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { login as apiLogin, register as apiRegister, getMe } from '../api/client.js'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(() => localStorage.getItem('skinora_token'))
  const [loading, setLoading] = useState(true)

  // On mount, restore session if token exists
  useEffect(() => {
    if (token) {
      getMe()
        .then(setUser)
        .catch(() => {
          // Token expired or invalid — clear it
          localStorage.removeItem('skinora_token')
          setToken(null)
        })
        .finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, [token])

  const login = useCallback(async (email, password) => {
    const data = await apiLogin({ email, password })
    const accessToken = data.data?.access_token
    if (!accessToken) throw new Error('No token received from server')
    localStorage.setItem('skinora_token', accessToken)
    setToken(accessToken)
    const me = await getMe()
    setUser(me)
    return me
  }, [])

  const register = useCallback(async ({ email, username, password, full_name }) => {
    const data = await apiRegister({ email, username, password, full_name })
    const accessToken = data.data?.access_token
    if (!accessToken) throw new Error('No token received from server')
    localStorage.setItem('skinora_token', accessToken)
    setToken(accessToken)
    const me = await getMe()
    setUser(me)
    return me
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('skinora_token')
    setToken(null)
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
