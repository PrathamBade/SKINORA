/**
 * SKINORA API Client
 *
 * Centralizes all HTTP communication with the FastAPI backend.
 * The Vite dev server proxies /api/* → http://127.0.0.1:8000
 * so no hardcoded origin is needed in development.
 */

const BASE_URL = '/api/v1'

// ─── Helpers ────────────────────────────────────────────────────────────────

function getAuthHeaders() {
  const token = localStorage.getItem('skinora_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...options.headers,
    },
    ...options,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(
      err?.detail?.message || err?.detail || err?.message || res.statusText,
    )
  }

  return res.json()
}

// ─── Auth ────────────────────────────────────────────────────────────────────

export async function register({ email, username, password, full_name }) {
  return request('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, username, password, full_name }),
  })
}

export async function login({ email, password }) {
  return request('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  })
}

export async function getMe() {
  return request('/users/me')
}

// ─── Analysis ────────────────────────────────────────────────────────────────

export async function uploadImage(file) {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${BASE_URL}/analysis/upload`, {
    method: 'POST',
    headers: { ...getAuthHeaders() },
    body: form,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(
      err?.detail?.message || err?.detail || err?.message || res.statusText,
    )
  }

  return res.json()
}

export async function getAnalysis(analysisId) {
  return request(`/analysis/${analysisId}`)
}

export async function getAnalysisHistory(limit = 20, offset = 0) {
  return request(`/analysis/history?limit=${limit}&offset=${offset}`)
}

export async function getObservations(analysisId) {
  return request(`/observations/${analysisId}`)
}

// ─── Recommendations ─────────────────────────────────────────────────────────

export async function getRecommendations(acneSeverity = null) {
  const query = acneSeverity !== null ? `?acne_severity=${acneSeverity}` : ''
  return request(`/recommendations${query}`)
}

// ─── ML Status ───────────────────────────────────────────────────────────────

export async function getMlStatus() {
  const res = await fetch(`${BASE_URL}/ml/status`)
  if (!res.ok) throw new Error('Could not reach SKINORA API')
  return res.json()
}
