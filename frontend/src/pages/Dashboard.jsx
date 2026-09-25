import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'
import { getAnalysisHistory, getMlStatus } from '../api/client.js'
import SeverityBadge from '../components/SeverityBadge.jsx'

export default function DashboardPage() {
  const { user } = useAuth()
  const [recent, setRecent] = useState([])
  const [mlStatus, setMlStatus] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getAnalysisHistory(5, 0), getMlStatus()])
      .then(([history, ml]) => {
        setRecent(history.analyses || [])
        setMlStatus(ml)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const completed = recent.filter((a) => a.status === 'completed')
  const latestObs = completed[0]?.observations?.[0]

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 space-y-8">
      {/* Welcome */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">
            Welcome back{user?.full_name ? `, ${user.full_name.split(' ')[0]}` : ''}
          </h1>
          <p className="text-gray-500 text-sm mt-1">Here&apos;s your skin health overview</p>
        </div>
        <Link
          to="/upload"
          className="inline-flex items-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold px-5 py-2.5 rounded-xl transition shadow-md shadow-emerald-600/30"
        >
          <span>+</span> New Analysis
        </Link>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
          <p className="text-xs text-gray-500 font-medium">Total Analyses</p>
          <p className="text-3xl font-bold text-white mt-1">{loading ? '—' : recent.length}</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
          <p className="text-xs text-gray-500 font-medium">Completed</p>
          <p className="text-3xl font-bold text-emerald-400 mt-1">{loading ? '—' : completed.length}</p>
        </div>
        <div className="bg-white/5 border border-white/10 rounded-2xl p-5">
          <p className="text-xs text-gray-500 font-medium">AI Model</p>
          <div className="flex items-center gap-2 mt-1">
            {mlStatus === null ? (
              <span className="text-gray-500 text-sm">Checking…</span>
            ) : mlStatus.model_loaded ? (
              <>
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400"></span>
                <span className="text-emerald-400 font-bold text-sm">Online</span>
              </>
            ) : (
              <>
                <span className="w-2.5 h-2.5 rounded-full bg-red-400"></span>
                <span className="text-red-400 font-bold text-sm">Offline</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Latest result highlight */}
      {latestObs && (
        <div className="bg-gradient-to-br from-emerald-950/60 to-gray-900 border border-emerald-500/20 rounded-2xl p-6">
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-3">Latest Result</p>
          <div className="flex items-start justify-between gap-4">
            <div>
              <SeverityBadge value={latestObs.value} />
              <p className="text-white font-semibold mt-2">{latestObs.value}</p>
              <p className="text-gray-400 text-sm mt-1">
                Confidence: <span className="text-white font-medium">{(latestObs.confidence * 100).toFixed(1)}%</span>
              </p>
            </div>
            <Link
              to="/history"
              className="text-xs text-emerald-400 hover:text-emerald-300 font-medium whitespace-nowrap"
            >
              View all →
            </Link>
          </div>
        </div>
      )}

      {/* Recent analyses */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Recent Analyses</h2>
          <Link to="/history" className="text-xs text-emerald-400 hover:text-emerald-300">View all →</Link>
        </div>

        {loading ? (
          <div className="flex justify-center py-12">
            <div className="w-6 h-6 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : recent.length === 0 ? (
          <div className="text-center py-16 border border-dashed border-white/10 rounded-2xl">
            <p className="text-gray-500 text-sm">No analyses yet.</p>
            <Link to="/upload" className="text-emerald-400 hover:text-emerald-300 text-sm font-medium mt-2 inline-block">
              Upload your first image →
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {recent.map((analysis) => {
              const obs = analysis.observations?.[0]
              return (
                <div
                  key={analysis.analysis_id}
                  className="flex items-center justify-between bg-white/5 border border-white/10 rounded-xl px-5 py-4 hover:border-emerald-500/30 hover:bg-emerald-500/5 transition"
                >
                  <div>
                    <p className="text-sm text-white font-medium truncate max-w-[200px]">
                      {analysis.original_filename || 'Uploaded image'}
                    </p>
                    <p className="text-xs text-gray-600 mt-0.5">
                      {new Date(analysis.created_at).toLocaleDateString('en-GB', {
                        day: 'numeric', month: 'short', year: 'numeric',
                      })}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    {obs ? (
                      <SeverityBadge value={obs.value} />
                    ) : (
                      <span className="text-xs text-gray-600 capitalize">{analysis.status}</span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
