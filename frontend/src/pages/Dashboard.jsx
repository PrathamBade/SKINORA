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
    // 1. Fetch ML model status independently
    getMlStatus()
      .then(setMlStatus)
      .catch((err) => {
        console.error('Failed to get ML status:', err)
        setMlStatus({ model_loaded: false, error: err.message })
      })

    // 2. Fetch recent user analyses
    getAnalysisHistory(5, 0)
      .then((history) => {
        setRecent(history.analyses || [])
      })
      .catch((err) => {
        console.error('Failed to get analysis history:', err)
      })
      .finally(() => setLoading(false))
  }, [])

  const completed = recent.filter((a) => a.status === 'completed')
  const latestObs = completed[0]?.observations?.[0]

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10 space-y-8">
      {/* Welcome */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-[#3F3430] tracking-tight">
            Welcome back{user?.full_name ? `, ${user.full_name.split(' ')[0]}` : ''}
          </h1>
          <p className="text-[#9B8C7B] text-sm mt-1">Here&apos;s your skin health and analysis overview</p>
        </div>
        <Link
          to="/upload"
          className="inline-flex items-center gap-2 bg-[#A37D6C] hover:bg-[#8A6454] text-white text-sm font-semibold px-5 py-2.5 rounded-xl transition shadow-soft"
        >
          <span>+</span> New Analysis
        </Link>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        <div className="bg-white border border-[#D3C0A8] rounded-2xl p-6 shadow-soft hover:shadow-soft-md transition">
          <p className="text-xs text-[#9B8C7B] font-semibold uppercase tracking-wider">Total Analyses</p>
          <p className="text-3xl font-bold text-[#3F3430] mt-2">{loading ? '—' : recent.length}</p>
        </div>
        <div className="bg-white border border-[#D3C0A8] rounded-2xl p-6 shadow-soft hover:shadow-soft-md transition">
          <p className="text-xs text-[#9B8C7B] font-semibold uppercase tracking-wider">Completed</p>
          <p className="text-3xl font-bold text-[#A37D6C] mt-2">{loading ? '—' : completed.length}</p>
        </div>
        <div className="bg-white border border-[#D3C0A8] rounded-2xl p-6 shadow-soft hover:shadow-soft-md transition">
          <p className="text-xs text-[#9B8C7B] font-semibold uppercase tracking-wider">AI Model</p>
          <div className="flex items-center gap-2 mt-2">
            {mlStatus === null ? (
              <span className="text-[#9B8C7B] text-sm">Checking…</span>
            ) : mlStatus.model_loaded ? (
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#B3D1B4]/30 border border-[#B3D1B4]">
                <span className="w-2 h-2 rounded-full bg-[#5B8E5D] animate-pulse"></span>
                <span className="text-[#3F3430] font-semibold text-xs">Online · ResNet18</span>
              </div>
            ) : (
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#E7B697]/30 border border-[#E7B697]">
                <span className="w-2 h-2 rounded-full bg-[#C26B4A]"></span>
                <span className="text-[#644A47] font-semibold text-xs">Offline</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Latest result highlight */}
      {latestObs && (
        <div className="bg-gradient-to-br from-[#F7F0E5] via-[#FFFBF1] to-white border border-[#D3C0A8] rounded-2xl p-6 sm:p-7 shadow-soft">
          <p className="text-xs text-[#9B8C7B] font-semibold uppercase tracking-wider mb-3">Latest Assessment</p>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-3">
                <SeverityBadge value={latestObs.value} />
                <p className="text-[#3F3430] font-bold text-lg sm:text-xl">{latestObs.value}</p>
              </div>
              <p className="text-[#644A47] text-sm mt-1.5">
                Model confidence: <span className="text-[#3F3430] font-semibold">{(latestObs.confidence * 100).toFixed(1)}%</span>
              </p>
            </div>
            <Link
              to="/history"
              className="text-xs font-semibold text-[#A37D6C] hover:text-[#8A6454] bg-white border border-[#D3C0A8] px-4 py-2 rounded-xl transition shadow-sm self-start sm:self-auto"
            >
              View history →
            </Link>
          </div>
        </div>
      )}

      {/* Recent analyses */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xs font-semibold text-[#9B8C7B] uppercase tracking-wider">Recent Analyses</h2>
          <Link to="/history" className="text-xs font-medium text-[#A37D6C] hover:text-[#8A6454]">View all →</Link>
        </div>

        {loading ? (
          <div className="flex justify-center py-16">
            <div className="w-7 h-7 border-3 border-[#A37D6C] border-t-transparent rounded-full animate-spin" />
          </div>
        ) : recent.length === 0 ? (
          <div className="text-center py-16 bg-white/70 border border-dashed border-[#D3C0A8] rounded-2xl">
            <p className="text-[#9B8C7B] text-sm">No skin assessments recorded yet.</p>
            <Link to="/upload" className="text-[#A37D6C] hover:text-[#8A6454] text-sm font-semibold mt-2 inline-block">
              Upload your first photo →
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {recent.map((analysis) => {
              const obs = analysis.observations?.[0]
              const statusDisplay =
                analysis.status === 'awaiting_model'
                  ? 'Awaiting Model'
                  : analysis.status
              return (
                <div
                  key={analysis.analysis_id}
                  className="flex items-center justify-between bg-white border border-[#D3C0A8] rounded-xl px-5 py-4 hover:border-[#A37D6C] hover:bg-[#F7F0E5]/40 transition shadow-soft"
                >
                  <div>
                    <p className="text-sm text-[#3F3430] font-semibold truncate max-w-[220px]">
                      {analysis.original_filename || 'Uploaded image'}
                    </p>
                    <p className="text-xs text-[#9B8C7B] mt-0.5">
                      {new Date(analysis.created_at).toLocaleDateString('en-GB', {
                        day: 'numeric',
                        month: 'short',
                        year: 'numeric',
                      })}
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    {obs ? (
                      <SeverityBadge value={obs.value} />
                    ) : (
                      <span className="text-xs text-[#9B8C7B] px-3 py-1 rounded-full bg-[#F7F0E5] border border-[#D3C0A8]">
                        {statusDisplay}
                      </span>
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
