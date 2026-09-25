import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getAnalysisHistory } from '../api/client.js'
import SeverityBadge from '../components/SeverityBadge.jsx'

const PAGE_SIZE = 10

export default function HistoryPage() {
  const [analyses, setAnalyses] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    setLoading(true)
    getAnalysisHistory(PAGE_SIZE, page * PAGE_SIZE)
      .then((data) => {
        setAnalyses(data.analyses || [])
        setTotal(data.total || 0)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [page])

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Analysis History</h1>
          <p className="text-gray-500 text-sm mt-1">{total} total session{total !== 1 ? 's' : ''}</p>
        </div>
        <Link
          to="/upload"
          className="bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold px-4 py-2 rounded-xl transition"
        >
          + New
        </Link>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-16">
          <div className="w-7 h-7 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : analyses.length === 0 ? (
        <div className="text-center py-20 border border-dashed border-white/10 rounded-2xl">
          <p className="text-gray-500 text-sm">No analyses found.</p>
          <Link to="/upload" className="text-emerald-400 hover:text-emerald-300 text-sm font-medium mt-2 inline-block">
            Upload your first image →
          </Link>
        </div>
      ) : (
        <div className="space-y-3">
          {analyses.map((analysis) => {
            const obs = analysis.observations?.[0]
            const date = new Date(analysis.created_at)
            return (
              <div
                key={analysis.analysis_id}
                className="bg-white/5 border border-white/10 rounded-2xl px-5 py-4 hover:border-emerald-500/30 hover:bg-emerald-500/5 transition"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="text-white text-sm font-medium truncate">
                        {analysis.original_filename || 'Uploaded image'}
                      </p>
                      {obs && <SeverityBadge value={obs.value} />}
                      {!obs && (
                        <span className="text-xs text-gray-600 capitalize border border-white/10 px-2 py-0.5 rounded-full">
                          {analysis.status}
                        </span>
                      )}
                    </div>
                    {obs && (
                      <p className="text-gray-500 text-xs mt-1">
                        Confidence: {(obs.confidence * 100).toFixed(1)}%
                      </p>
                    )}
                  </div>
                  <time className="text-xs text-gray-600 shrink-0">
                    {date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}
                  </time>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
            className="px-4 py-2 rounded-lg border border-white/10 text-gray-400 hover:text-white hover:border-white/30 text-sm disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            ← Prev
          </button>
          <span className="text-gray-600 text-sm">
            Page {page + 1} / {totalPages}
          </span>
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => p + 1)}
            className="px-4 py-2 rounded-lg border border-white/10 text-gray-400 hover:text-white hover:border-white/30 text-sm disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}
