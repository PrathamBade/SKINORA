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
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-10 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-[#3F3430] tracking-tight">Analysis History</h1>
          <p className="text-[#9B8C7B] text-sm mt-1">{total} total session{total !== 1 ? 's' : ''} recorded</p>
        </div>
        <Link
          to="/upload"
          className="bg-[#A37D6C] hover:bg-[#8A6454] text-white text-sm font-semibold px-4 py-2 rounded-xl transition shadow-soft"
        >
          + New
        </Link>
      </div>

      {error && (
        <div className="bg-[#E7B697]/25 border border-[#E7B697] text-[#644A47] rounded-xl px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-20">
          <div className="w-7 h-7 border-3 border-[#A37D6C] border-t-transparent rounded-full animate-spin" />
        </div>
      ) : analyses.length === 0 ? (
        <div className="text-center py-20 bg-white/70 border border-dashed border-[#D3C0A8] rounded-2xl">
          <p className="text-[#9B8C7B] text-sm">No analysis records found.</p>
          <Link to="/upload" className="text-[#A37D6C] hover:text-[#8A6454] text-sm font-semibold mt-2 inline-block">
            Upload your first photo →
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
                className="bg-white border border-[#D3C0A8] rounded-2xl px-5 py-4 hover:border-[#A37D6C] hover:bg-[#F7F0E5]/40 transition shadow-soft"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2.5 flex-wrap">
                      <p className="text-[#3F3430] text-sm font-semibold truncate">
                        {analysis.original_filename || 'Uploaded image'}
                      </p>
                      {obs && <SeverityBadge value={obs.value} />}
                      {!obs && (
                        <span className="text-xs text-[#9B8C7B] px-2.5 py-0.5 rounded-full bg-[#F7F0E5] border border-[#D3C0A8]">
                          {analysis.status === 'awaiting_model' ? 'Awaiting Model' : analysis.status}
                        </span>
                      )}
                    </div>
                    {obs && (
                      <p className="text-[#9B8C7B] text-xs mt-1.5">
                        Model Confidence: <span className="text-[#644A47] font-medium">{(obs.confidence * 100).toFixed(1)}%</span>
                      </p>
                    )}
                  </div>
                  <time className="text-xs text-[#9B8C7B] shrink-0 font-medium">
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
        <div className="flex items-center justify-center gap-3 pt-3">
          <button
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
            className="px-4 py-2 rounded-xl bg-white border border-[#D3C0A8] text-[#644A47] hover:bg-[#F7F0E5] hover:text-[#3F3430] text-sm shadow-sm disabled:opacity-40 disabled:cursor-not-allowed transition font-medium"
          >
            ← Prev
          </button>
          <span className="text-[#9B8C7B] text-sm">
            Page {page + 1} / {totalPages}
          </span>
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => p + 1)}
            className="px-4 py-2 rounded-xl bg-white border border-[#D3C0A8] text-[#644A47] hover:bg-[#F7F0E5] hover:text-[#3F3430] text-sm shadow-sm disabled:opacity-40 disabled:cursor-not-allowed transition font-medium"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}
