import { useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { uploadImage, getRecommendations } from '../api/client.js'
import SeverityBadge from '../components/SeverityBadge.jsx'

const SEVERITY_MAP = { 'Level 0 (Clear)': 0, 'Level 1 (Mild)': 1, 'Level 2 (Moderate)': 2, 'Level 3 (Severe)': 3 }

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [drag, setDrag] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [recommendations, setRecommendations] = useState(null)
  const [error, setError] = useState('')

  function handleFile(selectedFile) {
    if (!selectedFile) return
    const valid = ['image/jpeg', 'image/jpg', 'image/png']
    if (!valid.includes(selectedFile.type)) {
      setError('Only JPG and PNG images are supported.')
      return
    }
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('Image must be smaller than 10 MB.')
      return
    }
    setError('')
    setResult(null)
    setRecommendations(null)
    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
  }

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDrag(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  async function handleUpload() {
    if (!file) return
    setUploading(true)
    setError('')
    setResult(null)
    setRecommendations(null)
    try {
      const uploadRes = await uploadImage(file)
      setResult(uploadRes)

      // If completed with observations, fetch recommendations
      if (uploadRes.status === 'completed') {
        const obs = uploadRes.observations ?? []
        const firstObs = obs[0]
        if (firstObs) {
          const severity = SEVERITY_MAP[firstObs.value] ?? null
          const recs = await getRecommendations(severity)
          setRecommendations(recs)
        }
      }
    } catch (err) {
      setError(err.message || 'Upload failed. Please try again.')
    } finally {
      setUploading(false)
    }
  }

  function reset() {
    setFile(null)
    setPreview(null)
    setResult(null)
    setRecommendations(null)
    setError('')
  }

  const obs = result?.observations?.[0]

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Skin Analysis</h1>
        <p className="text-gray-500 text-sm mt-1">Upload a clear face photo to detect acne severity with AI</p>
      </div>

      {/* Drop zone */}
      {!result && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
          className={`relative border-2 border-dashed rounded-2xl cursor-pointer transition flex flex-col items-center justify-center gap-4 py-16 ${
            drag ? 'border-emerald-400 bg-emerald-400/5' : 'border-white/10 hover:border-emerald-500/50 hover:bg-white/5'
          }`}
        >
          <input
            id="file-input"
            type="file"
            accept="image/jpeg,image/jpg,image/png"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />

          {preview ? (
            <img
              src={preview}
              alt="preview"
              className="max-h-64 rounded-xl object-contain shadow-xl"
            />
          ) : (
            <>
              <div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center">
                <svg className="w-7 h-7 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
              </div>
              <div className="text-center">
                <p className="text-white text-sm font-medium">Drop an image here, or click to browse</p>
                <p className="text-gray-600 text-xs mt-1">JPG or PNG · Max 10 MB</p>
              </div>
            </>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Action buttons */}
      {file && !result && (
        <div className="flex gap-3">
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="flex-1 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition text-sm"
          >
            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Analysing…
              </span>
            ) : 'Analyse Image'}
          </button>
          <button
            onClick={reset}
            className="px-5 py-3 border border-white/10 text-gray-400 hover:text-white hover:border-white/30 rounded-xl text-sm transition"
          >
            Clear
          </button>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Image + prediction */}
          <div className="bg-white/5 border border-white/10 rounded-2xl overflow-hidden">
            {preview && (
              <img src={preview} alt="uploaded" className="w-full max-h-60 object-cover" />
            )}
            <div className="p-5 space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-xs text-gray-500 font-medium uppercase tracking-wider">Diagnosis</p>
                <span className="text-xs text-gray-600">{result.filename}</span>
              </div>

              {obs ? (
                <>
                  <div className="flex items-center gap-3">
                    <SeverityBadge value={obs.value} />
                    <span className="text-white font-semibold text-lg">{obs.value}</span>
                  </div>

                  {/* Confidence bar */}
                  <div>
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                      <span>Confidence</span>
                      <span className="text-white font-medium">{(obs.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 rounded-full transition-all duration-700"
                        style={{ width: `${(obs.confidence * 100).toFixed(1)}%` }}
                      />
                    </div>
                  </div>

                  {obs.description && (
                    <p className="text-gray-400 text-xs leading-relaxed border-t border-white/5 pt-3">
                      {obs.description.split('|')[0]?.trim()}
                    </p>
                  )}
                </>
              ) : (
                <p className="text-gray-500 text-sm">{result.message}</p>
              )}
            </div>
          </div>

          {/* Recommendations */}
          {recommendations && (
            <div className="bg-white/5 border border-emerald-500/20 rounded-2xl p-5 space-y-3">
              <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider">Skincare Guidance</h3>
              {recommendations.guidance?.map((tip, i) => (
                <div key={i} className="flex gap-3 text-sm">
                  <span className="text-emerald-500 font-bold shrink-0 mt-0.5">·</span>
                  <p className="text-gray-300">{tip}</p>
                </div>
              ))}
              {recommendations.disclaimer && (
                <p className="text-xs text-gray-600 italic border-t border-white/5 pt-3">
                  {recommendations.disclaimer}
                </p>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={reset}
              className="flex-1 border border-white/10 text-gray-300 hover:text-white hover:border-white/30 py-3 rounded-xl text-sm transition"
            >
              Analyse another image
            </button>
            <Link
              to="/history"
              className="flex-1 text-center bg-white/5 hover:bg-white/10 text-gray-300 hover:text-white py-3 rounded-xl text-sm transition"
            >
              View history
            </Link>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="text-center text-gray-700 text-xs">
        SKINORA is an academic project. Results are not a substitute for professional medical advice.
      </p>
    </div>
  )
}
