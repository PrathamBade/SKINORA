import { useState, useCallback, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { uploadImage, getRecommendations, getAnalysis, getMlStatus } from '../api/client.js'
import SeverityBadge from '../components/SeverityBadge.jsx'

const SEVERITY_MAP = {
  'Level 0 (Clear)': 0,
  'Level 1 (Mild)': 1,
  'Level 2 (Moderate)': 2,
  'Level 3 (Severe)': 3,
}

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [drag, setDrag] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [result, setResult] = useState(null)
  const [recommendations, setRecommendations] = useState(null)
  const [error, setError] = useState('')
  const [mlStatus, setMlStatus] = useState(null)

  useEffect(() => {
    getMlStatus()
      .then(setMlStatus)
      .catch((err) => {
        console.error('Failed to get ML status:', err)
        setMlStatus({ model_loaded: false, error: err.message })
      })
  }, [])

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

      // Fallback: If uploadRes did not carry observations directly, query analysis by id
      let fullAnalysis = uploadRes
      if (
        uploadRes.status === 'completed' &&
        (!uploadRes.observations || uploadRes.observations.length === 0)
      ) {
        try {
          fullAnalysis = await getAnalysis(uploadRes.analysis_id)
        } catch (fetchErr) {
          console.warn('Could not fetch detailed analysis:', fetchErr)
        }
      }

      setResult(fullAnalysis)

      // Fetch recommendations tailored to severity level
      const obsList = fullAnalysis.observations ?? []
      const firstObs = obsList[0]
      if (firstObs) {
        const severity = SEVERITY_MAP[firstObs.value] ?? null
        const recs = await getRecommendations(severity)
        setRecommendations(recs)
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
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-10 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-[#3F3430] tracking-tight">Skin Analysis</h1>
          <p className="text-[#9B8C7B] text-sm mt-1">
            Upload a clear face photo to assess acne severity with PyTorch ResNet18
          </p>
        </div>

        {/* Model status pill */}
        {mlStatus && (
          <div className="flex items-center gap-2 self-start sm:self-auto px-3.5 py-1.5 rounded-full bg-white border border-[#D3C0A8] text-xs shadow-sm">
            <span
              className={`w-2 h-2 rounded-full ${
                mlStatus.model_loaded ? 'bg-[#5B8E5D] animate-pulse' : 'bg-[#C26B4A]'
              }`}
            />
            <span className={mlStatus.model_loaded ? 'text-[#3F3430] font-semibold' : 'text-[#644A47] font-semibold'}>
              {mlStatus.model_loaded ? 'ResNet18 Ready' : 'Model Offline'}
            </span>
          </div>
        )}
      </div>

      {/* Model offline warning banner */}
      {mlStatus && !mlStatus.model_loaded && (
        <div className="bg-[#E7B697]/25 border border-[#E7B697] text-[#644A47] rounded-2xl px-5 py-3.5 text-sm flex items-center justify-between">
          <span>AI model is currently offline. Image uploads will receive status <code>awaiting_model</code>.</span>
          <button
            onClick={() => getMlStatus().then(setMlStatus)}
            className="text-xs font-semibold px-3 py-1 bg-[#A37D6C] text-white hover:bg-[#8A6454] rounded-lg transition shadow-xs"
          >
            Retry
          </button>
        </div>
      )}

      {/* Drop zone */}
      {!result && (
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setDrag(true)
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
          className={`relative border-2 border-dashed rounded-3xl cursor-pointer transition flex flex-col items-center justify-center gap-4 py-16 px-6 ${
            drag
              ? 'border-[#A37D6C] bg-[#E7B697]/20'
              : 'border-[#D3C0A8] bg-[#F7F0E5]/60 hover:bg-[#F7F0E5] hover:border-[#A37D6C]'
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
              className="max-h-72 rounded-2xl object-contain shadow-soft border border-[#D3C0A8]"
            />
          ) : (
            <>
              <div className="w-16 h-16 rounded-2xl bg-white border border-[#D3C0A8] flex items-center justify-center shadow-soft">
                <svg
                  className="w-8 h-8 text-[#A37D6C]"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.75}
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                  />
                </svg>
              </div>
              <div className="text-center">
                <p className="text-[#3F3430] text-sm font-semibold">Drop an image here, or click to browse</p>
                <p className="text-[#9B8C7B] text-xs mt-1">High resolution facial JPG or PNG · Max 10 MB</p>
              </div>
            </>
          )}
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="bg-[#E7B697]/25 border border-[#E7B697] text-[#644A47] rounded-xl px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Action buttons */}
      {file && !result && (
        <div className="flex gap-3">
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="flex-1 bg-[#A37D6C] hover:bg-[#8A6454] disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold py-3.5 rounded-xl transition text-sm shadow-soft"
          >
            {uploading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Analysing with ResNet18…
              </span>
            ) : (
              'Analyse Image'
            )}
          </button>
          <button
            onClick={reset}
            className="px-6 py-3.5 border border-[#D3C0A8] bg-white text-[#644A47] hover:bg-[#F7F0E5] hover:text-[#3F3430] rounded-xl text-sm font-medium transition shadow-sm"
          >
            Clear
          </button>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-5">
          <div className="bg-white border border-[#D3C0A8] rounded-3xl overflow-hidden shadow-soft">
            {preview && (
              <img src={preview} alt="uploaded" className="w-full max-h-64 object-cover" />
            )}
            <div className="p-6 space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-xs text-[#9B8C7B] font-semibold uppercase tracking-wider">Assessment Diagnosis</p>
                <span className="text-xs text-[#9B8C7B] font-mono bg-[#F7F0E5] border border-[#D3C0A8]/60 px-2.5 py-0.5 rounded-md">
                  {result.filename}
                </span>
              </div>

              {obs ? (
                <>
                  <div className="flex items-center gap-3">
                    <SeverityBadge value={obs.value} />
                    <span className="text-[#3F3430] font-bold text-xl">{obs.value}</span>
                  </div>

                  {/* Confidence bar */}
                  <div>
                    <div className="flex items-center justify-between text-xs text-[#9B8C7B] font-medium mb-1.5">
                      <span>Model Confidence</span>
                      <span className="text-[#3F3430] font-bold">
                        {(obs.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2.5 bg-[#F7F0E5] border border-[#D3C0A8]/60 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#A37D6C] rounded-full transition-all duration-700"
                        style={{ width: `${(obs.confidence * 100).toFixed(1)}%` }}
                      />
                    </div>
                  </div>

                  {obs.description && (
                    <p className="text-[#644A47] text-xs leading-relaxed border-t border-[#F7F0E5] pt-3.5">
                      {obs.description.split('|')[0]?.trim()}
                    </p>
                  )}
                </>
              ) : (
                <div className="bg-[#F7F0E5] border border-[#D3C0A8] rounded-2xl p-4 text-[#644A47] text-xs">
                  <p className="font-semibold text-[#3F3430]">{result.message}</p>
                  <p className="text-[#9B8C7B] mt-1">
                    Status: <code className="text-[#644A47] font-semibold">{result.status}</code>
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Recommendations */}
          {recommendations && (
            <div className="bg-[#F7F0E5]/90 border border-[#D3C0A8] rounded-3xl p-6 space-y-4 shadow-soft">
              <h3 className="text-sm font-bold text-[#A37D6C] uppercase tracking-wider">
                Tailored Skincare Guidance
              </h3>
              <div className="space-y-2.5">
                {recommendations.guidance?.map((tip, i) => (
                  <div key={i} className="bg-white/80 border border-[#D3C0A8]/60 rounded-xl p-3.5 flex gap-3 text-sm shadow-xs">
                    <span className="text-[#A37D6C] font-bold shrink-0 mt-0.5">•</span>
                    <p className="text-[#644A47] leading-relaxed">{tip}</p>
                  </div>
                ))}
              </div>
              {recommendations.disclaimer && (
                <p className="text-xs text-[#9B8C7B] italic border-t border-[#D3C0A8]/60 pt-3.5">
                  {recommendations.disclaimer}
                </p>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={reset}
              className="flex-1 border border-[#D3C0A8] bg-white text-[#644A47] hover:bg-[#F7F0E5] hover:text-[#3F3430] py-3.5 rounded-xl text-sm font-semibold transition shadow-sm"
            >
              Analyse another photo
            </button>
            <Link
              to="/history"
              className="flex-1 text-center bg-[#F7F0E5] border border-[#D3C0A8] text-[#644A47] hover:bg-[#E7B697]/25 hover:text-[#3F3430] py-3.5 rounded-xl text-sm font-semibold transition shadow-sm"
            >
              View history
            </Link>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="text-center text-[#9B8C7B] text-xs">
        SKINORA is an academic wellness project. Results are not a substitute for professional clinical advice.
      </p>
    </div>
  )
}
