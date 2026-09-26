import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

export default function RegisterPage() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', username: '', full_name: '', password: '', confirm: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function handleChange(e) {
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    if (form.password !== form.confirm) {
      setError('Passwords do not match.')
      return
    }
    if (form.password.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }
    setLoading(true)
    try {
      await register({ email: form.email, username: form.username, password: form.password, full_name: form.full_name })
      navigate('/dashboard')
    } catch (err) {
      setError(err.message || 'Registration failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-[85vh] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md bg-white border border-[#D3C0A8] rounded-3xl p-8 sm:p-10 shadow-soft">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-12 h-12 rounded-2xl bg-[#A37D6C] flex items-center justify-center text-white font-bold text-xl mx-auto mb-4 shadow-sm">
            S
          </div>
          <h1 className="text-2xl font-bold text-[#3F3430] tracking-tight">Create your account</h1>
          <p className="text-[#9B8C7B] text-sm mt-1">Begin your personalized skincare journey</p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-[#644A47] mb-1.5">Full Name</label>
            <input
              type="text"
              name="full_name"
              value={form.full_name}
              onChange={handleChange}
              placeholder="Jane Smith"
              className="w-full bg-white border border-[#D3C0A8] rounded-xl px-4 py-3 text-[#3F3430] placeholder-[#9B8C7B] text-sm focus:outline-none focus:border-[#A37D6C] focus:ring-2 focus:ring-[#A37D6C]/15 transition"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[#644A47] mb-1.5">Email address</label>
            <input
              type="email"
              name="email"
              value={form.email}
              onChange={handleChange}
              required
              placeholder="you@example.com"
              className="w-full bg-white border border-[#D3C0A8] rounded-xl px-4 py-3 text-[#3F3430] placeholder-[#9B8C7B] text-sm focus:outline-none focus:border-[#A37D6C] focus:ring-2 focus:ring-[#A37D6C]/15 transition"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[#644A47] mb-1.5">Username</label>
            <input
              type="text"
              name="username"
              value={form.username}
              onChange={handleChange}
              required
              placeholder="janesmith"
              className="w-full bg-white border border-[#D3C0A8] rounded-xl px-4 py-3 text-[#3F3430] placeholder-[#9B8C7B] text-sm focus:outline-none focus:border-[#A37D6C] focus:ring-2 focus:ring-[#A37D6C]/15 transition"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[#644A47] mb-1.5">Password</label>
            <input
              type="password"
              name="password"
              value={form.password}
              onChange={handleChange}
              required
              placeholder="Min. 8 characters"
              className="w-full bg-white border border-[#D3C0A8] rounded-xl px-4 py-3 text-[#3F3430] placeholder-[#9B8C7B] text-sm focus:outline-none focus:border-[#A37D6C] focus:ring-2 focus:ring-[#A37D6C]/15 transition"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[#644A47] mb-1.5">Confirm Password</label>
            <input
              type="password"
              name="confirm"
              value={form.confirm}
              onChange={handleChange}
              required
              placeholder="••••••••"
              className="w-full bg-white border border-[#D3C0A8] rounded-xl px-4 py-3 text-[#3F3430] placeholder-[#9B8C7B] text-sm focus:outline-none focus:border-[#A37D6C] focus:ring-2 focus:ring-[#A37D6C]/15 transition"
            />
          </div>

          {error && (
            <div className="bg-[#E7B697]/25 border border-[#E7B697] text-[#644A47] rounded-xl px-4 py-3 text-sm">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-[#A37D6C] hover:bg-[#8A6454] disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition text-sm shadow-soft mt-2"
          >
            {loading ? 'Creating account…' : 'Create account'}
          </button>
        </form>

        <p className="text-center text-[#9B8C7B] text-sm mt-6">
          Already have an account?{' '}
          <Link to="/login" className="text-[#A37D6C] hover:text-[#8A6454] font-semibold underline-offset-2 hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  )
}
