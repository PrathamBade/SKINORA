import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

/**
 * Top navigation bar — light beige & warm ivory aesthetic.
 */
export default function Navbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()

  function handleLogout() {
    logout()
    navigate('/login')
  }

  const linkClass = (path) =>
    `text-sm font-medium px-3.5 py-1.5 rounded-xl transition ${
      location.pathname === path
        ? 'bg-[#F7F0E5] text-[#3F3430] font-semibold border border-[#D3C0A8] shadow-sm'
        : 'text-[#9B8C7B] hover:text-[#3F3430] hover:bg-[#F7F0E5]/70'
    }`

  return (
    <nav className="sticky top-0 z-50 border-b border-[#D3C0A8] bg-[#FFFBF1]/90 backdrop-blur-md">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 flex items-center justify-between h-16">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-xl bg-[#A37D6C] flex items-center justify-center text-white font-bold text-sm shadow-sm transition group-hover:bg-[#8A6454]">
            S
          </div>
          <span className="font-bold text-lg text-[#3F3430] tracking-tight">SKINORA</span>
        </Link>

        {/* Nav links */}
        {user && (
          <div className="flex items-center gap-1.5">
            <Link to="/dashboard" className={linkClass('/dashboard')}>Dashboard</Link>
            <Link to="/upload" className={linkClass('/upload')}>Analyse</Link>
            <Link to="/history" className={linkClass('/history')}>History</Link>
          </div>
        )}

        {/* User / Auth */}
        <div className="flex items-center gap-3">
          {user ? (
            <>
              <span className="hidden sm:block text-xs font-medium text-[#9B8C7B] truncate max-w-[140px]">
                {user.email}
              </span>
              <button
                onClick={handleLogout}
                className="text-xs px-3.5 py-1.5 rounded-xl border border-[#D3C0A8] bg-white text-[#644A47] hover:bg-[#F7F0E5] hover:text-[#3F3430] transition font-medium shadow-sm"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link
                to="/login"
                className="text-xs font-semibold text-[#644A47] hover:text-[#3F3430] px-3.5 py-2 rounded-xl hover:bg-[#F7F0E5] transition"
              >
                Login
              </Link>
              <Link
                to="/register"
                className="text-xs bg-[#A37D6C] hover:bg-[#8A6454] text-white font-semibold px-4 py-2 rounded-xl transition shadow-sm"
              >
                Register
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  )
}
