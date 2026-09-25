import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'

/**
 * Top navigation bar — shows brand, nav links, and user/logout info.
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
    `text-sm font-medium px-3 py-1.5 rounded-lg transition ${
      location.pathname === path
        ? 'bg-emerald-500/20 text-emerald-400'
        : 'text-gray-400 hover:text-white hover:bg-white/5'
    }`

  return (
    <nav className="sticky top-0 z-50 border-b border-white/10 bg-gray-950/80 backdrop-blur-md">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 flex items-center justify-between h-14">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-2 group">
          <div className="w-7 h-7 rounded-lg bg-emerald-500 flex items-center justify-center text-white font-bold text-sm shadow-md shadow-emerald-500/30">
            S
          </div>
          <span className="font-bold text-white tracking-tight">SKINORA</span>
        </Link>

        {/* Nav links */}
        {user && (
          <div className="flex items-center gap-1">
            <Link to="/dashboard" className={linkClass('/dashboard')}>Dashboard</Link>
            <Link to="/upload" className={linkClass('/upload')}>Analyse</Link>
            <Link to="/history" className={linkClass('/history')}>History</Link>
          </div>
        )}

        {/* User / Auth */}
        <div className="flex items-center gap-3">
          {user ? (
            <>
              <span className="hidden sm:block text-xs text-gray-500 truncate max-w-[120px]">
                {user.email}
              </span>
              <button
                onClick={handleLogout}
                className="text-xs px-3 py-1.5 rounded-lg border border-white/10 text-gray-400 hover:text-white hover:border-white/30 transition"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <Link
                to="/login"
                className="text-xs text-gray-400 hover:text-white px-3 py-1.5 rounded-lg hover:bg-white/5 transition"
              >
                Login
              </Link>
              <Link
                to="/register"
                className="text-xs bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1.5 rounded-lg transition"
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
