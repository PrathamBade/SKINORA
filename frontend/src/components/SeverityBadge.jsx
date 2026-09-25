/**
 * Severity badge displayed on analysis & history cards.
 * Maps acne level 0-3 to a coloured pill.
 */

const SEVERITY_CONFIG = {
  'Level 0 (Clear)':    { label: 'Clear',    color: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' },
  'Level 1 (Mild)':     { label: 'Mild',     color: 'bg-yellow-500/20  text-yellow-400  border-yellow-500/30'  },
  'Level 2 (Moderate)': { label: 'Moderate', color: 'bg-orange-500/20  text-orange-400  border-orange-500/30'  },
  'Level 3 (Severe)':   { label: 'Severe',   color: 'bg-red-500/20     text-red-400     border-red-500/30'     },
}

export default function SeverityBadge({ value }) {
  const config = SEVERITY_CONFIG[value] ?? { label: value ?? '—', color: 'bg-gray-500/20 text-gray-400 border-gray-500/30' }
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${config.color}`}>
      {config.label}
    </span>
  )
}
