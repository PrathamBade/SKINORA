/**
 * Severity badge displayed on analysis & history cards.
 * Maps acne severity (Level 0-3) to calm, natural skincare tones.
 */

const SEVERITY_CONFIG = {
  'Level 0 (Clear)':    { label: 'Clear',    color: 'bg-[#B3D1B4]/50 text-[#3F3430] border-[#B3D1B4]' },
  'Level 1 (Mild)':     { label: 'Mild',     color: 'bg-[#E7B697]/40 text-[#644A47] border-[#E7B697]' },
  'Level 2 (Moderate)': { label: 'Moderate', color: 'bg-[#F3C88A]/45 text-[#644A47] border-[#F3C88A]' },
  'Level 3 (Severe)':   { label: 'Severe',   color: 'bg-[#A37D6C] text-white border-[#8A6454]' },
}

export default function SeverityBadge({ value }) {
  const config = SEVERITY_CONFIG[value] ?? {
    label: value ?? '—',
    color: 'bg-[#F7F0E5] text-[#9B8C7B] border-[#D3C0A8]',
  }
  return (
    <span
      className={`inline-flex items-center px-3 py-0.5 rounded-full text-xs font-semibold border ${config.color} shadow-sm`}
    >
      {config.label}
    </span>
  )
}
