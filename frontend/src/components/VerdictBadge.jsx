const config = {
  FAKE: {
    bg:   'bg-red-500/15',
    text: 'text-red-400',
    border: 'border-red-500/30',
    icon: '🔴',
    glow: 'shadow-red-500/20',
  },
  REAL: {
    bg:   'bg-green-500/15',
    text: 'text-green-400',
    border: 'border-green-500/30',
    icon: '🟢',
    glow: 'shadow-green-500/20',
  },
  UNCERTAIN: {
    bg:   'bg-yellow-500/15',
    text: 'text-yellow-400',
    border: 'border-yellow-500/30',
    icon: '🟡',
    glow: 'shadow-yellow-500/20',
  },
}

export default function VerdictBadge({ verdict }) {
  const c = config[verdict] || config.UNCERTAIN

  return (
    <div className={`
      inline-flex items-center gap-3 px-6 py-3 rounded-2xl border
      ${c.bg} ${c.border} shadow-lg ${c.glow}
    `}>
      <span className="text-2xl">{c.icon}</span>
      <div>
        <p className="text-xs text-slate-400 uppercase tracking-widest font-medium">Verdict</p>
        <p className={`text-2xl font-bold tracking-wide ${c.text}`}>{verdict}</p>
      </div>
    </div>
  )
}