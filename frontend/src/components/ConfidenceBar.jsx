export default function ConfidenceBar({ score, verdict }) {
  const pct = Math.round(score * 100)

  const color = {
    FAKE:      'bg-red-500',
    REAL:      'bg-green-500',
    UNCERTAIN: 'bg-yellow-500',
  }[verdict] || 'bg-slate-500'

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-2">
        <span className="text-slate-400 font-medium">Confidence Score</span>
        <span className="text-white font-bold">{pct}%</span>
      </div>
      <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}