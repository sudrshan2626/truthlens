function parseReason(text) {
  // Converts **Bold:** text → <strong>Bold:</strong> text
  const parts = text.split(/(\*\*[^*]+\*\*)/)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i} className="text-white">{part.slice(2, -2)}</strong>
    }
    return part
  })
}

export default function ReasonsList({ reasons }) {
  return (
    <div className="space-y-3">
      {reasons.map((reason, i) => (
        <div
          key={i}
          className="flex gap-4 p-4 bg-slate-800/50 rounded-xl border border-slate-700/50
                     hover:border-slate-600 transition-colors"
        >
          <span className="flex-shrink-0 w-7 h-7 rounded-full bg-slate-700 text-slate-300
                           flex items-center justify-center text-xs font-bold mt-0.5">
            {i + 1}
          </span>
          <p className="text-slate-300 text-sm leading-relaxed">
            {parseReason(reason)}
          </p>
        </div>
      ))}
    </div>
  )
}