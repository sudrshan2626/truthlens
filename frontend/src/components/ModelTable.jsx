export default function ModelTable({ predictions }) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-700">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-800 text-slate-400 uppercase text-xs tracking-wider">
            <th className="text-left px-4 py-3">Model</th>
            <th className="text-center px-4 py-3">Prediction</th>
            <th className="text-right px-4 py-3">Confidence</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-700/50">
          {predictions.map((p, i) => (
            <tr key={i} className="bg-slate-800/40 hover:bg-slate-700/40 transition-colors">
              <td className="px-4 py-3 text-slate-200 font-medium">{p.model_name}</td>
              <td className="px-4 py-3 text-center">
                <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                  p.prediction === 'REAL' ? 'bg-green-500/20 text-green-400' :
                  p.prediction === 'FAKE' ? 'bg-red-500/20 text-red-400' :
                  'bg-yellow-500/20 text-yellow-400'}`}>
                  {p.prediction}
                </span>
              </td>
              <td className="px-4 py-3 text-right">
                <div className="flex items-center justify-end gap-2">
                  <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        p.prediction === 'REAL' ? 'bg-green-500' :
                        p.prediction === 'FAKE' ? 'bg-red-500' : 'bg-yellow-500'}`}
                      style={{ width: `${Math.round(p.confidence * 100)}%` }}
                    />
                  </div>
                  <span className="text-slate-300 font-semibold w-10 text-right">
                    {Math.round(p.confidence * 100)}%
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
