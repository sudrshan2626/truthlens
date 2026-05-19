export default function LoadingSpinner() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16">
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 rounded-full border-4 border-slate-700" />
        <div className="absolute inset-0 rounded-full border-4 border-t-blue-500
                        animate-spin" />
      </div>
      <div className="text-center">
        <p className="text-slate-300 font-semibold">Analyzing content...</p>
        <p className="text-slate-500 text-sm mt-1">
          Running ML models + Gemini AI
        </p>
      </div>
    </div>
  )
}