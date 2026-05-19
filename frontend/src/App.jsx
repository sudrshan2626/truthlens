import { useState } from 'react'
import { analyzeText, analyzeURL } from './api/detection'
import VerdictBadge   from './components/VerdictBadge'
import ConfidenceBar  from './components/ConfidenceBar'
import ModelTable     from './components/ModelTable'
import ReasonsList    from './components/ReasonsList'
import LoadingSpinner from './components/LoadingSpinner'

const EXAMPLE_TEXTS = [
  "SHOCKING: Scientists confirm 5G towers are being used by governments to secretly control human behavior and thoughts.",
  "The Federal Reserve raised interest rates by 25 basis points, citing persistent inflation and strong labor market data.",
]

const EXAMPLE_URLS = [
  "https://www.bbc.com/news/articles/c9dlzy9g0y9o",
  "https://www.reuters.com/technology/artificial-intelligence/",
]

export default function App() {
  const [activeTab, setActiveTab] = useState('text')   // 'text' | 'url'
  const [text,      setText]      = useState('')
  const [url,       setUrl]       = useState('')
  const [result,    setResult]    = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [error,     setError]     = useState(null)

  const handleAnalyze = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = activeTab === 'text'
        ? await analyzeText(text)
        : await analyzeURL(url)
      setResult(data)
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Failed to connect to server. Is the backend running?'
      )
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setText('')
    setUrl('')
    setResult(null)
    setError(null)
  }

  const canAnalyze = activeTab === 'text'
    ? text.trim().length >= 10
    : url.trim().startsWith('http')

  return (
    <div className="min-h-screen bg-slate-900 text-white">

      {/* ── Header ── */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center
                            justify-center text-lg font-bold shadow-lg shadow-blue-500/30">
              T
            </div>
            <div>
              <h1 className="text-lg font-bold text-white leading-none">TruthLens</h1>
              <p className="text-xs text-slate-400">AI Misinformation Detector</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            API Online
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-10 space-y-8">

        {/* ── Hero ── */}
        <div className="text-center space-y-3">
          <h2 className="text-4xl font-extrabold bg-gradient-to-r from-blue-400
                         to-purple-400 bg-clip-text text-transparent">
            Detect Misinformation Instantly
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            Paste text or drop a news URL — powered by ML ensemble + Gemini AI.
          </p>
        </div>

        {/* ── Input Card ── */}
        <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-6 space-y-5">

          {/* Tab Switcher */}
          <div className="flex bg-slate-900/60 rounded-xl p-1 gap-1">
            {[
              { key: 'text', label: '📝  Paste Text' },
              { key: 'url',  label: '🔗  News URL'   },
            ].map(tab => (
              <button
                key={tab.key}
                onClick={() => { setActiveTab(tab.key); setResult(null); setError(null) }}
                className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-all ${
                  activeTab === tab.key
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Text Tab */}
          {activeTab === 'text' && (
            <div className="space-y-3">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste a news article, headline, or claim here..."
                rows={5}
                className="w-full bg-slate-900/60 border border-slate-700 rounded-xl
                           px-4 py-3 text-slate-200 placeholder-slate-500 text-sm
                           resize-none focus:outline-none focus:ring-2
                           focus:ring-blue-500/50 focus:border-blue-500/50 transition-all"
              />
              <div className="flex flex-wrap gap-2">
                <span className="text-xs text-slate-500 self-center">Try:</span>
                {EXAMPLE_TEXTS.map((ex, i) => (
                  <button
                    key={i}
                    onClick={() => { setText(ex); setResult(null) }}
                    className="text-xs px-3 py-1.5 rounded-lg bg-slate-700
                               hover:bg-slate-600 text-slate-300 transition-colors
                               border border-slate-600"
                  >
                    Example {i + 1}
                  </button>
                ))}
              </div>
              <p className="text-xs text-slate-600 text-right">
                {text.length} / 10,000 characters
              </p>
            </div>
          )}

          {/* URL Tab */}
          {activeTab === 'url' && (
            <div className="space-y-3">
              <div className="flex gap-2">
                <div className="flex-1 flex items-center gap-3 bg-slate-900/60
                                border border-slate-700 rounded-xl px-4 py-3
                                focus-within:ring-2 focus-within:ring-blue-500/50
                                focus-within:border-blue-500/50 transition-all">
                  <span className="text-slate-500 text-lg">🔗</span>
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://www.bbc.com/news/article..."
                    className="flex-1 bg-transparent text-slate-200 text-sm
                               placeholder-slate-500 focus:outline-none"
                  />
                  {url && (
                    <button
                      onClick={() => setUrl('')}
                      className="text-slate-500 hover:text-slate-300 text-lg"
                    >
                      ×
                    </button>
                  )}
                </div>
              </div>

              {/* Example URLs */}
              <div className="flex flex-wrap gap-2">
                <span className="text-xs text-slate-500 self-center">Try:</span>
                {EXAMPLE_URLS.map((ex, i) => (
                  <button
                    key={i}
                    onClick={() => { setUrl(ex); setResult(null) }}
                    className="text-xs px-3 py-1.5 rounded-lg bg-slate-700
                               hover:bg-slate-600 text-slate-300 transition-colors
                               border border-slate-600 max-w-xs truncate"
                  >
                    {new URL(ex).hostname}
                  </button>
                ))}
              </div>

              <p className="text-xs text-slate-500">
                ℹ️ Works with most public news sites. Paywalled articles may not extract correctly.
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleAnalyze}
              disabled={loading || !canAnalyze}
              className="flex-1 py-3 rounded-xl font-semibold text-sm
                         bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700
                         disabled:text-slate-500 transition-all
                         shadow-lg shadow-blue-500/20 disabled:shadow-none"
            >
              {loading ? 'Analyzing...' : 'Analyze Now →'}
            </button>
            {(text || url || result) && (
              <button
                onClick={handleClear}
                className="px-5 py-3 rounded-xl font-semibold text-sm
                           bg-slate-700 hover:bg-slate-600 transition-colors"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* ── Loading ── */}
        {loading && <LoadingSpinner />}

        {/* ── Error ── */}
        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl
                          text-red-400 text-sm">
            ⚠️ {error}
          </div>
        )}

        {/* ── Results ── */}
        {result && !loading && (
          <div className="space-y-6">

            {/* Article metadata (URL mode only) */}
            {result.article_title && (
              <div className="p-4 bg-slate-800/40 border border-slate-700
                              rounded-xl space-y-1">
                <p className="text-xs text-slate-500 uppercase tracking-wider">
                  Article Detected
                </p>
                <p className="text-slate-200 font-semibold">{result.article_title}</p>
                {result.article_authors?.length > 0 && (
                  <p className="text-slate-400 text-sm">
                    By {result.article_authors.join(', ')}
                  </p>
                )}
              </div>
            )}

            {/* Verdict + Confidence */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-6 space-y-5">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <VerdictBadge verdict={result.verdict} />
                <span className="text-xs text-slate-500 bg-slate-700/50
                                 px-3 py-1 rounded-full">
                  ⏱ {result.processing_time_seconds}s
                </span>
              </div>
              <ConfidenceBar
                score={result.confidence_score}
                verdict={result.verdict}
              />
            </div>

            {/* Model Predictions */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-6 space-y-4">
              <h3 className="font-semibold text-slate-200 flex items-center gap-2">
                <span className="text-blue-400">⚡</span> Model Predictions
              </h3>
              <ModelTable predictions={result.model_predictions} />
            </div>

            {/* AI Reasons */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-2xl p-6 space-y-4">
              <h3 className="font-semibold text-slate-200 flex items-center gap-2">
                <span className="text-purple-400">🤖</span> AI Analysis
                <span className="text-xs text-slate-500 font-normal ml-1">
                  powered by Gemini
                </span>
              </h3>
              <ReasonsList reasons={result.reasons} />
            </div>

          </div>
        )}

      </main>

      {/* ── Footer ── */}
      <footer className="text-center py-8 text-slate-600 text-xs
                         border-t border-slate-800 mt-10">
        TruthLens • ML Ensemble + Gemini AI • Built for AI/ML Portfolio
      </footer>

    </div>
  )
}