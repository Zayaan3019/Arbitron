import { useState, useEffect } from 'react'
import { Loader2, HelpCircle } from 'lucide-react'
import { api, ICStats } from '../api'
import { useStore } from '../store'
import clsx from 'clsx'

// Human-readable factor names
const FACTOR_LABELS: Record<string, { name: string; desc: string }> = {
  momentum_252_21: { name: '12-Month Momentum',      desc: 'Stocks that went up in the past year tend to keep going up' },
  momentum_126_21: { name: '6-Month Momentum',       desc: 'Same idea over 6 months — slightly shorter-term' },
  momentum_63_5:   { name: '3-Month Momentum',       desc: 'Recent 3-month price trend' },
  reversal_21:     { name: '1-Month Reversal',        desc: 'Stocks that fell last month tend to bounce back' },
  reversal_5:      { name: '1-Week Reversal',         desc: 'Very short-term mean-reversion signal' },
  volatility_63:   { name: '3-Month Volatility',     desc: 'How much the stock price moves around — lower tends to outperform' },
  volatility_21:   { name: '1-Month Volatility',     desc: 'Recent choppiness of the stock' },
  rsi_14:          { name: 'RSI (14-day)',            desc: 'Relative Strength Index — momentum oscillator over 2 weeks' },
  rsi_7:           { name: 'RSI (7-day)',             desc: 'Faster RSI — more sensitive to recent moves' },
  macd:            { name: 'MACD',                   desc: 'Moving Average Convergence/Divergence — trend change detector' },
  bollinger:       { name: 'Bollinger Band Position', desc: 'Where the stock sits within its recent price band' },
  price_to_high_252: { name: '52-Week High Proximity', desc: 'How close the stock is to its yearly high (breakout signal)' },
  acceleration_63: { name: '3-Month Acceleration',   desc: 'Whether the price trend is speeding up or slowing down' },
}

// Plain-English metric explanations
const METRIC_TIPS: Record<string, string> = {
  'IC Mean':   'Information Coefficient: how well the factor predicts next-month returns on average. Positive = right direction. Above 0.02 is good for a factor.',
  'IC Std':    'How consistent the IC is over time. Lower = more reliable. A factor with IC 0.05 every month beats one that swings wildly.',
  'IC IR':     'IC Information Ratio = IC Mean ÷ IC Std. Think of it as the Sharpe Ratio for the factor\'s predictiveness. Above 0.3 passes; above 0.5 is excellent.',
  't-stat':    'Statistical significance. Above 2.0 means there\'s less than 5% chance the result is random noise.',
  '% Positive':'The fraction of months where the factor predicted returns in the right direction. Above 50% is better than a coin flip.',
  'N Periods': 'How many months of data were used. More periods = more reliable conclusion.',
  'Risk Premium (γ)': 'Fama-MacBeth gamma: the average extra return earned per unit of factor exposure, controlling for all other factors.',
  'FM t-stat': 'Statistical significance of the risk premium. Above 2.0 means the premium is real, not random.',
}

function friendlyError(err: unknown): string {
  const msg = String(err)
  if (msg.includes('Failed to fetch') || msg.toLowerCase().includes('network')) {
    return 'Cannot reach the backend. Check VITE_API_URL in Vercel environment variables.'
  }
  return msg.replace(/^Error:\s*/, '')
}

function Tip({ text }: { text: string }) {
  return (
    <span className="relative group inline-flex items-center ml-1 cursor-help">
      <HelpCircle size={11} className="text-slate-600 group-hover:text-slate-400 transition-colors" />
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block
                       bg-slate-800 text-slate-200 text-[11px] rounded-xl px-3 py-2
                       border border-slate-700 w-56 z-20 shadow-2xl leading-relaxed">
        {text}
      </span>
    </span>
  )
}

export default function FactorsPage() {
  const [factors, setFactors] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const { universe, startDate, endDate, selectedFactor, setSelectedFactor, icStats, setIcStats } = useStore()

  useEffect(() => {
    api.factors.list().then((r) => setFactors(r.factors)).catch(console.error)
  }, [])

  async function evaluate(factor: string) {
    setLoading(true)
    setError(null)
    setSelectedFactor(factor)
    try {
      const stats = await api.factors.evaluate({ factor_name: factor, universe, start_date: startDate, end_date: endDate })
      setIcStats(stats)
    } catch (e) {
      setError(friendlyError(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-5">
      {/* Header */}
      <div className="space-y-1">
        <h1 className="text-base font-bold text-slate-100">Factor Explorer</h1>
        <p className="text-xs text-slate-400 leading-relaxed max-w-2xl">
          A <strong className="text-slate-300">factor</strong> is a measurable stock characteristic
          (like momentum or volatility) that may predict future returns.
          Click any factor below to test whether it actually works for the selected universe and time period.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Factor list */}
        <div className="card space-y-1">
          <p className="text-xs font-semibold text-slate-300 mb-3">Available Factors</p>
          {factors.length === 0 && (
            <p className="text-xs text-slate-600 italic">Loading…</p>
          )}
          {factors.map((f) => {
            const info = FACTOR_LABELS[f]
            return (
              <button
                key={f}
                onClick={() => evaluate(f)}
                className={clsx(
                  'w-full text-left px-3 py-2.5 rounded-xl text-xs transition-colors space-y-0.5',
                  selectedFactor === f
                    ? 'bg-brand-500/15 border border-brand-500/30 text-brand-300'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent',
                )}
              >
                <p className="font-medium">{info?.name ?? f}</p>
                {info && <p className="text-[10px] text-slate-600 leading-snug">{info.desc}</p>}
              </button>
            )
          })}
        </div>

        {/* Results panel */}
        <div className="col-span-2 space-y-4">
          {!selectedFactor && !loading && (
            <div className="card flex flex-col items-center justify-center py-12 text-center space-y-2">
              <p className="text-slate-400 text-sm font-medium">Select a factor on the left</p>
              <p className="text-xs text-slate-600 max-w-xs">
                We'll run an Information Coefficient analysis and Fama-MacBeth regression to tell you
                how predictive it is.
              </p>
            </div>
          )}

          {loading && (
            <div className="card flex flex-col items-center justify-center py-12 gap-3">
              <Loader2 size={20} className="animate-spin text-brand-500" />
              <p className="text-xs text-slate-500">Evaluating factor — this may take 20–30 seconds…</p>
            </div>
          )}

          {error && (
            <div className="card border border-red-800/40 bg-red-950/20">
              <p className="text-xs font-semibold text-red-400 mb-1">Error</p>
              <p className="text-xs text-red-300">{error}</p>
            </div>
          )}

          {icStats && !loading && <ICStatsCard stats={icStats} />}
        </div>
      </div>
    </div>
  )
}

function ICStatsCard({ stats }: { stats: ICStats }) {
  const ir = stats.ic_ir
  const grade =
    ir > 0.5 ? { label: 'Excellent Signal', color: 'text-green-400', bg: 'bg-green-950/30 border-green-800/40', summary: 'This factor is a strong and consistent predictor of stock returns. Highly worth using in a strategy.' }
    : ir > 0.3 ? { label: 'Good Signal', color: 'text-blue-400', bg: 'bg-blue-950/30 border-blue-800/40', summary: 'This factor shows a meaningful edge. The IC Information Ratio clears the standard hurdle of 0.3.' }
    : ir > 0.1 ? { label: 'Weak Signal', color: 'text-yellow-400', bg: 'bg-yellow-950/30 border-yellow-800/40', summary: 'Some signal exists but it\'s not reliable enough to use alone. Could work combined with others.' }
    : { label: 'No Reliable Signal', color: 'text-red-400', bg: 'bg-red-950/30 border-red-800/40', summary: 'This factor does not reliably predict returns for the selected universe and date range.' }

  const info = FACTOR_LABELS[stats.factor_name]

  return (
    <div className="space-y-4">
      {/* Verdict banner */}
      <div className={clsx('card border', grade.bg)}>
        <div className="flex items-start justify-between mb-2">
          <div>
            <p className="text-sm font-bold text-slate-100">{info?.name ?? stats.factor_name}</p>
            <p className="text-[10px] text-slate-500 font-mono mt-0.5">{stats.factor_name}</p>
          </div>
          <span className={clsx('text-xs font-bold px-2.5 py-1 rounded-lg border', grade.color, grade.bg)}>
            {grade.label}
          </span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">{grade.summary}</p>
      </div>

      {/* IC stats */}
      <div className="card">
        <p className="text-xs font-semibold text-slate-300 mb-4">
          Predictive Strength
          <span className="text-[10px] text-slate-600 font-normal ml-2">Information Coefficient analysis</span>
        </p>
        <div className="grid grid-cols-3 gap-4">
          <StatCell
            label="IC Mean" tip={METRIC_TIPS['IC Mean']}
            value={stats.ic_mean.toFixed(4)}
            positive={stats.ic_mean > 0}
          />
          <StatCell
            label="IC Std" tip={METRIC_TIPS['IC Std']}
            value={stats.ic_std.toFixed(4)}
          />
          <StatCell
            label="IC IR" tip={METRIC_TIPS['IC IR']}
            value={stats.ic_ir.toFixed(4)}
            positive={stats.ic_ir > 0.3}
          />
          <StatCell
            label="t-stat" tip={METRIC_TIPS['t-stat']}
            value={stats.t_stat.toFixed(2)}
            positive={Math.abs(stats.t_stat) > 2}
          />
          <StatCell
            label="% Positive" tip={METRIC_TIPS['% Positive']}
            value={`${(stats.pct_positive * 100).toFixed(1)}%`}
            positive={stats.pct_positive > 0.5}
          />
          <StatCell
            label="N Periods" tip={METRIC_TIPS['N Periods']}
            value={String(stats.n_periods)}
          />
        </div>
      </div>

      {/* Fama-MacBeth */}
      {stats.fama_macbeth_gamma != null && (
        <div className="card">
          <p className="text-xs font-semibold text-slate-300 mb-1">
            Risk Premium
            <span className="text-[10px] text-slate-600 font-normal ml-2">Fama-MacBeth cross-sectional regression</span>
          </p>
          <p className="text-[11px] text-slate-500 mb-4 leading-relaxed">
            After controlling for all other stocks, does this factor earn a statistically significant extra return?
          </p>
          <div className="grid grid-cols-2 gap-4">
            <StatCell
              label="Risk Premium (γ)" tip={METRIC_TIPS['Risk Premium (γ)']}
              value={stats.fama_macbeth_gamma!.toFixed(6)}
              positive={(stats.fama_macbeth_gamma ?? 0) > 0}
            />
            <StatCell
              label="FM t-stat" tip={METRIC_TIPS['FM t-stat']}
              value={(stats.fama_macbeth_t_stat ?? 0).toFixed(2)}
              positive={Math.abs(stats.fama_macbeth_t_stat ?? 0) > 2}
            />
          </div>
        </div>
      )}
    </div>
  )
}

function StatCell({ label, value, positive, tip }: { label: string; value: string; positive?: boolean; tip?: string }) {
  return (
    <div>
      <p className="text-[10px] text-slate-500 uppercase tracking-wider flex items-center">
        {label}
        {tip && <Tip text={tip} />}
      </p>
      <p className={clsx(
        'text-lg font-semibold font-mono mt-1',
        positive === true  && 'text-green-400',
        positive === false && 'text-red-400',
        positive === undefined && 'text-slate-100',
      )}>
        {value}
      </p>
    </div>
  )
}
