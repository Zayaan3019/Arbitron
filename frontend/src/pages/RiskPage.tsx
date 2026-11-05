import { useState } from 'react'
import { Loader2, ShieldCheck, ShieldAlert, HelpCircle } from 'lucide-react'
import { api, RiskResult } from '../api'
import { useStore } from '../store'
import { above, fmt, isNum, orZero, pct, type Num } from '../format'
import DataCaveatNotice from '../components/DataCaveatNotice'
import clsx from 'clsx'

const METRIC_TIPS: Record<string, string> = {
  'VaR 95%':   'Value at Risk (95%): On a typical bad day (1 in 20), you won\'t lose more than this percentage of your portfolio.',
  'CVaR 95%':  'Conditional VaR (95%): When things really go wrong (the worst 5% of days), you expect to lose about this much on average.',
  'VaR 99%':   'Like VaR 95% but for rarer events — 1 in 100 days. This is the "stress test" number.',
  'CVaR 99%':  'The average loss in the worst 1% of days. Think of this as your tail-risk exposure.',
  'Ulcer Index':'Measures how painful and prolonged your drawdowns are. Lower is better. 0 = no drawdowns ever.',
  'Tail Ratio': 'Ratio of the 95th percentile gain to the 95th percentile loss. Above 1.0 = upside is bigger than downside.',
  'Market Beta':'How closely your portfolio moves with the overall market. Beta = 1.0 means it moves exactly with the market.',
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
                       border border-slate-700 w-60 z-20 shadow-2xl leading-relaxed">
        {text}
      </span>
    </span>
  )
}

export default function RiskPage() {
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const { universe, startDate, endDate, riskResult, setRiskResult } = useStore()

  async function compute() {
    setLoading(true)
    setError(null)
    try {
      const r = await api.risk.compute({ universe, start_date: startDate, end_date: endDate })
      setRiskResult(r)
    } catch (e) {
      setError(friendlyError(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <h1 className="text-base font-bold text-slate-100">Risk Dashboard</h1>
          <p className="text-xs text-slate-400 leading-relaxed max-w-2xl">
            Understand the downside risk of a portfolio before investing.
            This page measures how much you could lose on bad days, how long drawdowns last,
            and whether the market is in a bull or bear regime.
          </p>
        </div>
        <button
          onClick={compute}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-brand-500
                     text-white text-sm font-medium hover:bg-brand-600 disabled:opacity-50 transition-colors flex-shrink-0"
        >
          {loading && <Loader2 size={14} className="animate-spin" />}
          {loading ? 'Analysing…' : 'Analyse Risk'}
        </button>
      </div>

      {!riskResult && !loading && !error && (
        <div className="card flex flex-col items-center justify-center py-12 text-center space-y-2">
          <ShieldAlert size={28} className="text-slate-600" />
          <p className="text-slate-400 text-sm font-medium">Click "Analyse Risk" to get started</p>
          <p className="text-xs text-slate-600 max-w-xs">
            We'll calculate VaR, CVaR, drawdown statistics, and the current market regime for the selected universe.
          </p>
        </div>
      )}

      {loading && (
        <div className="card flex flex-col items-center justify-center py-12 gap-3">
          <Loader2 size={20} className="animate-spin text-brand-500" />
          <p className="text-xs text-slate-500">Running risk analysis — this may take 20–30 seconds…</p>
        </div>
      )}

      {error && (
        <div className="card border border-red-800/40 bg-red-950/20">
          <p className="text-xs font-semibold text-red-400 mb-1">Error</p>
          <p className="text-xs text-red-300">{error}</p>
        </div>
      )}

      {riskResult && !loading && <RiskDashboard r={riskResult} />}
    </div>
  )
}

function RiskDashboard({ r }: { r: RiskResult }) {
  const regime = r.regime ?? 'Unknown'
  const isBull = regime === 'Bull'

  // Plain English risk level
  const riskLevel =
    !isNum(r.var_95) ? { label: 'Unknown Risk', color: 'text-slate-400', bg: 'bg-slate-800/40 border-slate-700' }
    : r.var_95 < 0.01 ? { label: 'Low Risk',    color: 'text-green-400', bg: 'bg-green-950/30 border-green-800/40' }
    : r.var_95 < 0.02 ? { label: 'Moderate Risk', color: 'text-yellow-400', bg: 'bg-yellow-950/30 border-yellow-800/40' }
    : { label: 'High Risk', color: 'text-red-400', bg: 'bg-red-950/30 border-red-800/40' }

  return (
    <div className="space-y-4">
      {/* Plain English summary */}
      <div className={clsx('card border', riskLevel.bg)}>
        <p className={clsx('text-sm font-bold mb-1', riskLevel.color)}>{riskLevel.label} Portfolio</p>
        <p className="text-xs text-slate-300 leading-relaxed">
          On a typical bad day, expect to lose up to{' '}
          <strong className="text-red-400">{pct(r.var_95, 2)}</strong> of the portfolio (VaR 95%).
          In a genuine crisis, the average bad day could be{' '}
          <strong className="text-red-400">{pct(r.cvar_95, 2)}</strong> (CVaR 95%).
          The longest drawdown period lasted <strong className="text-slate-200">{r.max_drawdown_duration_days} days</strong>,
          measured over {r.n_observations} trading days.
        </p>
      </div>

      <DataCaveatNotice caveats={r.data_caveats} />

      {/* Market Regime */}
      <div className="card flex items-center justify-between">
        <div>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Current Market Regime</p>
          <p className="text-2xl font-bold font-mono text-slate-100">{regime}</p>
          <p className="text-xs text-slate-500 mt-1">
            {isBull
              ? 'Markets are trending upward with low volatility'
              : 'Markets are choppy or trending downward'}
          </p>
          <p className="text-[10px] text-slate-600 mt-1.5 leading-relaxed max-w-md">
            {r.regime_as_of ? `As of ${r.regime_as_of}. ` : ''}
            This is the current regime only — computed from data up to that date and
            nothing after it. Historical regime labels are deliberately not shown here:
            a full-sample decoded path would label the past using the future.
          </p>
        </div>
        <div className={clsx(
          'w-14 h-14 rounded-2xl flex items-center justify-center',
          isBull ? 'bg-green-500/15 border border-green-800/40' : 'bg-red-500/15 border border-red-800/40',
        )}>
          {isBull
            ? <ShieldCheck size={26} className="text-green-400" />
            : <ShieldAlert size={26} className="text-red-400" />}
        </div>
      </div>

      {/* VaR / CVaR */}
      <div className="card">
        <p className="text-xs font-semibold text-slate-300 mb-4">
          Daily Loss Estimates
          <span className="text-[10px] text-slate-600 font-normal ml-2">How much you could lose on a bad day</span>
        </p>
        <div className="grid grid-cols-2 gap-6">
          <VarBar label="VaR 95%"  tip={METRIC_TIPS['VaR 95%']}  value={r.var_95}  />
          <VarBar label="CVaR 95%" tip={METRIC_TIPS['CVaR 95%']} value={r.cvar_95} />
          <VarBar label="VaR 99%"  tip={METRIC_TIPS['VaR 99%']}  value={r.var_99}  />
          <VarBar label="CVaR 99%" tip={METRIC_TIPS['CVaR 99%']} value={r.cvar_99} />
        </div>
      </div>

      {/* Other metrics */}
      <div className="card">
        <p className="text-xs font-semibold text-slate-300 mb-4">Additional Risk Metrics</p>
        <div className="grid grid-cols-3 gap-4">
          <StatCell label="Ulcer Index"  tip={METRIC_TIPS['Ulcer Index']}  value={fmt(r.ulcer_index, 4)} />
          <StatCell label="Tail Ratio"   tip={METRIC_TIPS['Tail Ratio']}   value={fmt(r.tail_ratio, 3)} positive={above(r.tail_ratio, 1)} />
          <StatCell label="Market Beta"  tip={METRIC_TIPS['Market Beta']}  value={fmt(r.market_beta, 3)} />
        </div>
      </div>

      {/* Drawdown stats */}
      <div className="card">
        <p className="text-xs font-semibold text-slate-300 mb-4">
          Drawdown History
          <span className="text-[10px] text-slate-600 font-normal ml-2">How long and how often losses lasted</span>
        </p>
        <div className="grid grid-cols-3 gap-4">
          <StatCell label="Longest Drawdown"   value={`${r.max_drawdown_duration_days} days`} />
          <StatCell label="Average Drawdown"   value={`${r.avg_drawdown_duration_days} days`} />
          <StatCell label="Number of Drawdowns" value={String(r.n_drawdown_periods)} />
        </div>
      </div>

      {/* Strategy advice */}
      <div className={clsx(
        'card border',
        isBull ? 'border-green-800/30 bg-green-950/10' : 'border-orange-800/30 bg-orange-950/10',
      )}>
        <p className="text-xs font-semibold text-slate-200 mb-2">
          {isBull ? '📈 Factor Strategy Recommendation (Bull Regime)' : '📉 Factor Strategy Recommendation (Bear Regime)'}
        </p>
        <p className="text-xs text-slate-400 leading-relaxed">
          {isBull
            ? 'In bull / low-volatility markets, momentum factors historically outperform. Consider favouring 12-month momentum and trend-following signals. Reduce exposure to reversal and defensive factors.'
            : 'In bear / high-volatility markets, reversal and low-volatility factors tend to outperform. Reduce momentum exposure and favour defensive, quality, and low-volatility signals.'
          }
        </p>
      </div>
    </div>
  )
}

function VarBar({ label, value, tip }: { label: string; value: Num; tip?: string }) {
  const width = Math.min(orZero(value) * 500, 100)

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider flex items-center">
          {label}
          {tip && <Tip text={tip} />}
        </span>
        <span className="text-sm font-mono font-semibold text-red-400">{pct(value, 2)}</span>
      </div>
      <div className="bg-surface-DEFAULT rounded-full h-2">
        <div className="bg-red-500 h-2 rounded-full transition-all" style={{ width: `${width}%` }} />
      </div>
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
