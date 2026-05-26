/**
 * Risk analytics dashboard — VaR, CVaR, drawdown stats, regime indicator.
 */

import { useState } from 'react'
import { Loader2, ShieldAlert, ShieldCheck, AlertTriangle } from 'lucide-react'
import { api, RiskResult } from '../api'
import { useStore } from '../store'
import clsx from 'clsx'

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
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-base font-semibold text-slate-100">Risk Analytics</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            VaR · CVaR · Drawdown duration · Market beta · Regime
          </p>
        </div>
        <button onClick={compute} disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-500
                     text-white text-sm hover:bg-brand-600 disabled:opacity-50 transition-colors">
          {loading && <Loader2 size={14} className="animate-spin" />}
          {loading ? 'Computing...' : 'Compute Risk'}
        </button>
      </div>

      {error && <div className="card text-sm text-red-400">{error}</div>}

      {riskResult && !loading && <RiskDashboard r={riskResult} />}
    </div>
  )
}

function RiskDashboard({ r }: { r: RiskResult }) {
  const regime = r.regime ?? 'Unknown'
  const isBull = regime === 'Bull'

  return (
    <div className="space-y-4">
      {/* Regime indicator */}
      <div className="card flex items-center justify-between">
        <div>
          <p className="stat-label">Market Regime</p>
          <p className="text-xl font-bold mt-1 font-mono">{regime}</p>
        </div>
        <div className={clsx(
          'w-12 h-12 rounded-full flex items-center justify-center',
          isBull ? 'bg-green-500/20' : 'bg-red-500/20'
        )}>
          {isBull
            ? <ShieldCheck size={24} className="text-green-400" />
            : <ShieldAlert size={24} className="text-red-400" />
          }
        </div>
      </div>

      {/* VaR / CVaR */}
      <div className="card">
        <p className="stat-label mb-4">Value-at-Risk (daily)</p>
        <div className="grid grid-cols-2 gap-6">
          <VarBar label="VaR 95%"   value={r.var_95}  />
          <VarBar label="CVaR 95%"  value={r.cvar_95} />
          <VarBar label="VaR 99%"   value={r.var_99}  />
          <VarBar label="CVaR 99%"  value={r.cvar_99} />
        </div>
      </div>

      {/* Tail metrics */}
      <div className="card grid grid-cols-3 gap-4">
        <Stat label="Ulcer Index"   value={r.ulcer_index.toFixed(4)} />
        <Stat label="Tail Ratio"    value={r.tail_ratio.toFixed(3)}
              pos={r.tail_ratio > 1} />
        <Stat label="Market Beta"   value={r.market_beta?.toFixed(3) ?? 'N/A'} />
      </div>

      {/* Drawdown stats */}
      <div className="card">
        <p className="stat-label mb-4">Drawdown Analytics</p>
        <div className="grid grid-cols-3 gap-4">
          <Stat label="Max DD Duration" value={`${r.max_drawdown_duration_days}d`} />
          <Stat label="Avg DD Duration" value={`${r.avg_drawdown_duration_days}d`} />
          <Stat label="# DD Periods"    value={String(r.n_drawdown_periods)} />
        </div>
      </div>

      {/* Regime advice */}
      <div className={clsx(
        'card border',
        isBull ? 'border-green-800/40 bg-green-950/20' : 'border-orange-800/40 bg-orange-950/20'
      )}>
        <div className="flex items-start gap-3">
          <AlertTriangle size={16} className={isBull ? 'text-green-400 mt-0.5' : 'text-orange-400 mt-0.5'} />
          <div>
            <p className="text-sm font-medium text-slate-200 mb-1">Factor Tilt Recommendation</p>
            <p className="text-xs text-slate-400 leading-relaxed">
              {isBull
                ? 'Bull/Low-vol regime detected. Momentum and trend-following factors tend to outperform. Consider overweighting momentum_12_1 and reducing defensive factors.'
                : 'Bear/High-vol regime detected. Reversal and low-volatility factors tend to outperform. Reduce momentum exposure, overweight defensive quality and low-vol factors.'
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function VarBar({ label, value }: { label: string; value: number }) {
  const pct = (value * 100).toFixed(2)
  const width = Math.min(value * 500, 100)   // scale for bar

  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="stat-label">{label}</span>
        <span className="text-sm font-mono text-red-400">{pct}%</span>
      </div>
      <div className="bg-surface-DEFAULT rounded-full h-2">
        <div
          className="bg-red-500 h-2 rounded-full transition-all"
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  )
}

function Stat({ label, value, pos }: { label: string; value: string; pos?: boolean }) {
  return (
    <div>
      <p className="stat-label">{label}</p>
      <p className={clsx(
        'stat-value',
        pos === true  && 'text-green-400',
        pos === false && 'text-red-400',
      )}>{value}</p>
    </div>
  )
}
