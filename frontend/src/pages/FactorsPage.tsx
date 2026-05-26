/**
 * Factor browser — select a factor, view IC stats and Fama-MacBeth results.
 */

import { useState, useEffect } from 'react'
import { Loader2, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { api, ICStats } from '../api'
import { useStore } from '../store'
import clsx from 'clsx'

const UNIVERSE_OPTIONS = ['SP500', 'TECH', 'FINANCE', 'HEALTHCARE', 'ENERGY', 'CONSUMER']

export default function FactorsPage() {
  const [factors, setFactors]   = useState<string[]>([])
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)

  const { universe, startDate, endDate, setUniverse,
          selectedFactor, setSelectedFactor, icStats, setIcStats } = useStore()

  useEffect(() => {
    api.factors.list().then((r) => setFactors(r.factors)).catch(console.error)
  }, [])

  async function evaluate(factor: string) {
    setLoading(true)
    setError(null)
    setSelectedFactor(factor)
    try {
      const stats = await api.factors.evaluate({
        factor_name: factor, universe, start_date: startDate, end_date: endDate,
      })
      setIcStats(stats)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-base font-semibold text-slate-100">Factor Browser</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          IC analysis and Fama-MacBeth cross-sectional regression
        </p>
      </div>

      {/* Controls */}
      <div className="flex gap-3 flex-wrap">
        <select
          value={universe}
          onChange={(e) => setUniverse(e.target.value)}
          className="bg-surface-card border border-surface-border text-sm text-slate-300
                     rounded-lg px-3 py-1.5 focus:outline-none focus:border-brand-500/60"
        >
          {UNIVERSE_OPTIONS.map((u) => (
            <option key={u} value={u}>{u}</option>
          ))}
        </select>
        <div className="flex gap-2 items-center text-xs text-slate-400">
          <input type="date" value={startDate}
            onChange={(e) => useStore.getState().setDateRange(e.target.value, endDate)}
            className="bg-surface-card border border-surface-border text-slate-300
                       rounded-lg px-3 py-1.5 focus:outline-none" />
          <span>→</span>
          <input type="date" value={endDate}
            onChange={(e) => useStore.getState().setDateRange(startDate, e.target.value)}
            className="bg-surface-card border border-surface-border text-slate-300
                       rounded-lg px-3 py-1.5 focus:outline-none" />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Factor list */}
        <div className="card space-y-1">
          <p className="stat-label mb-3">Available Factors</p>
          {factors.map((f) => (
            <button
              key={f}
              onClick={() => evaluate(f)}
              className={clsx(
                'w-full text-left px-3 py-2 rounded-lg text-sm transition-colors font-mono',
                selectedFactor === f
                  ? 'bg-brand-500/20 text-brand-500'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
              )}
            >
              {f}
            </button>
          ))}
        </div>

        {/* IC stats */}
        <div className="col-span-2 space-y-4">
          {loading && (
            <div className="card flex items-center justify-center h-40">
              <Loader2 size={20} className="animate-spin text-brand-500" />
            </div>
          )}
          {error && (
            <div className="card text-sm text-red-400">{error}</div>
          )}
          {icStats && !loading && <ICStatsCard stats={icStats} />}
        </div>
      </div>
    </div>
  )
}

function ICStatsCard({ stats }: { stats: ICStats }) {
  const ir = stats.ic_ir
  const signal =
    ir > 0.5 ? 'strong' : ir > 0.3 ? 'pass' : ir > 0.1 ? 'weak' : 'fail'

  return (
    <div className="space-y-4">
      {/* Signal quality badge */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-slate-100 font-mono">
            {stats.factor_name}
          </h2>
          <SignalBadge signal={signal} />
        </div>

        <div className="grid grid-cols-3 gap-4">
          <Stat label="IC Mean" value={fmt4(stats.ic_mean)}
                positive={stats.ic_mean > 0} />
          <Stat label="IC Std" value={fmt4(stats.ic_std)} />
          <Stat label="IC IR" value={fmt4(stats.ic_ir)}
                positive={stats.ic_ir > 0.3} />
          <Stat label="t-stat" value={fmt2(stats.t_stat)}
                positive={Math.abs(stats.t_stat) > 2} />
          <Stat label="% Positive" value={pct(stats.pct_positive)} />
          <Stat label="N periods" value={String(stats.n_periods)} />
        </div>
      </div>

      {/* Fama-MacBeth */}
      {stats.fama_macbeth_gamma != null && (
        <div className="card">
          <p className="stat-label mb-3">Fama-MacBeth Regression</p>
          <div className="grid grid-cols-2 gap-4">
            <Stat label="γ (Risk Premium)" value={fmt6(stats.fama_macbeth_gamma!)}
                  positive={(stats.fama_macbeth_gamma ?? 0) > 0} />
            <Stat label="t-stat" value={fmt2(stats.fama_macbeth_t_stat ?? 0)}
                  positive={Math.abs(stats.fama_macbeth_t_stat ?? 0) > 2} />
          </div>
        </div>
      )}
    </div>
  )
}

function SignalBadge({ signal }: { signal: string }) {
  const config = {
    strong: { text: 'Strong Signal',  cls: 'bg-green-900/40 text-green-400 border-green-800/50' },
    pass:   { text: 'Passes Hurdle',  cls: 'bg-blue-900/40  text-blue-400  border-blue-800/50'  },
    weak:   { text: 'Weak',           cls: 'bg-yellow-900/40 text-yellow-400 border-yellow-800/50' },
    fail:   { text: 'Fails Hurdle',   cls: 'bg-red-900/40   text-red-400   border-red-800/50'   },
  }[signal]!

  return (
    <span className={`px-2.5 py-0.5 rounded text-xs font-medium border ${config.cls}`}>
      {config.text}
    </span>
  )
}

function Stat({
  label, value, positive,
}: { label: string; value: string; positive?: boolean }) {
  return (
    <div>
      <p className="stat-label">{label}</p>
      <p className={clsx(
        'stat-value text-base',
        positive === true  && 'text-green-400',
        positive === false && 'text-red-400',
      )}>
        {value}
      </p>
    </div>
  )
}

const fmt4 = (n: number) => n.toFixed(4)
const fmt2 = (n: number) => n.toFixed(2)
const fmt6 = (n: number) => n.toFixed(6)
const pct  = (n: number) => `${(n * 100).toFixed(1)}%`
