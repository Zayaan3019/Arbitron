/**
 * Backtest viewer — run walk-forward factor backtests, view equity curve.
 */

import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { api, BacktestResult } from '../api'
import { useStore } from '../store'
import clsx from 'clsx'

const FACTORS = [
  'momentum_252_21', 'momentum_126_21', 'momentum_63_5',
  'reversal_21', 'reversal_5', 'volatility_63', 'volatility_21',
  'rsi_14', 'rsi_7', 'macd', 'bollinger', 'price_to_high_252', 'acceleration_63',
]

export default function BacktestPage() {
  const [factor, setFactor]           = useState('momentum_12_1')
  const [nQuantiles, setNQuantiles]   = useState(5)
  const [txCost, setTxCost]           = useState(0.001)
  const [longOnly, setLongOnly]       = useState(false)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState<string | null>(null)

  const { universe, startDate, endDate, backtestResult, setBacktestResult } = useStore()

  async function run() {
    setLoading(true)
    setError(null)
    try {
      const result = await api.backtest.run({
        factor_name: factor, universe,
        start_date: startDate, end_date: endDate,
        n_quantiles: nQuantiles,
        transaction_cost: txCost,
        long_only: longOnly,
      })
      setBacktestResult(result)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-base font-semibold text-slate-100">Backtest</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Walk-forward factor backtest with transaction costs and purged splits
        </p>
      </div>

      {/* Config */}
      <div className="card grid grid-cols-2 gap-4">
        <div>
          <label className="stat-label block mb-1">Factor</label>
          <select value={factor} onChange={(e) => setFactor(e.target.value)}
            className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                       text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none">
            {FACTORS.map((f) => <option key={f}>{f}</option>)}
          </select>
        </div>
        <div>
          <label className="stat-label block mb-1">Quantiles</label>
          <input type="number" min={2} max={10} value={nQuantiles}
            onChange={(e) => setNQuantiles(Number(e.target.value))}
            className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                       text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none" />
        </div>
        <div>
          <label className="stat-label block mb-1">Transaction Cost</label>
          <input type="number" step={0.0005} min={0} max={0.05} value={txCost}
            onChange={(e) => setTxCost(Number(e.target.value))}
            className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                       text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none" />
        </div>
        <div className="flex items-end">
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={longOnly}
              onChange={(e) => setLongOnly(e.target.checked)}
              className="w-4 h-4 accent-brand-500" />
            <span className="text-sm text-slate-300">Long only</span>
          </label>
        </div>
        <div className="col-span-2">
          <button onClick={run} disabled={loading}
            className="w-full py-2 rounded-xl bg-brand-500 text-white text-sm
                       hover:bg-brand-600 disabled:opacity-50 transition-colors
                       flex items-center justify-center gap-2">
            {loading && <Loader2 size={14} className="animate-spin" />}
            {loading ? 'Running...' : 'Run Backtest'}
          </button>
        </div>
      </div>

      {error && <div className="card text-sm text-red-400">{error}</div>}

      {backtestResult && !loading && <BacktestResultView result={backtestResult} />}
    </div>
  )
}

function BacktestResultView({ result }: { result: BacktestResult }) {
  const m = result.metrics
  const chartData = result.dates.map((d, i) => ({
    date: d.slice(0, 7),
    cum:  +((result.cumulative_returns[i] ?? 1 - 1) * 100).toFixed(2),
    dd:   +(result.drawdown[i] * 100).toFixed(2),
  }))

  return (
    <div className="space-y-4">
      {/* Metrics grid */}
      <div className="card grid grid-cols-4 gap-4">
        <MetricTile label="CAGR"         value={pct(m.cagr)}    pos={m.cagr > 0} />
        <MetricTile label="Sharpe"       value={m.sharpe.toFixed(2)} pos={m.sharpe > 0.5} />
        <MetricTile label="Sortino"      value={m.sortino.toFixed(2)} pos={m.sortino > 0.5} />
        <MetricTile label="Calmar"       value={m.calmar.toFixed(2)} pos={m.calmar > 0.5} />
        <MetricTile label="Max DD"       value={pct(m.max_drawdown)} pos={false} />
        <MetricTile label="Win Rate"     value={pct(m.win_rate)} pos={m.win_rate > 0.5} />
        <MetricTile label="Ann Vol"      value={pct(m.ann_volatility)} />
        <MetricTile label="Skewness"     value={m.skewness.toFixed(2)} pos={m.skewness > 0} />
      </div>

      {/* Equity curve */}
      <div className="card">
        <p className="stat-label mb-3">Cumulative Return (%)</p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
            <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }}
                   tickCount={6} />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }}
                   tickFormatter={(v) => `${v}%`} />
            <Tooltip
              contentStyle={{ background: '#161b27', border: '1px solid #252d3d', fontSize: 12 }}
              formatter={(v: number) => [`${v.toFixed(2)}%`, 'Return']}
            />
            <ReferenceLine y={0} stroke="#252d3d" />
            <Line type="monotone" dataKey="cum" stroke="#4f72ff"
                  dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown */}
      <div className="card">
        <p className="stat-label mb-3">Drawdown (%)</p>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
            <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickCount={6} />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }}
                   tickFormatter={(v) => `${v}%`} />
            <Tooltip
              contentStyle={{ background: '#161b27', border: '1px solid #252d3d', fontSize: 12 }}
              formatter={(v: number) => [`${v.toFixed(2)}%`, 'Drawdown']}
            />
            <Line type="monotone" dataKey="dd" stroke="#f87171"
                  dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function MetricTile({ label, value, pos }: { label: string; value: string; pos?: boolean }) {
  return (
    <div>
      <p className="stat-label">{label}</p>
      <p className={clsx(
        'stat-value',
        pos === true  && 'text-green-400',
        pos === false && 'text-red-400',
      )}>
        {value}
      </p>
    </div>
  )
}

const pct = (n: number) => `${(n * 100).toFixed(2)}%`
