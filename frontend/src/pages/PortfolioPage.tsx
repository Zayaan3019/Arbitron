/**
 * Portfolio optimization page — weights pie chart, efficient frontier.
 */

import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from 'recharts'
import { api, WeightsResult, FrontierResult } from '../api'
import { useStore } from '../store'
import clsx from 'clsx'

const OPTIMIZERS = [
  { value: 'max_sharpe',    label: 'Max Sharpe'   },
  { value: 'min_variance',  label: 'Min Variance' },
  { value: 'risk_parity',   label: 'Risk Parity'  },
]

const COLORS = [
  '#4f72ff','#10b981','#f59e0b','#ef4444','#8b5cf6',
  '#06b6d4','#f97316','#84cc16','#ec4899','#14b8a6',
]

export default function PortfolioPage() {
  const [optimizer, setOptimizer]     = useState('max_sharpe')
  const [maxWeight, setMaxWeight]     = useState(0.3)
  const [loading, setLoading]         = useState(false)
  const [frontierLoading, setFL]      = useState(false)
  const [frontier, setFrontier]       = useState<FrontierResult | null>(null)
  const [error, setError]             = useState<string | null>(null)

  const { universe, startDate, endDate, portfolioWeights, setPortfolioWeights } = useStore()

  async function optimise() {
    setLoading(true)
    setError(null)
    try {
      const w = await api.portfolio.optimize({
        universe, start_date: startDate, end_date: endDate,
        optimizer, max_weight: maxWeight,
      })
      setPortfolioWeights(w)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function loadFrontier() {
    setFL(true)
    try {
      const f = await api.portfolio.frontier({ universe, start_date: startDate, end_date: endDate })
      setFrontier(f)
    } catch (e) {
      setError(String(e))
    } finally {
      setFL(false)
    }
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-base font-semibold text-slate-100">Portfolio Optimization</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Ledoit-Wolf covariance · MVO · Risk Parity · Efficient Frontier
        </p>
      </div>

      {/* Config */}
      <div className="card grid grid-cols-3 gap-4">
        <div>
          <label className="stat-label block mb-1">Optimizer</label>
          <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}
            className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                       text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none">
            {OPTIMIZERS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="stat-label block mb-1">Max Weight</label>
          <input type="number" step={0.05} min={0.05} max={1} value={maxWeight}
            onChange={(e) => setMaxWeight(Number(e.target.value))}
            className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                       text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none" />
        </div>
        <div className="flex items-end gap-2">
          <button onClick={optimise} disabled={loading}
            className="flex-1 py-1.5 rounded-xl bg-brand-500 text-white text-sm
                       hover:bg-brand-600 disabled:opacity-50 transition-colors
                       flex items-center justify-center gap-2">
            {loading && <Loader2 size={14} className="animate-spin" />}
            Optimise
          </button>
          <button onClick={loadFrontier} disabled={frontierLoading}
            className="flex-1 py-1.5 rounded-xl border border-surface-border text-slate-300
                       text-sm hover:border-brand-500/50 hover:text-slate-100 transition-colors
                       flex items-center justify-center gap-2">
            {frontierLoading && <Loader2 size={14} className="animate-spin" />}
            Frontier
          </button>
        </div>
      </div>

      {error && <div className="card text-sm text-red-400">{error}</div>}

      {portfolioWeights && !loading && (
        <WeightsView weights={portfolioWeights} />
      )}

      {frontier && !frontierLoading && (
        <FrontierChart frontier={frontier} optimal={portfolioWeights} />
      )}
    </div>
  )
}

function WeightsView({ weights: w }: { weights: WeightsResult }) {
  const pieData = w.tickers
    .map((t, i) => ({ name: t, value: +(w.weights[i] * 100).toFixed(2) }))
    .filter((d) => d.value > 0.5)
    .sort((a, b) => b.value - a.value)

  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Performance */}
      <div className="card space-y-4">
        <p className="stat-label">Portfolio Metrics</p>
        <div className="grid grid-cols-2 gap-4">
          <Stat label="Expected Return" value={pct(w.expected_return)} pos={true} />
          <Stat label="Volatility" value={pct(w.volatility)} />
          <Stat label="Sharpe Ratio" value={w.sharpe.toFixed(3)} pos={w.sharpe > 0.5} />
          <Stat label="Diversif. Ratio" value={w.diversification_ratio.toFixed(3)} pos={true} />
        </div>

        {/* Top weights table */}
        <div>
          <p className="stat-label mb-2">Top Holdings</p>
          <div className="space-y-1">
            {pieData.slice(0, 8).map((d, i) => (
              <div key={d.name} className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                <span className="text-xs font-mono text-slate-300 w-14">{d.name}</span>
                <div className="flex-1 bg-surface-DEFAULT rounded-full h-1.5">
                  <div
                    className="h-1.5 rounded-full"
                    style={{ width: `${d.value}%`, background: COLORS[i % COLORS.length] }}
                  />
                </div>
                <span className="text-xs font-mono text-slate-400 w-12 text-right">
                  {d.value.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Pie chart */}
      <div className="card">
        <p className="stat-label mb-3">Weight Distribution</p>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name"
                 cx="50%" cy="50%" outerRadius={80} innerRadius={40}>
              {pieData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ background: '#161b27', border: '1px solid #252d3d', fontSize: 11 }}
              formatter={(v: number) => [`${v.toFixed(1)}%`]}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function FrontierChart({ frontier: f, optimal }: {
  frontier: FrontierResult
  optimal: WeightsResult | null
}) {
  const data = f.volatilities.map((v, i) => ({
    vol:    +(v * 100).toFixed(2),
    ret:    +(f.returns[i] * 100).toFixed(2),
    sharpe: +f.sharpes[i].toFixed(3),
  }))

  const opt = optimal
    ? [{ vol: +(optimal.volatility * 100).toFixed(2), ret: +(optimal.expected_return * 100).toFixed(2) }]
    : []

  return (
    <div className="card">
      <p className="stat-label mb-3">Efficient Frontier</p>
      <ResponsiveContainer width="100%" height={250}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
          <XAxis dataKey="vol" name="Volatility" unit="%" type="number"
                 tick={{ fill: '#64748b', fontSize: 11 }} label={{
                   value: 'Volatility (%)', position: 'insideBottom', offset: -5,
                   fill: '#64748b', fontSize: 11
                 }} />
          <YAxis dataKey="ret" name="Return" unit="%"
                 tick={{ fill: '#64748b', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#161b27', border: '1px solid #252d3d', fontSize: 11 }}
            cursor={{ strokeDasharray: '3 3' }}
          />
          <Scatter data={data} fill="#4f72ff" opacity={0.7} />
          {opt.length > 0 && (
            <Scatter data={opt} fill="#10b981" shape="star" />
          )}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

function Stat({ label, value, pos }: { label: string; value: string; pos?: boolean }) {
  return (
    <div>
      <p className="stat-label">{label}</p>
      <p className={clsx(
        'stat-value text-base',
        pos === true  && 'text-green-400',
        pos === false && 'text-red-400',
      )}>{value}</p>
    </div>
  )
}

const pct = (n: number) => `${(n * 100).toFixed(2)}%`
