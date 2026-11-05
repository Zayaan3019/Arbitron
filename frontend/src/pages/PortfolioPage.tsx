import { useState } from 'react'
import { Loader2, HelpCircle } from 'lucide-react'
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts'
import { api, WeightsResult, FrontierResult } from '../api'
import { useStore } from '../store'
import { above, fmt, isNum, orZero, pct } from '../format'
import DataCaveatNotice from '../components/DataCaveatNotice'
import clsx from 'clsx'

const OPTIMIZERS = [
  {
    value: 'max_sharpe',
    label: 'Best Risk-Adjusted Return',
    desc: 'Maximise the Sharpe ratio — best return for the amount of risk you take on. Recommended for most investors.',
  },
  {
    value: 'min_variance',
    label: 'Lowest Risk',
    desc: 'Minimise portfolio volatility regardless of return. Good for conservative investors who prioritise stability.',
  },
  {
    value: 'risk_parity',
    label: 'Equal Risk Contribution',
    desc: 'Each stock contributes equally to total portfolio risk. Naturally diversified and robust to assumptions.',
  },
]

const COLORS = [
  '#4f72ff','#10b981','#f59e0b','#ef4444','#8b5cf6',
  '#06b6d4','#f97316','#84cc16','#ec4899','#14b8a6',
]

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

export default function PortfolioPage() {
  const [optimizer, setOptimizer]   = useState('max_sharpe')
  const [maxWeight, setMaxWeight]   = useState(0.3)
  const [loading, setLoading]       = useState(false)
  const [frontierLoading, setFL]    = useState(false)
  const [frontier, setFrontier]     = useState<FrontierResult | null>(null)
  const [error, setError]           = useState<string | null>(null)

  const { universe, startDate, endDate, portfolioWeights, setPortfolioWeights } = useStore()

  async function optimise() {
    setLoading(true)
    setError(null)
    try {
      const w = await api.portfolio.optimize({ universe, start_date: startDate, end_date: endDate, optimizer, max_weight: maxWeight })
      setPortfolioWeights(w)
    } catch (e) {
      setError(friendlyError(e))
    } finally {
      setLoading(false)
    }
  }

  async function loadFrontier() {
    setFL(true)
    setError(null)
    try {
      const f = await api.portfolio.frontier({ universe, start_date: startDate, end_date: endDate })
      setFrontier(f)
    } catch (e) {
      setError(friendlyError(e))
    } finally {
      setFL(false)
    }
  }

  const selectedOpt = OPTIMIZERS.find(o => o.value === optimizer)

  return (
    <div className="p-6 space-y-5">
      {/* Header */}
      <div className="space-y-1">
        <h1 className="text-base font-bold text-slate-100">Portfolio Builder</h1>
        <p className="text-xs text-slate-400 leading-relaxed max-w-2xl">
          Find the optimal mix of stocks that balances return and risk.
          Choose a strategy below, then hit <strong className="text-slate-300">Build Portfolio</strong>.
          Optionally view the <strong className="text-slate-300">Efficient Frontier</strong> to see all possible risk/return combinations.
        </p>
      </div>

      {/* Config */}
      <div className="card space-y-4">
        <p className="text-xs font-semibold text-slate-300">Choose Your Strategy</p>

        <div className="space-y-2">
          {OPTIMIZERS.map((o) => (
            <label
              key={o.value}
              className={clsx(
                'flex items-start gap-3 px-4 py-3 rounded-xl border cursor-pointer transition-colors',
                optimizer === o.value
                  ? 'border-brand-500/40 bg-brand-500/10'
                  : 'border-surface-border hover:border-slate-600 hover:bg-white/3',
              )}
            >
              <input
                type="radio" name="optimizer" value={o.value}
                checked={optimizer === o.value}
                onChange={() => setOptimizer(o.value)}
                className="mt-0.5 accent-brand-500 flex-shrink-0"
              />
              <div>
                <p className={clsx('text-sm font-semibold', optimizer === o.value ? 'text-brand-300' : 'text-slate-200')}>
                  {o.label}
                </p>
                <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{o.desc}</p>
              </div>
            </label>
          ))}
        </div>

        <div>
          <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1.5">
            Max weight per stock
            <span className="normal-case tracking-normal ml-1 text-slate-600">— prevents putting all eggs in one basket</span>
          </label>
          <div className="flex items-center gap-3">
            <input
              type="range" min={0.05} max={1} step={0.05} value={maxWeight}
              onChange={(e) => setMaxWeight(Number(e.target.value))}
              className="flex-1 accent-brand-500"
            />
            <span className="text-sm font-mono text-slate-300 w-12 text-right">
              {(maxWeight * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        <div className="flex gap-2">
          <button
            onClick={optimise} disabled={loading}
            className="flex-1 py-2.5 rounded-xl bg-brand-500 text-white text-sm font-medium
                       hover:bg-brand-600 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
          >
            {loading && <Loader2 size={14} className="animate-spin" />}
            {loading ? 'Building portfolio…' : 'Build Portfolio'}
          </button>
          <button
            onClick={loadFrontier} disabled={frontierLoading}
            className="px-4 py-2.5 rounded-xl border border-surface-border text-slate-300 text-sm
                       hover:border-brand-500/40 hover:text-slate-100 transition-colors flex items-center gap-2"
          >
            {frontierLoading && <Loader2 size={14} className="animate-spin" />}
            {frontierLoading ? 'Loading…' : 'Show Efficient Frontier'}
          </button>
        </div>
      </div>

      {error && (
        <div className="card border border-red-800/40 bg-red-950/20">
          <p className="text-xs font-semibold text-red-400 mb-1">Error</p>
          <p className="text-xs text-red-300">{error}</p>
        </div>
      )}

      {portfolioWeights && !loading && <WeightsView weights={portfolioWeights} optimizerLabel={selectedOpt?.label ?? ''} />}

      {frontier && !frontierLoading && <FrontierChart frontier={frontier} optimal={portfolioWeights} />}
    </div>
  )
}

function WeightsView({ weights: w, optimizerLabel }: { weights: WeightsResult; optimizerLabel: string }) {
  // A long-short book has negative weights; the pie only shows the long leg,
  // so filter on magnitude rather than assuming everything is positive.
  const pieData = w.tickers
    .map((t, i) => ({ name: t, value: +(w.weights[i] * 100).toFixed(2) }))
    .filter((d) => d.value > 0.5)
    .sort((a, b) => b.value - a.value)

  const shorts = w.tickers
    .map((t, i) => ({ name: t, value: +(w.weights[i] * 100).toFixed(2) }))
    .filter((d) => d.value < -0.5)
    .sort((a, b) => a.value - b.value)

  // Plain English summary
  const topHolding = pieData[0]
  const concentration = pieData.slice(0, 3).reduce((s, d) => s + d.value, 0)

  return (
    <div className="space-y-4">
      {/* Summary banner */}
      <div className="card border border-brand-500/20 bg-brand-500/5">
        <p className="text-xs font-semibold text-slate-200 mb-1">Portfolio Summary — {optimizerLabel}</p>
        <p className="text-xs text-slate-400 leading-relaxed">
          Expected <span className="text-green-400 font-medium">{pct(w.expected_return)} annual return</span> with{' '}
          <span className="text-slate-300 font-medium">{pct(w.volatility)} annual volatility</span> (Sharpe {fmt(w.sharpe)}).
          Top holding is <span className="text-slate-300 font-medium">{topHolding?.name ?? '—'}</span> at {fmt(topHolding?.value, 1)}%,
          and the top 3 stocks make up {fmt(concentration, 0)}% of the portfolio.
        </p>
        <p className="text-[10px] text-slate-600 mt-1.5">
          Estimated from {w.n_observations} complete daily observations. Weights sum to{' '}
          <span className="font-mono">{w.weight_sum.toFixed(6)}</span>.
        </p>
      </div>

      <DataCaveatNotice caveats={w.data_caveats} />

      <div className="grid grid-cols-2 gap-4">
        {/* Metrics + holdings */}
        <div className="card space-y-4">
          <p className="text-xs font-semibold text-slate-300">Portfolio Metrics</p>
          <div className="grid grid-cols-2 gap-4">
            <StatCell
              label="Expected Return"
              tip="Estimated annual return based on historical average returns of the selected stocks."
              value={pct(w.expected_return)}
              positive={above(w.expected_return, 0)}
            />
            <StatCell
              label="Annual Volatility"
              tip="How much the portfolio value will swing up and down per year. Lower = smoother ride."
              value={pct(w.volatility)}
            />
            <StatCell
              label="Sharpe Ratio"
              tip="Return earned per unit of risk. Above 1.0 is good. This is the main metric to optimise."
              value={fmt(w.sharpe, 3)}
              positive={above(w.sharpe, 0.5)}
            />
            <StatCell
              label="Diversification"
              tip="How much diversification benefit you get vs holding each stock alone. Higher = more diversified."
              value={fmt(w.diversification_ratio, 3)}
              positive={above(w.diversification_ratio, 1)}
            />
          </div>

          <div>
            <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Stock Allocations</p>
            <div className="space-y-1.5">
              {pieData.slice(0, 8).map((d, i) => (
                <div key={d.name} className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                  <span className="text-xs font-mono text-slate-300 w-14">{d.name}</span>
                  <div className="flex-1 bg-surface-DEFAULT rounded-full h-1.5">
                    <div className="h-1.5 rounded-full transition-all" style={{ width: `${d.value}%`, background: COLORS[i % COLORS.length] }} />
                  </div>
                  <span className="text-xs font-mono text-slate-400 w-12 text-right">{d.value.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>

          {shorts.length > 0 && (
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">
                Short Positions
                <span className="normal-case tracking-normal ml-1 text-slate-600">
                  — held, not silently zeroed
                </span>
              </p>
              <div className="flex flex-wrap gap-2">
                {shorts.map((d) => (
                  <span key={d.name} className="text-[11px] font-mono px-2 py-1 rounded-lg
                                                bg-red-950/30 border border-red-800/40 text-red-300">
                    {d.name} {d.value.toFixed(1)}%
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Pie chart */}
        <div className="card">
          <p className="text-xs font-semibold text-slate-300 mb-3">Weight Distribution</p>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} innerRadius={45} paddingAngle={2}>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: '#161b27', border: '1px solid #252d3d', fontSize: 12 }}
                formatter={(v: number) => [`${v.toFixed(1)}%`]}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

function FrontierChart({ frontier: f, optimal }: { frontier: FrontierResult; optimal: WeightsResult | null }) {
  const data = f.volatilities
    .map((v, i) => ({
      vol:    isNum(v) ? +(v * 100).toFixed(2) : null,
      ret:    isNum(f.returns[i]) ? +(orZero(f.returns[i]) * 100).toFixed(2) : null,
      sharpe: isNum(f.sharpes[i]) ? +orZero(f.sharpes[i]).toFixed(3) : null,
    }))
    .filter((d): d is { vol: number; ret: number; sharpe: number | null } =>
      d.vol !== null && d.ret !== null)

  const opt = optimal && isNum(optimal.volatility) && isNum(optimal.expected_return)
    ? [{
        vol: +(orZero(optimal.volatility) * 100).toFixed(2),
        ret: +(orZero(optimal.expected_return) * 100).toFixed(2),
      }]
    : []

  return (
    <div className="card">
      <p className="text-xs font-semibold text-slate-300 mb-1">Efficient Frontier</p>
      <p className="text-[10px] text-slate-500 mb-3">
        Every point is a possible portfolio. The curve shows the best possible return for each level of risk.
        {opt.length > 0 && ' The green star ★ is your optimised portfolio.'}
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
          <XAxis dataKey="vol" name="Risk (Volatility)" unit="%" type="number"
                 tick={{ fill: '#64748b', fontSize: 11 }}
                 label={{ value: 'Annual Volatility (Risk) %', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }} />
          <YAxis dataKey="ret" name="Return" unit="%" tick={{ fill: '#64748b', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#161b27', border: '1px solid #252d3d', fontSize: 11 }}
            cursor={{ strokeDasharray: '3 3' }}
          />
          <Scatter data={data} fill="#4f72ff" opacity={0.6} />
          {opt.length > 0 && <Scatter data={opt} fill="#10b981" shape="star" />}
        </ScatterChart>
      </ResponsiveContainer>
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
