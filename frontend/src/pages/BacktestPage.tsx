import { useState } from 'react'
import { Loader2, HelpCircle } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { api, BacktestResult } from '../api'
import { useStore } from '../store'
import clsx from 'clsx'

const FACTORS: Array<{ value: string; label: string }> = [
  { value: 'momentum_252_21', label: '12-Month Momentum'       },
  { value: 'momentum_126_21', label: '6-Month Momentum'        },
  { value: 'momentum_63_5',   label: '3-Month Momentum'        },
  { value: 'reversal_21',     label: '1-Month Reversal'         },
  { value: 'reversal_5',      label: '1-Week Reversal'          },
  { value: 'volatility_63',   label: '3-Month Volatility'      },
  { value: 'volatility_21',   label: '1-Month Volatility'      },
  { value: 'rsi_14',          label: 'RSI (14-day)'            },
  { value: 'rsi_7',           label: 'RSI (7-day)'             },
  { value: 'macd',            label: 'MACD'                    },
  { value: 'bollinger',       label: 'Bollinger Band Position' },
  { value: 'price_to_high_252', label: '52-Week High Proximity' },
  { value: 'acceleration_63', label: '3-Month Acceleration'    },
]

const METRIC_TIPS: Record<string, string> = {
  'CAGR':     'Compound Annual Growth Rate — the average yearly return of the strategy. 10% CAGR means $10,000 grows to $16,105 in 5 years.',
  'Sharpe':   'Return per unit of risk. Above 1.0 is good; above 2.0 is excellent. Negative means the strategy loses more than risk-free cash.',
  'Sortino':  'Like Sharpe, but only penalizes downside moves — more fair for strategies that have big upside swings.',
  'Calmar':   'Annual return divided by max drawdown. Higher = better reward for the worst drop you experienced.',
  'Max DD':   'Maximum Drawdown: the worst peak-to-trough drop ever. If you had $100K at the peak, this is how low it fell.',
  'Win Rate': 'Fraction of rebalancing periods where the strategy made money. Above 50% is better than random.',
  'Ann Vol':  'Annualised Volatility: how much the strategy\'s value bounces around per year. Lower = smoother ride.',
  'Skewness': 'Positive skew means occasional large wins; negative means occasional large losses. Positive is preferable.',
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

export default function BacktestPage() {
  const [factor, setFactor]         = useState('momentum_252_21')
  const [nQuantiles, setNQuantiles] = useState(5)
  const [txCost, setTxCost]         = useState(0.001)
  const [longOnly, setLongOnly]     = useState(false)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState<string | null>(null)

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
      setError(friendlyError(e))
    } finally {
      setLoading(false)
    }
  }

  const selectedLabel = FACTORS.find(f => f.value === factor)?.label ?? factor

  return (
    <div className="p-6 space-y-5">
      {/* Header */}
      <div className="space-y-1">
        <h1 className="text-base font-bold text-slate-100">Strategy Backtester</h1>
        <p className="text-xs text-slate-400 leading-relaxed max-w-2xl">
          Pick a factor signal, choose how to trade it, and see how the strategy would have performed
          historically. The backtest splits stocks into groups (quantiles) and buys the top group vs selling the bottom.
        </p>
      </div>

      {/* Config card */}
      <div className="card space-y-4">
        <p className="text-xs font-semibold text-slate-300">Configure Your Strategy</p>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1.5">
              Factor Signal
            </label>
            <select
              value={factor}
              onChange={(e) => setFactor(e.target.value)}
              className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                         text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-brand-500/60"
            >
              {FACTORS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
            </select>
          </div>

          <div>
            <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1.5">
              Number of Groups (Quantiles)
              <span className="normal-case tracking-normal ml-1 text-slate-600">— stocks split into N buckets</span>
            </label>
            <input
              type="number" min={2} max={10} value={nQuantiles}
              onChange={(e) => setNQuantiles(Number(e.target.value))}
              className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                         text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-brand-500/60"
            />
          </div>

          <div>
            <label className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1.5">
              Transaction Cost per Trade
              <span className="normal-case tracking-normal ml-1 text-slate-600">— e.g. 0.001 = 0.1%</span>
            </label>
            <input
              type="number" step={0.0005} min={0} max={0.05} value={txCost}
              onChange={(e) => setTxCost(Number(e.target.value))}
              className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                         text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-brand-500/60"
            />
          </div>

          <div className="flex items-center">
            <label className="flex items-center gap-2.5 cursor-pointer group">
              <div className="relative">
                <input
                  type="checkbox" checked={longOnly}
                  onChange={(e) => setLongOnly(e.target.checked)}
                  className="w-4 h-4 accent-brand-500"
                />
              </div>
              <div>
                <p className="text-sm text-slate-300 font-medium">Long-only mode</p>
                <p className="text-[10px] text-slate-600">Only buy top stocks (no short selling)</p>
              </div>
            </label>
          </div>
        </div>

        <button
          onClick={run}
          disabled={loading}
          className="w-full py-2.5 rounded-xl bg-brand-500 text-white text-sm font-medium
                     hover:bg-brand-600 disabled:opacity-50 transition-colors
                     flex items-center justify-center gap-2"
        >
          {loading && <Loader2 size={14} className="animate-spin" />}
          {loading ? 'Running simulation — please wait…' : `Run Backtest: ${selectedLabel}`}
        </button>
      </div>

      {error && (
        <div className="card border border-red-800/40 bg-red-950/20">
          <p className="text-xs font-semibold text-red-400 mb-1">Error</p>
          <p className="text-xs text-red-300">{error}</p>
        </div>
      )}

      {backtestResult && !loading && <BacktestResultView result={backtestResult} />}
    </div>
  )
}

function BacktestResultView({ result }: { result: BacktestResult }) {
  const m = result.metrics

  // Strategy verdict
  const score = (m.sharpe > 1 ? 2 : m.sharpe > 0.5 ? 1 : 0)
              + (m.cagr > 0.1 ? 2 : m.cagr > 0 ? 1 : 0)
              + (m.max_drawdown > -0.2 ? 1 : 0)
  const verdict =
    score >= 4 ? { label: 'Strong Strategy',  color: 'text-green-400', bg: 'bg-green-950/30 border-green-800/40', icon: '🚀' }
    : score >= 2 ? { label: 'Decent Strategy', color: 'text-blue-400',  bg: 'bg-blue-950/30 border-blue-800/40',  icon: '✅' }
    : { label: 'Weak Strategy', color: 'text-orange-400', bg: 'bg-orange-950/30 border-orange-800/40', icon: '⚠️' }

  const chartData = result.dates.map((d, i) => ({
    date: d.slice(0, 7),
    cum:  +((result.cumulative_returns[i] - 1) * 100).toFixed(2),
    dd:   +(result.drawdown[i] * 100).toFixed(2),
  }))

  return (
    <div className="space-y-4">
      {/* Verdict */}
      <div className={clsx('card border', verdict.bg)}>
        <div className="flex items-center gap-3">
          <span className="text-2xl">{verdict.icon}</span>
          <div>
            <p className={clsx('text-sm font-bold', verdict.color)}>{verdict.label}</p>
            <p className="text-xs text-slate-400 mt-0.5">
              {`${(m.cagr * 100).toFixed(1)}% annual return · ${m.sharpe.toFixed(2)} Sharpe · ${(m.max_drawdown * 100).toFixed(1)}% worst drawdown`}
            </p>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="card">
        <p className="text-xs font-semibold text-slate-300 mb-4">Performance Metrics</p>
        <div className="grid grid-cols-4 gap-4">
          <MetricTile label="CAGR"     tip={METRIC_TIPS['CAGR']}     value={`${(m.cagr*100).toFixed(1)}%`}      pos={m.cagr > 0} />
          <MetricTile label="Sharpe"   tip={METRIC_TIPS['Sharpe']}   value={m.sharpe.toFixed(2)}                pos={m.sharpe > 0.5} />
          <MetricTile label="Sortino"  tip={METRIC_TIPS['Sortino']}  value={m.sortino.toFixed(2)}               pos={m.sortino > 0.5} />
          <MetricTile label="Calmar"   tip={METRIC_TIPS['Calmar']}   value={m.calmar.toFixed(2)}                pos={m.calmar > 0.5} />
          <MetricTile label="Max DD"   tip={METRIC_TIPS['Max DD']}   value={`${(m.max_drawdown*100).toFixed(1)}%`} pos={false} />
          <MetricTile label="Win Rate" tip={METRIC_TIPS['Win Rate']} value={`${(m.win_rate*100).toFixed(1)}%`} pos={m.win_rate > 0.5} />
          <MetricTile label="Ann Vol"  tip={METRIC_TIPS['Ann Vol']}  value={`${(m.ann_volatility*100).toFixed(1)}%`} />
          <MetricTile label="Skewness" tip={METRIC_TIPS['Skewness']} value={m.skewness.toFixed(2)}             pos={m.skewness > 0} />
        </div>
      </div>

      {/* Equity curve */}
      <div className="card">
        <p className="text-xs font-semibold text-slate-300 mb-1">Cumulative Return Over Time</p>
        <p className="text-[10px] text-slate-600 mb-3">How $1 invested in the strategy grows (or shrinks) over time</p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
            <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickCount={6} />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
            <Tooltip
              contentStyle={{ background: '#161b27', border: '1px solid #252d3d', fontSize: 12 }}
              formatter={(v: number) => [`${v.toFixed(2)}%`, 'Return']}
            />
            <ReferenceLine y={0} stroke="#475569" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="cum" stroke="#4f72ff" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Drawdown */}
      <div className="card">
        <p className="text-xs font-semibold text-slate-300 mb-1">Drawdown — Losses from Peak</p>
        <p className="text-[10px] text-slate-600 mb-3">How far the strategy fell from its highest point at each date</p>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
            <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickCount={6} />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
            <Tooltip
              contentStyle={{ background: '#161b27', border: '1px solid #252d3d', fontSize: 12 }}
              formatter={(v: number) => [`${v.toFixed(2)}%`, 'Drawdown']}
            />
            <Line type="monotone" dataKey="dd" stroke="#f87171" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function MetricTile({ label, value, pos, tip }: { label: string; value: string; pos?: boolean; tip?: string }) {
  return (
    <div>
      <p className="text-[10px] text-slate-500 uppercase tracking-wider flex items-center">
        {label}
        {tip && <Tip text={tip} />}
      </p>
      <p className={clsx(
        'text-lg font-semibold font-mono mt-1',
        pos === true  && 'text-green-400',
        pos === false && 'text-red-400',
        pos === undefined && 'text-slate-100',
      )}>
        {value}
      </p>
    </div>
  )
}
