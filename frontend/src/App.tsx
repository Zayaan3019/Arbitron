import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { Bot, BarChart3, TrendingUp, ShieldAlert, Cpu, Wifi, WifiOff } from 'lucide-react'
import AgentPage    from './pages/AgentPage'
import FactorsPage  from './pages/FactorsPage'
import BacktestPage from './pages/BacktestPage'
import PortfolioPage from './pages/PortfolioPage'
import RiskPage     from './pages/RiskPage'
import { useStore } from './store'
import clsx from 'clsx'

const NAV = [
  { to: '/',          icon: Bot,        label: 'AI Agent',    desc: 'Chat to research stocks' },
  { to: '/factors',   icon: BarChart3,  label: 'Factors',     desc: 'Test what predicts returns' },
  { to: '/backtest',  icon: TrendingUp, label: 'Backtest',    desc: 'Replay a strategy in history' },
  { to: '/portfolio', icon: Cpu,        label: 'Portfolio',   desc: 'Find your optimal mix' },
  { to: '/risk',      icon: ShieldAlert,label: 'Risk',        desc: 'Measure your downside' },
]

const UNIVERSES = [
  { value: 'SP500',      label: 'S&P 500 (default)'   },
  { value: 'Technology', label: 'Technology'           },
  { value: 'Healthcare', label: 'Healthcare'           },
  { value: 'Financials', label: 'Financials'           },
  { value: 'Energy',     label: 'Energy'               },
  { value: 'ConsumerDisc', label: 'Consumer'           },
]

function BackendStatus() {
  const [status, setStatus] = useState<'checking' | 'warming' | 'ok' | 'error'>('checking')

  useEffect(() => {
    const base = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
    let cancelled = false

    // After 6s still checking → show "warming up" (Render free-tier cold start takes ~50s)
    const warmTimer = setTimeout(() => {
      if (!cancelled) setStatus('warming')
    }, 6000)

    const attempt = (retries: number) => {
      const controller = new AbortController()
      const t = setTimeout(() => controller.abort(), 55000)

      fetch(`${base}/health`, { signal: controller.signal })
        .then(r => {
          clearTimeout(t)
          if (!cancelled) setStatus(r.ok ? 'ok' : 'error')
        })
        .catch(() => {
          clearTimeout(t)
          if (!cancelled && retries > 0) {
            setTimeout(() => attempt(retries - 1), 5000)
          } else if (!cancelled) {
            setStatus('error')
          }
        })
    }

    attempt(3)
    return () => {
      cancelled = true
      clearTimeout(warmTimer)
    }
  }, [])

  if (status === 'checking') return (
    <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
      <span className="w-1.5 h-1.5 rounded-full bg-slate-600 animate-pulse" />
      Connecting…
    </div>
  )

  if (status === 'warming') return (
    <div className="flex items-center gap-1.5 text-[10px] text-yellow-500">
      <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 animate-pulse" />
      Backend warming up…
    </div>
  )

  return (
    <div className={clsx(
      'flex items-center gap-1.5 text-[10px] font-medium px-2 py-1 rounded-md',
      status === 'ok'
        ? 'text-green-400 bg-green-950/40'
        : 'text-red-400 bg-red-950/40',
    )}>
      {status === 'ok' ? <Wifi size={10} /> : <WifiOff size={10} />}
      {status === 'ok' ? 'Backend online' : 'Backend offline — check Render env vars'}
    </div>
  )
}

export default function App() {
  const { universe, startDate, endDate, setUniverse, setDateRange } = useStore()

  return (
    <BrowserRouter>
      <div className="flex h-full">
        {/* ── Sidebar ── */}
        <aside className="w-60 flex-shrink-0 flex flex-col bg-surface-card border-r border-surface-border">

          {/* Logo */}
          <div className="px-4 py-4 border-b border-surface-border">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-brand-500 flex items-center justify-center shadow-lg shadow-brand-500/30">
                <Bot size={17} className="text-white" />
              </div>
              <div>
                <p className="text-sm font-bold tracking-tight text-slate-100">ARBITRON</p>
                <p className="text-[10px] text-slate-500 leading-none mt-0.5">Quant Research Platform</p>
              </div>
            </div>
          </div>

          {/* Nav */}
          <nav className="py-3 px-2 space-y-0.5">
            {NAV.map(({ to, icon: Icon, label, desc }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 px-3 py-2.5 rounded-xl transition-colors group',
                    isActive
                      ? 'bg-brand-500/15 text-brand-400 border border-brand-500/20'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent',
                  )
                }
              >
                <Icon size={15} className="flex-shrink-0" />
                <div>
                  <p className="text-xs font-semibold leading-none">{label}</p>
                  <p className="text-[10px] text-slate-600 mt-0.5 group-hover:text-slate-500 leading-none">{desc}</p>
                </div>
              </NavLink>
            ))}
          </nav>

          {/* Global settings */}
          <div className="mx-3 mt-1 p-3 rounded-xl bg-surface-DEFAULT border border-surface-border space-y-3">
            <p className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">
              Settings — shared across all pages
            </p>

            <div>
              <label className="text-[10px] text-slate-500 block mb-1.5">Stock Universe</label>
              <select
                value={universe}
                onChange={(e) => setUniverse(e.target.value)}
                className="w-full bg-surface-card border border-surface-border text-xs text-slate-300
                           rounded-lg px-2 py-1.5 focus:outline-none focus:border-brand-500/60"
              >
                {UNIVERSES.map((u) => (
                  <option key={u.value} value={u.value}>{u.label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-[10px] text-slate-500 block mb-1.5">Date Range</label>
              <div className="space-y-1">
                <input
                  type="date" value={startDate}
                  onChange={(e) => setDateRange(e.target.value, endDate)}
                  className="w-full bg-surface-card border border-surface-border text-xs text-slate-300
                             rounded-lg px-2 py-1.5 focus:outline-none focus:border-brand-500/60"
                />
                <input
                  type="date" value={endDate}
                  onChange={(e) => setDateRange(startDate, e.target.value)}
                  className="w-full bg-surface-card border border-surface-border text-xs text-slate-300
                             rounded-lg px-2 py-1.5 focus:outline-none focus:border-brand-500/60"
                />
              </div>
            </div>
          </div>

          <div className="flex-1" />

          {/* Footer */}
          <div className="px-4 py-3 border-t border-surface-border space-y-2">
            <BackendStatus />
            <p className="text-[10px] text-slate-600">v0.1.0 · Groq · Llama 3.3 70B</p>
          </div>
        </aside>

        {/* ── Main ── */}
        <main className="flex-1 overflow-y-auto bg-surface-DEFAULT">
          <Routes>
            <Route path="/"          element={<AgentPage />} />
            <Route path="/factors"   element={<FactorsPage />} />
            <Route path="/backtest"  element={<BacktestPage />} />
            <Route path="/portfolio" element={<PortfolioPage />} />
            <Route path="/risk"      element={<RiskPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
