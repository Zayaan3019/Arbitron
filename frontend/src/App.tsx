import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { Bot, BarChart3, TrendingUp, ShieldAlert, Cpu } from 'lucide-react'
import AgentPage from './pages/AgentPage'
import FactorsPage from './pages/FactorsPage'
import BacktestPage from './pages/BacktestPage'
import PortfolioPage from './pages/PortfolioPage'
import RiskPage from './pages/RiskPage'
import clsx from 'clsx'

const NAV = [
  { to: '/',          icon: Bot,        label: 'Agent'     },
  { to: '/factors',   icon: BarChart3,  label: 'Factors'   },
  { to: '/backtest',  icon: TrendingUp, label: 'Backtest'  },
  { to: '/portfolio', icon: Cpu,        label: 'Portfolio' },
  { to: '/risk',      icon: ShieldAlert,label: 'Risk'      },
]

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-full">
        {/* Sidebar */}
        <aside className="w-56 flex-shrink-0 flex flex-col bg-surface-card border-r border-surface-border">
          {/* Logo */}
          <div className="px-5 py-5 border-b border-surface-border">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-brand-500 flex items-center justify-center">
                <Bot size={16} className="text-white" />
              </div>
              <span className="text-sm font-bold tracking-tight text-slate-100">
                ARBITRON
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1">Autonomous Quant Research</p>
          </div>

          {/* Nav */}
          <nav className="flex-1 py-4 px-2 space-y-0.5">
            {NAV.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
                    isActive
                      ? 'bg-brand-500/20 text-brand-500 font-medium'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                  )
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </nav>

          <div className="px-4 py-3 border-t border-surface-border">
            <p className="text-[10px] text-slate-600 font-mono">v0.1.0 · claude-opus-4-5</p>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 overflow-y-auto">
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
