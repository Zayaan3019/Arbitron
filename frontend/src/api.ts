/**
 * Typed API client for the Arbitron FastAPI backend.
 */

// In production set VITE_API_URL=https://your-app.onrender.com; in dev leave unset (Vite proxy handles /api)
const _BACKEND = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
const BASE = `${_BACKEND}/api`

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ICStats {
  factor_name: string
  ic_mean: number
  ic_std: number
  ic_ir: number
  t_stat: number
  pct_positive: number
  n_periods: number
  fama_macbeth_gamma?: number
  fama_macbeth_t_stat?: number
}

export interface BacktestMetrics {
  factor_name: string
  cagr: number
  sharpe: number
  sortino: number
  calmar: number
  max_drawdown: number
  win_rate: number
  total_return: number
  ann_volatility: number
  skewness: number
  kurtosis: number
  n_periods: number
}

export interface BacktestResult {
  metrics: BacktestMetrics
  dates: string[]
  cumulative_returns: number[]
  drawdown: number[]
}

export interface WeightsResult {
  tickers: string[]
  weights: number[]
  expected_return: number
  volatility: number
  sharpe: number
  diversification_ratio: number
}

export interface FrontierResult {
  returns: number[]
  volatilities: number[]
  sharpes: number[]
}

export interface RiskResult {
  var_95: number
  var_99: number
  cvar_95: number
  cvar_99: number
  ulcer_index: number
  tail_ratio: number
  max_drawdown_duration_days: number
  avg_drawdown_duration_days: number
  n_drawdown_periods: number
  market_beta?: number
  regime?: string
}

export interface AgentRunResult {
  session_id: string
  status: string
  message: string
}

export interface AgentEvent {
  event_type: 'thinking' | 'tool_call' | 'tool_result' | 'text' | 'done' | 'error' | 'ping'
  data: unknown
  session_id?: string
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
  return res.json()
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(res.statusText)
  return res.json()
}

// ---------------------------------------------------------------------------
// API methods
// ---------------------------------------------------------------------------

export const api = {
  factors: {
    list: () => get<{ factors: string[] }>('/factors'),

    evaluate: (params: {
      factor_name: string
      universe?: string
      start_date?: string
      end_date?: string
      horizon?: number
    }) => post<ICStats>('/factors/evaluate', params),
  },

  backtest: {
    run: (params: {
      factor_name: string
      universe?: string
      start_date?: string
      end_date?: string
      n_quantiles?: number
      rebal_freq?: string
      transaction_cost?: number
      long_only?: boolean
    }) => post<BacktestResult>('/backtest', params),
  },

  portfolio: {
    optimize: (params: {
      universe?: string
      optimizer?: string
      start_date?: string
      end_date?: string
      rf?: number
      max_weight?: number
    }) => post<WeightsResult>('/portfolio/optimize', params),

    frontier: (params: {
      universe?: string
      start_date?: string
      end_date?: string
    }) => post<FrontierResult>('/portfolio/frontier', params),
  },

  risk: {
    compute: (params: {
      universe?: string
      weights?: Record<string, number>
      start_date?: string
      end_date?: string
    }) => post<RiskResult>('/risk', params),
  },

  agent: {
    run: (task: string, max_iterations = 15) =>
      post<AgentRunResult>('/agent/run', { task, max_iterations }),
  },

  universes: {
    list: () => get<Record<string, string[]>>('/universes'),
  },
}

// ---------------------------------------------------------------------------
// WebSocket helper
// ---------------------------------------------------------------------------

export function connectAgentStream(
  sessionId: string,
  onEvent: (event: AgentEvent) => void,
  onClose?: () => void,
): () => void {
  let wsBase: string
  if (_BACKEND) {
    wsBase = _BACKEND.replace(/^https?/, (s: string) => (s === 'https' ? 'wss' : 'ws'))
  } else {
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
    wsBase = `${protocol}://${location.host}`
  }
  const ws = new WebSocket(`${wsBase}/ws/agent/${sessionId}`)

  ws.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data))
    } catch {
      // ignore malformed frames
    }
  }

  ws.onclose = () => onClose?.()
  ws.onerror = () => onClose?.()

  return () => ws.close()
}
