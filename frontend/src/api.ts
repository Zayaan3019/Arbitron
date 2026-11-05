/**
 * Typed API client for the Arbitron FastAPI backend.
 *
 * Two things to know:
 *
 * 1. Numeric fields are `number | null`. The backend returns null for metrics
 *    that are genuinely undefined (Sharpe on a handful of observations,
 *    Deflated Sharpe with a single trial) rather than coercing them to 0.
 *    Use the helpers in `format.ts` to render them.
 *
 * 2. When the backend runs with `ARBITRON_REQUIRE_API_KEY=true`, every /api
 *    request needs an `X-API-Key` header and the agent WebSocket needs an
 *    `api_key` query parameter. Set `VITE_API_KEY` at build time.
 *
 *    This is a shared secret for a public demo, not user authentication: a
 *    value compiled into a browser bundle is readable by anyone who opens it.
 *    It stops drive-by traffic and casual scraping from spending the
 *    operator's LLM quota. It is not a substitute for real auth.
 */

// In production set VITE_API_URL=https://your-app.onrender.com; in dev leave unset (Vite proxy handles /api)
const _BACKEND = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
const _API_KEY = (import.meta.env.VITE_API_KEY ?? '').trim()
const BASE = `${_BACKEND}/api`

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DataCaveats {
  point_in_time: boolean | null
  survivorship_bias: boolean | null
  headline: string
  details: string[]
  remediation?: string | null
  universe?: string | null
}

export interface ICStats {
  factor_name: string
  horizon: number
  ic_mean: number | null
  ic_std: number | null
  ic_ir: number | null
  /** Newey-West HAC corrected — the one to quote. */
  t_stat: number | null
  /** iid t-stat, inflated ~sqrt(horizon) by overlapping returns. Reference only. */
  t_stat_naive: number | null
  hac_lags: number
  pct_positive: number | null
  n_periods: number
  fama_macbeth_gamma: number | null
  fama_macbeth_t_stat: number | null
  fama_macbeth_t_stat_naive: number | null
  fama_macbeth_p_value: number | null
  q5_minus_q1_ann: number | null
  data_caveats: DataCaveats
}

export interface BacktestMetrics {
  factor_name: string
  cagr: number | null
  sharpe: number | null
  sortino: number | null
  calmar: number | null
  max_drawdown: number | null
  win_rate: number | null
  total_return: number | null
  ann_volatility: number | null
  skewness: number | null
  kurtosis: number | null
  n_periods: number
}

export interface BacktestResult {
  metrics: BacktestMetrics
  dates: string[]
  cumulative_returns: Array<number | null>
  drawdown: Array<number | null>
  n_rebalances: number
  /** ticker → holding days on which the name had no price and was dropped. */
  dropped_names: Record<string, number>
  data_caveats: DataCaveats
}

export interface WeightsResult {
  tickers: string[]
  weights: number[]
  /** Always 1.0 to within 1e-9 — exposed so the invariant is checkable. */
  weight_sum: number
  expected_return: number | null
  volatility: number | null
  sharpe: number | null
  diversification_ratio: number | null
  n_observations: number
  data_caveats: DataCaveats
}

export interface FrontierResult {
  returns: Array<number | null>
  volatilities: Array<number | null>
  sharpes: Array<number | null>
  data_caveats: DataCaveats
}

export interface RiskResult {
  var_95: number | null
  var_99: number | null
  cvar_95: number | null
  cvar_99: number | null
  ulcer_index: number | null
  tail_ratio: number | null
  max_drawdown_duration_days: number
  avg_drawdown_duration_days: number
  n_drawdown_periods: number
  market_beta: number | null
  regime: string | null
  regime_as_of: string | null
  regime_is_causal: boolean
  n_observations: number
  data_caveats: DataCaveats
}

export interface AgentRunResult {
  session_id: string
  status: string
  message: string
}

export interface LabGate {
  name: string
  value: number | null
  threshold: number
  operator: string
  /** `null` means NOT EVALUABLE — never conflate with a pass. */
  pass: boolean | null
  evaluable: boolean
  reason: string | null
}

export interface LabCandidate {
  name: string
  kind: string
  research_score?: number | null
  evaluation?: Record<string, number | null>
  metrics?: Record<string, number | null>
  audit?: {
    verdict: string
    sharpe: number | null
    haircut_sharpe: number | null
    haircut_components?: Record<string, number | null>
    probabilistic_sharpe: number | null
    deflated_sharpe: number | null
    deflated_sharpe_evaluable?: boolean
    deflated_sharpe_reason?: string | null
    expected_max_sharpe_hurdle?: number | null
    n_trials?: number
    n_peer_strategies?: number
    pbo: number | null
    max_drawdown_loss: number | null
    gates: LabGate[]
    warnings: string[]
  }
  regime_robustness?: Record<string, {
    sharpe: number | null
    mean_return_ann: number | null
    hit_rate: number | null
    n_days: number
  }>
}

export interface LabReport {
  lab_verdict: string
  date_range: {
    start: string | null
    end: string | null
    n_days: number
    n_assets: number
  }
  settings: Record<string, unknown>
  champion: LabCandidate | null
  candidates: LabCandidate[]
  defense_summary: {
    tested_strategies?: number
    n_trials_deflated_against?: number
    deployable?: number
    watchlist?: number
    rejected?: number
    global_pbo?: Record<string, unknown>
  }
  foundation: {
    enabled: boolean
    diagnostics?: Record<string, unknown>
    latest_embedding?: Record<string, number>
    latest_reconstruction_error?: number | null
    state_embedding_dates?: string[]
    state_z1?: number[]
    error?: string
  }
  champion_equity_curve: Array<{ date: string; value: number }>
  research_memo: string[]
  data_caveats?: DataCaveats
}

export interface LabRunResult {
  report: LabReport
}

export interface AgentEvent {
  event_type:
    | 'thinking' | 'tool_call' | 'tool_result' | 'usage'
    | 'text' | 'done' | 'error' | 'ping'
  data: unknown
  session_id?: string
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function headers(): Record<string, string> {
  const h: Record<string, string> = { 'Content-Type': 'application/json' }
  if (_API_KEY) h['X-API-Key'] = _API_KEY
  return h
}

async function failure(res: Response): Promise<never> {
  const body = await res.json().catch(() => null)
  const detail = body?.detail ?? res.statusText

  if (res.status === 401) {
    throw new Error(
      'Unauthorized. The backend requires an API key — set VITE_API_KEY in the frontend environment.',
    )
  }
  if (res.status === 429) {
    throw new Error(`Rate limited. ${detail}`)
  }
  throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify(body),
  })
  if (!res.ok) return failure(res)
  return res.json()
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: headers() })
  if (!res.ok) return failure(res)
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
      tickers?: string[]
      start_date?: string
      end_date?: string
      horizon?: number
    }) => post<ICStats>('/factors/evaluate', params),
  },

  backtest: {
    run: (params: {
      factor_name: string
      universe?: string
      tickers?: string[]
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
      tickers?: string[]
      optimizer?: string
      start_date?: string
      end_date?: string
      rf?: number
      max_weight?: number
      long_only?: boolean
    }) => post<WeightsResult>('/portfolio/optimize', params),

    frontier: (params: {
      universe?: string
      tickers?: string[]
      start_date?: string
      end_date?: string
      long_only?: boolean
    }) => post<FrontierResult>('/portfolio/frontier', params),
  },

  risk: {
    compute: (params: {
      universe?: string
      tickers?: string[]
      weights?: Record<string, number>
      start_date?: string
      end_date?: string
    }) => post<RiskResult>('/risk', params),
  },

  lab: {
    run: (params: {
      universe?: string
      tickers?: string[]
      start_date?: string
      end_date?: string
      factor_names?: string[]
      horizon?: number
      include_foundation?: boolean
      n_quantiles?: number
      transaction_cost?: number
      max_factors?: number
    }) => post<LabRunResult>('/lab/run', params),
  },

  agent: {
    run: (task: string, max_iterations = 15) =>
      post<AgentRunResult>('/agent/run', { task, max_iterations }),
  },

  universes: {
    list: () => get<{ universes: Record<string, string[]>; data_caveats: DataCaveats }>('/universes'),
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

  // Browsers cannot set headers on a WebSocket handshake, so the key travels
  // as a query parameter here.
  const query = _API_KEY ? `?api_key=${encodeURIComponent(_API_KEY)}` : ''
  const ws = new WebSocket(`${wsBase}/ws/agent/${sessionId}${query}`)

  ws.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data))
    } catch {
      // ignore malformed frames
    }
  }

  ws.onclose = (e) => {
    if (e.code === 4401) {
      onEvent({
        event_type: 'error',
        data: { message: 'Unauthorized: the agent stream requires a valid API key (VITE_API_KEY).' },
      })
    }
    onClose?.()
  }
  ws.onerror = () => onClose?.()

  return () => ws.close()
}
