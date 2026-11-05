import { useState } from 'react'
import type { ReactNode } from 'react'
import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Loader2,
  Radar,
  ShieldAlert,
  ShieldCheck,
  XCircle,
} from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import clsx from 'clsx'
import { api, LabCandidate, LabGate, LabReport } from '../api'
import { useStore } from '../store'
import { fmt as fmtNum, pct as pctNum, type Num } from '../format'
import DataCaveatNotice from '../components/DataCaveatNotice'

function friendlyError(err: unknown): string {
  const msg = String(err)
  if (msg.includes('Failed to fetch') || msg.toLowerCase().includes('network')) {
    return 'Cannot reach the backend. Check VITE_API_URL or start the FastAPI server.'
  }
  return msg.replace(/^Error:\s*/, '')
}

const fmt = (value: unknown, digits = 2) => fmtNum(value as Num, digits)
const pct = (value: unknown, digits = 1) => pctNum(value as Num, digits)

export default function LabPage() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [includeFoundation, setIncludeFoundation] = useState(true)
  const [maxFactors, setMaxFactors] = useState(6)
  const [horizon, setHorizon] = useState(21)

  const { universe, startDate, endDate, labResult, setLabResult } = useStore()

  async function runLab() {
    setLoading(true)
    setError(null)
    try {
      const result = await api.lab.run({
        universe,
        start_date: startDate,
        end_date: endDate,
        horizon,
        include_foundation: includeFoundation,
        max_factors: maxFactors,
      })
      setLabResult(result.report)
    } catch (e) {
      setError(friendlyError(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-base font-bold text-slate-100">Autonomous Quant Research Lab</h1>
          <p className="text-xs text-slate-400 leading-relaxed max-w-3xl">
            Factor discovery, market representation learning, backtest auditing, and deployment gates in one workflow.
          </p>
        </div>
        <button
          onClick={runLab}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-brand-500
                     text-white text-sm font-medium hover:bg-brand-600 disabled:opacity-50 transition-colors"
        >
          {loading ? <Loader2 size={15} className="animate-spin" /> : <Radar size={15} />}
          {loading ? 'Running Lab' : 'Run Lab'}
        </button>
      </div>

      <div className="card">
        <div className="grid grid-cols-4 gap-4">
          <Control label="Forward Horizon">
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                         text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-brand-500/60"
            >
              <option value={10}>10 trading days</option>
              <option value={21}>21 trading days</option>
              <option value={63}>63 trading days</option>
            </select>
          </Control>
          <Control label="Research Queue">
            <input
              type="number"
              min={1}
              max={9}
              value={maxFactors}
              onChange={(e) => setMaxFactors(Number(e.target.value))}
              className="w-full bg-surface-DEFAULT border border-surface-border text-sm
                         text-slate-300 rounded-lg px-3 py-1.5 focus:outline-none focus:border-brand-500/60"
            />
          </Control>
          <Control label="Foundation Signal">
            <label className="flex items-center gap-2.5 h-8 cursor-pointer">
              <input
                type="checkbox"
                checked={includeFoundation}
                onChange={(e) => setIncludeFoundation(e.target.checked)}
                className="w-4 h-4 accent-brand-500"
              />
              <span className="text-sm text-slate-300">Enabled</span>
            </label>
          </Control>
          <Control label="Active Universe">
            <div className="h-8 flex items-center text-sm text-slate-300 font-mono">{universe}</div>
          </Control>
        </div>
      </div>

      {loading && (
        <div className="card flex items-center justify-center gap-3 py-12">
          <Loader2 size={20} className="animate-spin text-brand-500" />
          <p className="text-xs text-slate-500">Running walk-forward research, statistical audit gates, and foundation encoding.</p>
        </div>
      )}

      {error && (
        <div className="card border border-red-800/40 bg-red-950/20">
          <p className="text-xs font-semibold text-red-400 mb-1">Research Lab Error</p>
          <p className="text-xs text-red-300">{error}</p>
        </div>
      )}

      {!loading && !error && !labResult && (
        <div className="card flex flex-col items-center justify-center py-12 gap-2 text-center">
          <BrainCircuit size={28} className="text-slate-600" />
          <p className="text-sm font-semibold text-slate-300">No lab run yet</p>
          <p className="text-xs text-slate-600 max-w-md">
            Run the lab to rank candidates with statistical defenses and market-foundation diagnostics.
          </p>
        </div>
      )}

      {labResult && !loading && <LabReportView report={labResult} />}
    </div>
  )
}

function Control({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">{label}</p>
      {children}
    </div>
  )
}

function LabReportView({ report }: { report: LabReport }) {
  const champion = report.champion
  const verdict = champion?.audit?.verdict ?? report.lab_verdict
  const isReady = verdict === 'deployable' || report.lab_verdict === 'candidate_ready'
  const dsrEvaluable = champion?.audit?.deflated_sharpe != null

  return (
    <div className="space-y-4">
      <div className={clsx(
        'card border',
        isReady ? 'border-green-800/40 bg-green-950/20' : 'border-orange-800/40 bg-orange-950/20',
      )}>
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Champion</p>
            <p className="text-xl font-bold text-slate-100">{champion?.name ?? 'No champion'}</p>
            <p className="text-xs text-slate-400 mt-1">
              {champion
                ? `${champion.kind} candidate with score ${fmt(champion.research_score, 1)}`
                : 'No strategy survived the walk-forward research run.'}
            </p>
          </div>
          <div className="grid grid-cols-4 gap-4 min-w-[520px]">
            <Metric label="Sharpe" value={fmt(champion?.metrics?.sharpe)} />
            <Metric
              label="DSR"
              value={dsrEvaluable ? pct(champion?.audit?.deflated_sharpe) : 'n/e'}
              muted={!dsrEvaluable}
            />
            <Metric label="PBO" value={pct(champion?.audit?.pbo)} danger />
            <Metric label="Max DD" value={pct(champion?.metrics?.max_drawdown)} danger />
          </div>
        </div>
      </div>

      {!dsrEvaluable && champion && (
        <div className="card border border-amber-800/40 bg-amber-950/20">
          <p className="text-[11px] font-semibold text-amber-300 mb-1">
            Deflated Sharpe not evaluable
          </p>
          <p className="text-[11px] text-amber-200/80 leading-relaxed">
            {champion.audit?.deflated_sharpe_reason ??
              'The multiple-testing hurdle could not be estimated for this run.'}
            {' '}The Sharpe above is therefore <strong>un-deflated</strong> — treat it as a
            ceiling, not a result.
          </p>
        </div>
      )}

      <DataCaveatNotice caveats={report.data_caveats} />

      {champion?.audit?.n_trials != null && (
        <p className="text-[10px] text-slate-600 px-1">
          Deflated against {champion.audit.n_trials} trials
          {champion.audit.n_peer_strategies != null &&
            ` (${champion.audit.n_peer_strategies} in this comparison, the rest from earlier
              backtests in this session — the search is the burden, not the shortlist)`}
          {champion.audit.expected_max_sharpe_hurdle != null &&
            `. Selection hurdle: ${fmt(champion.audit.expected_max_sharpe_hurdle)} Sharpe.`}
        </p>
      )}

      <div className="grid grid-cols-3 gap-4">
        <DefensePanel gates={champion?.audit?.gates ?? []} />
        <FoundationPanel report={report} />
        <MemoPanel memo={report.research_memo} />
      </div>

      <div className="grid grid-cols-5 gap-4">
        <div className="card col-span-3">
          <p className="text-xs font-semibold text-slate-300 mb-3">Champion Equity Curve</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={report.champion_equity_curve}>
              <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickCount={6} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#161b27', border: '1px solid #252d3d', fontSize: 12 }} />
              <Line type="monotone" dataKey="value" stroke="#4f72ff" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <RegimePanel candidate={champion} />
      </div>

      <CandidateTable candidates={report.candidates} />
    </div>
  )
}

function DefensePanel({ gates }: { gates: LabGate[] }) {
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-3">
        <ShieldCheck size={15} className="text-green-400" />
        <p className="text-xs font-semibold text-slate-300">Statistical Gates</p>
      </div>
      <div className="space-y-2">
        {gates.map((gate) => (
          <div key={gate.name} className="flex items-center justify-between gap-3"
               title={gate.reason ?? undefined}>
            <span className="text-xs text-slate-500">{gate.name}</span>
            <span className={clsx(
              'inline-flex items-center gap-1 text-xs font-mono',
              gate.pass === true && 'text-green-400',
              gate.pass === false && 'text-red-400',
              gate.pass === null && 'text-amber-400',
            )}>
              {gate.pass === true && <CheckCircle2 size={12} />}
              {gate.pass === false && <XCircle size={12} />}
              {gate.pass === null && <ShieldAlert size={12} />}
              {gate.pass === null ? 'not evaluable' : fmt(gate.value, 3)}
            </span>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-600 mt-3 leading-relaxed">
        A gate marked <span className="text-amber-400">not evaluable</span> is not a pass. Any
        candidate carrying one is held out of "deployable".
      </p>
    </div>
  )
}

function FoundationPanel({ report }: { report: LabReport }) {
  const diagnostics = report.foundation.diagnostics ?? {}
  const enabled = report.foundation.enabled
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-3">
        <BrainCircuit size={15} className="text-brand-500" />
        <p className="text-xs font-semibold text-slate-300">Foundation Adapter</p>
      </div>
      {enabled ? (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <Metric label="Refits" value={String(diagnostics.refits ?? 0)} />
            <Metric label="Coverage" value={pct(Number(diagnostics.coverage_ratio ?? 0))} />
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Latest State Embedding</p>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(report.foundation.latest_embedding ?? {}).slice(0, 4).map(([k, v]) => (
                <span key={k} className="px-2 py-1 rounded-md bg-surface-DEFAULT text-[10px] font-mono text-slate-400">
                  {k}: {fmt(v, 2)}
                </span>
              ))}
            </div>
          </div>
          <p className="text-xs text-slate-500">
            Reconstruction error: {fmt(report.foundation.latest_reconstruction_error, 4)}
          </p>
        </div>
      ) : (
        <p className="text-xs text-slate-500">{report.foundation.error ?? 'Disabled for this run.'}</p>
      )}
    </div>
  )
}

function MemoPanel({ memo }: { memo: string[] }) {
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-3">
        <Activity size={15} className="text-cyan-400" />
        <p className="text-xs font-semibold text-slate-300">Research Memo</p>
      </div>
      <div className="space-y-2">
        {memo.slice(0, 4).map((line, i) => (
          <p key={i} className="text-xs text-slate-400 leading-relaxed">{line}</p>
        ))}
      </div>
    </div>
  )
}

function RegimePanel({ candidate }: { candidate: LabCandidate | null }) {
  const regimes = Object.entries(candidate?.regime_robustness ?? {})
  return (
    <div className="card col-span-2">
      <p className="text-xs font-semibold text-slate-300 mb-3">Regime Robustness</p>
      <div className="grid grid-cols-2 gap-3">
        {regimes.map(([name, stats]) => (
          <div key={name} className="bg-surface-DEFAULT border border-surface-border rounded-lg p-3">
            <p className="text-xs font-semibold text-slate-300 mb-2">{name}</p>
            <Metric label="Sharpe" value={fmt(stats.sharpe)} />
            <Metric label="Hit Rate" value={pct(stats.hit_rate)} />
          </div>
        ))}
      </div>
    </div>
  )
}

function CandidateTable({ candidates }: { candidates: LabCandidate[] }) {
  return (
    <div className="card">
      <p className="text-xs font-semibold text-slate-300 mb-3">Ranked Candidate Queue</p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-slate-500 border-b border-surface-border">
              <th className="py-2 font-medium">Candidate</th>
              <th className="py-2 font-medium">Kind</th>
              <th className="py-2 font-medium">Score</th>
              <th className="py-2 font-medium">Sharpe</th>
              <th className="py-2 font-medium">DSR</th>
              <th className="py-2 font-medium">PBO</th>
              <th className="py-2 font-medium">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((candidate) => (
              <tr key={candidate.name} className="border-b border-surface-border/60">
                <td className="py-2 text-slate-200 font-medium">{candidate.name}</td>
                <td className="py-2 text-slate-500">{candidate.kind}</td>
                <td className="py-2 font-mono text-slate-300">{fmt(candidate.research_score, 1)}</td>
                <td className="py-2 font-mono text-slate-300">{fmt(candidate.metrics?.sharpe)}</td>
                <td className={clsx(
                  'py-2 font-mono',
                  candidate.audit?.deflated_sharpe == null ? 'text-amber-400/80' : 'text-slate-300',
                )}>
                  {candidate.audit?.deflated_sharpe == null
                    ? 'n/e'
                    : pct(candidate.audit.deflated_sharpe)}
                </td>
                <td className="py-2 font-mono text-slate-300">{pct(candidate.audit?.pbo)}</td>
                <td className="py-2">
                  <span className={clsx(
                    'px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wider',
                    candidate.audit?.verdict === 'deployable' && 'bg-green-950/40 text-green-400',
                    candidate.audit?.verdict === 'watchlist' && 'bg-yellow-950/40 text-yellow-400',
                    candidate.audit?.verdict === 'reject' && 'bg-red-950/40 text-red-400',
                  )}>
                    {candidate.audit?.verdict ?? 'N/A'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Metric(
  { label, value, danger = false, muted = false }:
  { label: string; value: string; danger?: boolean; muted?: boolean },
) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className={clsx(
        'text-base font-semibold font-mono mt-0.5',
        muted ? 'text-amber-400' : danger ? 'text-red-400' : 'text-slate-100',
      )}>
        {value}
      </p>
    </div>
  )
}
