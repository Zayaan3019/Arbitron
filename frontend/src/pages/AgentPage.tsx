import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, Trash2, Zap, Bot, Search, BarChart2, TrendingUp, ShieldCheck } from 'lucide-react'
import clsx from 'clsx'
import { api, connectAgentStream, AgentEvent } from '../api'
import { useStore, nextId, AgentChatMessage } from '../store'

// Human-readable names for agent tool calls
const TOOL_LABELS: Record<string, string> = {
  fetch_universe:    'Fetching stock list',
  compute_factor:    'Computing signal',
  evaluate_factor:   'Evaluating signal quality',
  run_backtest:      'Running backtest simulation',
  optimize_portfolio:'Optimizing portfolio weights',
  compute_risk:      'Analyzing portfolio risk',
  detect_regime:     'Detecting market regime',
  predict_returns:   'Predicting returns',
}

const QUICK_TASKS = [
  { icon: Search,     label: 'Find the best momentum signal in tech stocks from 2021–2024',           hint: 'Discovers which momentum factor predicts returns best' },
  { icon: TrendingUp, label: 'Run a full backtest of the top 3 factors and rank them by Sharpe ratio', hint: 'Compares strategies side by side' },
  { icon: BarChart2,  label: 'Build an optimal portfolio from S&P 500 stocks and show the allocation', hint: 'Runs portfolio optimization for you' },
  { icon: ShieldCheck,label: 'What is the current market regime and what does it mean for my strategy?', hint: 'Detects bull/bear signals and gives advice' },
]

function friendlyError(err: unknown): string {
  const msg = String(err)
  if (msg.includes('Failed to fetch') || msg.toLowerCase().includes('network')) {
    return 'Cannot reach the backend server. Check that VITE_API_URL is set in your Vercel environment variables and the Render service is running.'
  }
  return msg.replace(/^Error:\s*/, '')
}

export default function AgentPage() {
  const [input, setInput] = useState('')
  const [tokenUsage, setTokenUsage] = useState<{ used: number; budget: number } | null>(null)
  const bottomRef         = useRef<HTMLDivElement>(null)
  const disconnectRef     = useRef<(() => void) | null>(null)

  const { agentMessages, agentRunning, appendAgentMessage, setAgentRunning, clearAgentMessages } = useStore()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [agentMessages])

  async function submit(task: string) {
    if (!task.trim() || agentRunning) return
    setInput('')
    setAgentRunning(true)
    appendAgentMessage({ id: nextId(), type: 'user', content: task, timestamp: Date.now() })

    try {
      const { session_id } = await api.agent.run(task)
      disconnectRef.current = connectAgentStream(
        session_id,
        (event: AgentEvent) => handleEvent(event),
        () => setAgentRunning(false),
      )
    } catch (err) {
      appendAgentMessage({ id: nextId(), type: 'error', content: friendlyError(err), timestamp: Date.now() })
      setAgentRunning(false)
    }
  }

  function handleEvent(event: AgentEvent) {
    if (event.event_type === 'ping') return
    if (event.event_type === 'thinking') {
      appendAgentMessage({ id: nextId(), type: 'thinking', content: String(event.data ?? ''), timestamp: Date.now() })
    } else if (event.event_type === 'tool_call') {
      const d = event.data as Record<string, unknown>
      const toolName = String(d.tool ?? '')
      appendAgentMessage({
        id: nextId(), type: 'tool_call',
        content: TOOL_LABELS[toolName] ?? toolName,
        timestamp: Date.now(), meta: d,
      })
    } else if (event.event_type === 'tool_result') {
      const d = event.data as Record<string, unknown>
      const ok = !(d.result as Record<string, unknown>)?.error
      const toolName = String(d.tool ?? '')
      appendAgentMessage({
        id: nextId(), type: 'tool_result',
        content: `${ok ? 'Done' : 'Failed'}: ${TOOL_LABELS[toolName] ?? toolName}`,
        timestamp: Date.now(), meta: d,
      })
    } else if (event.event_type === 'usage') {
      // Token accounting, surfaced so a run that stops on budget rather than
      // on a conclusion is visibly different from one that finished.
      const d = event.data as Record<string, number>
      setTokenUsage({ used: d.cumulative_tokens ?? 0, budget: d.budget ?? 0 })
    } else if (event.event_type === 'text') {
      appendAgentMessage({ id: nextId(), type: 'text', content: String(event.data ?? ''), timestamp: Date.now() })
    } else if (event.event_type === 'error') {
      const d = event.data as Record<string, unknown> | string | null
      const message = typeof d === 'string' ? d : String((d as Record<string, unknown>)?.message ?? 'Agent run failed.')
      appendAgentMessage({ id: nextId(), type: 'error', content: message, timestamp: Date.now() })
      setAgentRunning(false)
    } else if (event.event_type === 'done') {
      setAgentRunning(false)
    }
  }

  function stop() {
    disconnectRef.current?.()
    setAgentRunning(false)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-4 border-b border-surface-border bg-surface-card flex items-center justify-between">
        <div>
          <h1 className="text-sm font-bold text-slate-100">AI Research Assistant</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Describe what you want to research — the AI writes and runs the analysis for you
          </p>
          {tokenUsage && tokenUsage.budget > 0 && (
            <p className="text-[10px] text-slate-600 mt-1 font-mono">
              {tokenUsage.used.toLocaleString()} / {tokenUsage.budget.toLocaleString()} tokens
              {tokenUsage.used >= tokenUsage.budget && ' — budget reached, run halted'}
            </p>
          )}
        </div>
        {agentMessages.length > 0 && (
          <button
            onClick={clearAgentMessages}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400
                       hover:text-slate-200 border border-surface-border rounded-lg
                       hover:border-slate-600 transition-colors"
          >
            <Trash2 size={11} /> Clear chat
          </button>
        )}
      </div>

      {/* Welcome / quick tasks */}
      {agentMessages.length === 0 && (
        <div className="px-6 py-8 space-y-6">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-xl bg-brand-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
              <Bot size={18} className="text-brand-400" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-200 mb-1">Hello! I'm your quant research AI.</p>
              <p className="text-xs text-slate-400 leading-relaxed max-w-lg">
                Tell me what you want to research in plain English. I'll automatically fetch market data,
                compute signals, run backtests, and give you a clear summary — no coding needed.
              </p>
            </div>
          </div>

          <div>
            <p className="text-xs text-slate-500 mb-3 font-medium uppercase tracking-wider">Try one of these</p>
            <div className="grid grid-cols-2 gap-2">
              {QUICK_TASKS.map((t) => (
                <button
                  key={t.label}
                  onClick={() => submit(t.label)}
                  className="text-left px-4 py-3 rounded-xl border border-surface-border
                             bg-surface-card hover:border-brand-500/40 hover:bg-brand-500/5
                             transition-colors space-y-1 group"
                >
                  <div className="flex items-center gap-2">
                    <t.icon size={13} className="text-brand-400 flex-shrink-0" />
                    <span className="text-xs text-slate-200 font-medium leading-snug">{t.label}</span>
                  </div>
                  <p className="text-[10px] text-slate-600 group-hover:text-slate-500 pl-5">{t.hint}</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3">
        {agentMessages.map((m) => <MessageBubble key={m.id} msg={m} />)}
        {agentRunning && (
          <div className="flex items-center gap-2 text-slate-500 text-xs py-1">
            <Loader2 size={12} className="animate-spin" />
            AI is working on your request…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="px-6 py-4 border-t border-surface-border bg-surface-card">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && submit(input)}
            placeholder="e.g. Which sector had the best momentum signal last year?"
            disabled={agentRunning}
            className="flex-1 bg-surface-DEFAULT border border-surface-border rounded-xl
                       px-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-600
                       focus:outline-none focus:border-brand-500/60 disabled:opacity-50"
          />
          {agentRunning ? (
            <button
              onClick={stop}
              className="px-4 py-2.5 rounded-xl bg-red-600/20 text-red-400 border
                         border-red-800/50 text-sm hover:bg-red-600/30 transition-colors"
            >
              Stop
            </button>
          ) : (
            <button
              onClick={() => submit(input)}
              disabled={!input.trim()}
              className="px-4 py-2.5 rounded-xl bg-brand-500 text-white text-sm
                         hover:bg-brand-600 disabled:opacity-40 transition-colors flex items-center gap-2"
            >
              <Send size={14} /> Ask
            </button>
          )}
        </div>
        <p className="text-[10px] text-slate-600 mt-2">
          The AI can take 30–60 seconds to research a complex task. It will show each step it takes.
        </p>
      </div>
    </div>
  )
}

function MessageBubble({ msg }: { msg: AgentChatMessage }) {
  if (msg.type === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] px-4 py-2.5 rounded-2xl rounded-tr-sm bg-brand-500 text-white text-sm leading-relaxed">
          {msg.content}
        </div>
      </div>
    )
  }

  if (msg.type === 'text') {
    return (
      <div className="flex gap-2.5">
        <div className="w-6 h-6 rounded-lg bg-brand-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
          <Bot size={12} className="text-brand-400" />
        </div>
        <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-tl-sm bg-surface-card
                        border border-surface-border text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
          {msg.content}
        </div>
      </div>
    )
  }

  if (msg.type === 'thinking') {
    return (
      <div className="text-xs text-slate-600 pl-9 italic leading-relaxed max-w-[85%]">
        💭 {msg.content}
      </div>
    )
  }

  if (msg.type === 'tool_call') {
    return (
      <div className="flex items-center gap-2 pl-9 text-xs text-blue-400">
        <Loader2 size={10} className="animate-spin flex-shrink-0" />
        <span>{msg.content}…</span>
      </div>
    )
  }

  if (msg.type === 'tool_result') {
    const ok = msg.content.startsWith('Done')
    const result = msg.meta?.result as Record<string, unknown> | undefined
    const errMsg: string | null = !ok
      ? ((result?.message as string) || (typeof result?.error === 'string' ? result.error : null))
      : null
    return (
      <div className="pl-9 space-y-0.5">
        <div className={clsx('flex items-center gap-1.5 text-xs font-medium', ok ? 'text-green-400' : 'text-red-400')}>
          <span>{ok ? '✓' : '✗'}</span>
          <span>{msg.content}</span>
        </div>
        {errMsg && (
          <p className="text-[11px] text-red-300/70 leading-relaxed bg-red-950/30 rounded-lg px-2 py-1 border border-red-900/30">
            {errMsg}
          </p>
        )}
      </div>
    )
  }

  if (msg.type === 'error') {
    return (
      <div className="pl-9">
        <div className="text-xs text-red-300 bg-red-950/40 border border-red-800/40
                        rounded-xl px-4 py-3 leading-relaxed max-w-[85%]">
          <p className="font-semibold mb-1">Something went wrong</p>
          <p>{msg.content}</p>
        </div>
      </div>
    )
  }

  return null
}
