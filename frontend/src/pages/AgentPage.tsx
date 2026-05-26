/**
 * Agent chat page — send research tasks, stream events from the ReAct loop.
 */

import { useState, useRef, useEffect } from 'react'
import { Send, Loader2, Trash2, Zap } from 'lucide-react'
import clsx from 'clsx'
import { api, connectAgentStream, AgentEvent } from '../api'
import { useStore, nextId, AgentChatMessage } from '../store'

const QUICK_TASKS = [
  'Screen momentum and volatility factors in SP500 from 2021–2024',
  'Run a full quant pipeline: evaluate all factors, backtest top 3, optimise portfolio',
  'Detect the current market regime and recommend factor tilts',
  'Compare momentum_12_1 vs rsi_contrarian in the TECH universe',
]

export default function AgentPage() {
  const [input, setInput]   = useState('')
  const bottomRef           = useRef<HTMLDivElement>(null)
  const disconnectRef       = useRef<(() => void) | null>(null)

  const { agentMessages, agentRunning, appendAgentMessage, setAgentRunning, clearAgentMessages } =
    useStore()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [agentMessages])

  async function submit(task: string) {
    if (!task.trim() || agentRunning) return
    setInput('')
    setAgentRunning(true)

    appendAgentMessage({
      id: nextId(), type: 'user', content: task, timestamp: Date.now(),
    })

    try {
      const { session_id } = await api.agent.run(task)

      disconnectRef.current = connectAgentStream(
        session_id,
        (event: AgentEvent) => handleEvent(event),
        () => setAgentRunning(false),
      )
    } catch (err) {
      appendAgentMessage({
        id: nextId(), type: 'error',
        content: `Failed to start agent: ${err}`,
        timestamp: Date.now(),
      })
      setAgentRunning(false)
    }
  }

  function handleEvent(event: AgentEvent) {
    if (event.event_type === 'ping') return

    if (event.event_type === 'thinking') {
      appendAgentMessage({
        id: nextId(), type: 'thinking',
        content: String(event.data ?? ''),
        timestamp: Date.now(),
      })
    } else if (event.event_type === 'tool_call') {
      const d = event.data as Record<string, unknown>
      appendAgentMessage({
        id: nextId(), type: 'tool_call',
        content: `→ ${d.tool}`,
        timestamp: Date.now(),
        meta: d as Record<string, unknown>,
      })
    } else if (event.event_type === 'tool_result') {
      const d = event.data as Record<string, unknown>
      const ok = !(d.result as Record<string, unknown>)?.error
      appendAgentMessage({
        id: nextId(), type: 'tool_result',
        content: `${ok ? '✓' : '✗'} ${d.tool}`,
        timestamp: Date.now(),
        meta: d as Record<string, unknown>,
      })
    } else if (event.event_type === 'text') {
      appendAgentMessage({
        id: nextId(), type: 'text',
        content: String(event.data ?? ''),
        timestamp: Date.now(),
      })
    } else if (event.event_type === 'done' || event.event_type === 'error') {
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
      <div className="px-6 py-4 border-b border-surface-border flex items-center justify-between">
        <div>
          <h1 className="text-base font-semibold text-slate-100">Research Agent</h1>
          <p className="text-xs text-slate-500">Claude-powered autonomous quant research loop</p>
        </div>
        <button
          onClick={clearAgentMessages}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400
                     hover:text-slate-200 border border-surface-border rounded-lg
                     hover:border-slate-600 transition-colors"
        >
          <Trash2 size={12} /> Clear
        </button>
      </div>

      {/* Quick tasks */}
      {agentMessages.length === 0 && (
        <div className="px-6 py-6">
          <p className="text-xs text-slate-500 mb-3 uppercase tracking-wider">Quick start</p>
          <div className="grid grid-cols-2 gap-2">
            {QUICK_TASKS.map((t) => (
              <button
                key={t}
                onClick={() => submit(t)}
                className="text-left px-4 py-3 rounded-xl border border-surface-border
                           bg-surface-card hover:border-brand-500/50 hover:bg-brand-500/5
                           text-xs text-slate-300 transition-colors leading-relaxed"
              >
                <Zap size={12} className="inline text-brand-500 mr-1.5 mb-0.5" />
                {t}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-2">
        {agentMessages.map((m) => <MessageBubble key={m.id} msg={m} />)}
        {agentRunning && (
          <div className="flex items-center gap-2 text-slate-500 text-xs py-1">
            <Loader2 size={12} className="animate-spin" />
            Agent is researching...
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-surface-border">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && submit(input)}
            placeholder="Describe a research task..."
            disabled={agentRunning}
            className="flex-1 bg-surface-card border border-surface-border rounded-xl
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
                         hover:bg-brand-600 disabled:opacity-40 transition-colors"
            >
              <Send size={14} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function MessageBubble({ msg }: { msg: AgentChatMessage }) {
  const [expanded, setExpanded] = useState(false)

  if (msg.type === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[70%] px-4 py-2.5 rounded-2xl rounded-tr-sm
                        bg-brand-500 text-white text-sm leading-relaxed">
          {msg.content}
        </div>
      </div>
    )
  }

  if (msg.type === 'text') {
    return (
      <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-tl-sm
                      bg-surface-card border border-surface-border
                      text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
        {msg.content}
      </div>
    )
  }

  if (msg.type === 'thinking') {
    return (
      <div className="text-xs text-slate-500 pl-1 italic leading-relaxed max-w-[85%] truncate">
        {msg.content}
      </div>
    )
  }

  if (msg.type === 'tool_call') {
    return (
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-start gap-2 text-xs text-left"
      >
        <span className="text-blue-400 font-mono">{msg.content}</span>
        {msg.meta && (
          <span className="text-slate-600 font-mono truncate max-w-xs">
            {JSON.stringify(msg.meta.input).slice(0, 80)}
          </span>
        )}
      </button>
    )
  }

  if (msg.type === 'tool_result') {
    const ok = msg.content.startsWith('✓')
    return (
      <div
        className={clsx(
          'text-xs font-mono px-2 py-0.5 rounded inline-block',
          ok ? 'text-green-400' : 'text-red-400'
        )}
      >
        {msg.content}
      </div>
    )
  }

  if (msg.type === 'error') {
    return (
      <div className="text-xs text-red-400 bg-red-950/30 border border-red-800/30
                      rounded-lg px-3 py-2 font-mono">
        {msg.content}
      </div>
    )
  }

  return null
}
