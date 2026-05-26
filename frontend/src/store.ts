/**
 * Zustand global state store.
 */

import { create } from 'zustand'
import type { ICStats, BacktestResult, WeightsResult, RiskResult, AgentEvent } from './api'

interface AppState {
  // Shared settings
  universe: string
  startDate: string
  endDate: string
  setUniverse: (u: string) => void
  setDateRange: (start: string, end: string) => void

  // Factor browser
  selectedFactor: string
  icStats: ICStats | null
  backtestResult: BacktestResult | null
  setSelectedFactor: (f: string) => void
  setIcStats: (s: ICStats | null) => void
  setBacktestResult: (r: BacktestResult | null) => void

  // Portfolio
  portfolioWeights: WeightsResult | null
  setPortfolioWeights: (w: WeightsResult | null) => void

  // Risk
  riskResult: RiskResult | null
  setRiskResult: (r: RiskResult | null) => void

  // Agent chat
  agentMessages: AgentChatMessage[]
  agentRunning: boolean
  appendAgentMessage: (m: AgentChatMessage) => void
  setAgentRunning: (v: boolean) => void
  clearAgentMessages: () => void
}

export interface AgentChatMessage {
  id: string
  type: 'user' | 'thinking' | 'tool_call' | 'tool_result' | 'text' | 'error'
  content: string
  timestamp: number
  meta?: Record<string, unknown>
}

let _msgId = 0
export const nextId = () => String(++_msgId)

export const useStore = create<AppState>((set) => ({
  universe:  'SP500',
  startDate: '2021-01-01',
  endDate:   '2024-12-31',
  setUniverse:  (u) => set({ universe: u }),
  setDateRange: (start, end) => set({ startDate: start, endDate: end }),

  selectedFactor: 'momentum_12_1',
  icStats:        null,
  backtestResult: null,
  setSelectedFactor: (f) => set({ selectedFactor: f }),
  setIcStats:        (s) => set({ icStats: s }),
  setBacktestResult: (r) => set({ backtestResult: r }),

  portfolioWeights: null,
  setPortfolioWeights: (w) => set({ portfolioWeights: w }),

  riskResult: null,
  setRiskResult: (r) => set({ riskResult: r }),

  agentMessages: [],
  agentRunning:  false,
  appendAgentMessage: (m) => set((s) => ({ agentMessages: [...s.agentMessages, m] })),
  setAgentRunning:    (v) => set({ agentRunning: v }),
  clearAgentMessages: () => set({ agentMessages: [] }),
}))
