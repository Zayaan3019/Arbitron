import { useState } from 'react'
import { AlertTriangle, ChevronDown, ChevronRight } from 'lucide-react'
import type { DataCaveats } from '../api'

/**
 * Renders the `data_caveats` block that every numeric API response carries.
 *
 * The built-in universes are today's index members, not point-in-time
 * membership, so every backtest run on them is upward-biased. That limitation
 * is stated in the README, in the API response, and — here — on screen next to
 * the number it applies to, because a caveat only in the docs is a caveat
 * nobody reads.
 */
export default function DataCaveatNotice({ caveats }: { caveats?: DataCaveats }) {
  const [open, setOpen] = useState(false)
  if (!caveats) return null

  const hasDetails = (caveats.details?.length ?? 0) > 0

  return (
    <div className="card border border-amber-800/40 bg-amber-950/20">
      <button
        onClick={() => hasDetails && setOpen((v) => !v)}
        className="w-full flex items-start gap-2.5 text-left"
        disabled={!hasDetails}
      >
        <AlertTriangle size={14} className="text-amber-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <p className="text-[11px] font-semibold text-amber-300 mb-0.5 flex items-center gap-1">
            Data limitation
            {hasDetails && (open
              ? <ChevronDown size={12} className="text-amber-500" />
              : <ChevronRight size={12} className="text-amber-500" />)}
          </p>
          <p className="text-[11px] text-amber-200/80 leading-relaxed">{caveats.headline}</p>
        </div>
      </button>

      {open && hasDetails && (
        <ul className="mt-3 ml-6 space-y-1.5 list-disc list-outside marker:text-amber-600">
          {caveats.details.map((d, i) => (
            <li key={i} className="text-[11px] text-amber-200/70 leading-relaxed pl-1">{d}</li>
          ))}
          {caveats.remediation && (
            <li className="text-[11px] text-amber-200/90 leading-relaxed pl-1 font-medium">
              {caveats.remediation}
            </li>
          )}
        </ul>
      )}
    </div>
  )
}
