/**
 * Null-safe formatting helpers.
 *
 * Every numeric field the API returns is nullable, because quant metrics are
 * genuinely undefined in ordinary cases — a Sharpe on two observations, a
 * Calmar with no drawdown, a Deflated Sharpe with only one trial. The backend
 * sends `null` rather than coercing those to a plausible-looking 0.0, so the
 * UI must render "n/a" rather than crash on `.toFixed()` or print a fake zero.
 */

export type Num = number | null | undefined

export const NA = '—'

/** Is this a real, finite number we can show? */
export function isNum(v: Num): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

/** Fixed-decimal number, or an em-dash when not evaluable. */
export function fmt(v: Num, digits = 2): string {
  return isNum(v) ? v.toFixed(digits) : NA
}

/** Percentage from a fraction (0.0123 → "1.2%"), or an em-dash. */
export function pct(v: Num, digits = 1): string {
  return isNum(v) ? `${(v * 100).toFixed(digits)}%` : NA
}

/** Signed percentage, useful for returns. */
export function signedPct(v: Num, digits = 1): string {
  if (!isNum(v)) return NA
  const s = (v * 100).toFixed(digits)
  return v > 0 ? `+${s}%` : `${s}%`
}

/**
 * Tri-state comparison for colour coding: `undefined` when the value is not
 * evaluable, so a missing metric renders neutral instead of red.
 */
export function above(v: Num, threshold: number): boolean | undefined {
  return isNum(v) ? v > threshold : undefined
}

export function below(v: Num, threshold: number): boolean | undefined {
  return isNum(v) ? v < threshold : undefined
}

/** Coerce to a number for arithmetic, with an explicit fallback. */
export function orZero(v: Num, fallback = 0): number {
  return isNum(v) ? v : fallback
}

/** Integer with thousands separators. */
export function int(v: Num): string {
  return isNum(v) ? Math.round(v).toLocaleString() : NA
}
