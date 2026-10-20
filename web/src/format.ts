export const eur = (x: number | null | undefined, digits = 0) =>
  x === null || x === undefined ? "—" : new Intl.NumberFormat("en-GB", { style: "currency", currency: "EUR", maximumFractionDigits: digits }).format(x);
export const num = (x: number, digits = 0) => new Intl.NumberFormat("en-GB", { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(x);
export const pctChange = (from: number | null, to: number) => (from ? (to - from) / from : null);
export const signed = (x: number | null, digits = 0) => (x === null ? "new" : `${x > 0 ? "+" : x < 0 ? "−" : "±"}${Math.abs(x * 100).toFixed(digits)}%`);

/** Linear scale with "nice" rounded upper bound, for the hand-drawn SVG charts. */
export function niceMax(x: number): number {
  if (x <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(x));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= x) return m * p;
  return 10 * p;
}
