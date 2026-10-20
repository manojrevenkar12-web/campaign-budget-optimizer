import { eur, niceMax, num } from "./format";
import type { Curve, Line } from "./api";

const COLORS = ["#2d5bd7", "#d9480f", "#2b8a3e", "#ae3ec9", "#0c8599", "#e8590c", "#5c7cfa", "#a61e4d"];

/** Posterior-mean conversions vs daily spend per ad set, with today's recommended spend marked. */
export function Curves({ curves, lines }: { curves: Curve[]; lines?: Line[] }) {
  const W = 560, H = 260, L = 48, B = 34, T = 12, R = 12;
  const xMax = niceMax(Math.max(...curves.flatMap((c) => c.spend)));
  const yMax = niceMax(Math.max(...curves.flatMap((c) => c.conversions)));
  const x = (v: number) => L + (v / xMax) * (W - L - R);
  const y = (v: number) => H - B - (v / yMax) * (H - B - T);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <figure className="chart" aria-label="Response curves">
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t * yMax)} y2={y(t * yMax)} className="grid" />
            <text x={L - 6} y={y(t * yMax) + 4} textAnchor="end" className="tick">{num(t * yMax)}</text>
            <text x={x(t * xMax)} y={H - B + 16} textAnchor="middle" className="tick">{eur(t * xMax)}</text>
          </g>
        ))}
        {curves.map((c, i) => {
          const d = c.spend.map((s, j) => `${j ? "L" : "M"}${x(s).toFixed(1)},${y(c.conversions[j]).toFixed(1)}`).join(" ");
          const line = lines?.find((l) => l.ad_set_id === c.ad_set_id);
          return (
            <g key={c.ad_set_id}>
              <path d={d} fill="none" stroke={COLORS[i % COLORS.length]} strokeWidth={2} />
              {line && <circle cx={x(line.recommended_spend)} cy={y(line.expected_conversions)} r={4.5} fill={COLORS[i % COLORS.length]} stroke="#fff" strokeWidth={1.5} />}
            </g>
          );
        })}
      </svg>
      <figcaption className="legend">
        {curves.map((c, i) => <span key={c.ad_set_id}><i style={{ background: COLORS[i % COLORS.length] }} />{c.ad_set_id.split("-").pop()} · {c.platform}</span>)}
        <span className="muted">Dots: today's recommended spend</span>
      </figcaption>
    </figure>
  );
}
