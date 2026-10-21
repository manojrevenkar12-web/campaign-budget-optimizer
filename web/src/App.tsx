import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, ApiError, session } from "./api";
import { Curves } from "./Curves";
import { eur, num, pctChange, signed } from "./format";

const tomorrow = () => new Date(Date.now() + 86_400_000).toISOString().slice(0, 10);

export function App() {
  const [key, setKey] = useState(session.get());
  const [selected, setSelected] = useState<string | null>(null);
  if (!key) return <SignIn onKey={(k) => { session.set(k); setKey(k); }} />;
  return (
    <div className="app">
      <header className="top">
        <span className="brand">Budget optimizer</span>
        <button className="link" onClick={() => { session.set(null); setKey(null); }}>Sign out</button>
      </header>
      <div className="layout">
        <CampaignList selected={selected} onSelect={setSelected} />
        <main>{selected ? <CampaignView key={selected} id={selected} /> : <p className="empty">Choose a campaign to plan tomorrow's budget split.</p>}</main>
      </div>
    </div>
  );
}

function SignIn({ onKey }: { onKey: (k: string) => void }) {
  const [k, setK] = useState("");
  return (
    <form className="signin" onSubmit={(e) => { e.preventDefault(); if (k.trim()) onKey(k.trim()); }}>
      <h1>Budget optimizer</h1>
      <p>Paste your API key. It stays in this browser tab only.</p>
      <input aria-label="API key" value={k} onChange={(e) => setK(e.target.value)} placeholder="cbo_…" autoFocus />
      <button className="primary">Continue</button>
    </form>
  );
}

function CampaignList({ selected, onSelect }: { selected: string | null; onSelect: (id: string) => void }) {
  const q = useQuery({ queryKey: ["campaigns"], queryFn: api.campaigns });
  if (q.error) return <aside className="side"><p className="error">{q.error instanceof ApiError && q.error.status === 401 ? "That API key was not accepted." : String(q.error)}</p></aside>;
  return (
    <aside className="side" aria-label="Campaigns">
      <h2>Campaigns</h2>
      <ul>
        {q.data?.campaigns.map((c) => (
          <li key={c.id}>
            <button className={c.id === selected ? "row active" : "row"} onClick={() => onSelect(c.id)}>
              <span>{c.name || c.id}</span>
              <span className="muted">{eur(c.daily_budget)}/day · {c.ad_sets} ad sets</span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}

function CampaignView({ id }: { id: string }) {
  const camp = useQuery({ queryKey: ["campaign", id], queryFn: () => api.campaign(id) });
  const risk = useQuery({ queryKey: ["risk", id], queryFn: () => api.atRisk(id) });
  const [day, setDay] = useState(tomorrow());
  const [budget, setBudget] = useState<string>("");
  const rec = useMutation({ mutationFn: () => api.recommend(id, day, budget ? Number(budget) : undefined) });
  if (!camp.data) return null;
  const r = rec.data;
  const before = r?.ad_sets.reduce((s, l) => s + (l.previous_spend ?? 0), 0);
  return (
    <div className="stack">
      <div className="head">
        <div>
          <h1>{camp.data.name || id}</h1>
          <p className="muted">{camp.data.ad_sets.length} ad sets · {eur(camp.data.daily_budget)} daily budget</p>
        </div>
        <form className="controls" onSubmit={(e) => { e.preventDefault(); rec.mutate(); }}>
          <label>Day<input type="date" value={day} onChange={(e) => setDay(e.target.value)} /></label>
          <label>Budget<input inputMode="decimal" placeholder={String(camp.data.daily_budget)} value={budget} onChange={(e) => setBudget(e.target.value)} /></label>
          <button className="primary" disabled={rec.isPending}>Recommend split</button>
        </form>
      </div>
      {rec.error && <p className="error" role="alert">{String(rec.error)}</p>}
      {r && (
        <section className="panel" aria-label="Recommendation">
          <div className="kpis">
            <div><b>{num(r.expected_conversions, 0)}</b><span>expected conversions</span></div>
            <div><b>{eur(r.expected_cost_per_conversion, 2)}</b><span>expected cost per conversion</span></div>
            <div><b>{before ? eur(before) : "—"} → {eur(r.budget)}</b><span>spend yesterday → planned</span></div>
          </div>
          <table>
            <thead><tr><th>Ad set</th><th className="r">Yesterday</th><th className="r">Planned</th><th className="r">Change</th><th className="r">Conversions (80%)</th><th className="r">Next conversion costs</th><th /></tr></thead>
            <tbody>
              {r.ad_sets.map((l) => (
                <tr key={l.ad_set_id}>
                  <td>{l.ad_set_id}<span className="platform">{l.platform}</span></td>
                  <td className="r">{eur(l.previous_spend)}</td>
                  <td className="r"><b>{eur(l.recommended_spend)}</b></td>
                  <td className={`r ${(pctChange(l.previous_spend, l.recommended_spend) ?? 0) >= 0 ? "up" : "down"}`}>{signed(pctChange(l.previous_spend, l.recommended_spend))}</td>
                  <td className="r">{num(l.expected_conversions, 1)} <span className="muted">({num(l.conversions_80pct[0], 0)}–{num(l.conversions_80pct[1], 0)})</span></td>
                  <td className="r">{eur(l.marginal_cost_per_conversion)}</td>
                  <td>
                    {l.cold_start && <span className="badge">new, learning</span>}
                    {l.at_upper_guardrail && <span className="badge">{r.guardrails.relaxed ? "at widened limit" : `capped at +${Math.round(r.guardrails.max_change * 100)}%`}</span>}
                    {l.at_lower_guardrail && <span className="badge">{r.guardrails.relaxed ? "at widened floor" : "floor"}</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="note">Budgets move at most {Math.round(r.guardrails.max_change * 100)}% a day so the ad platforms' delivery learning is not reset.{r.guardrails.relaxed && " This budget is outside what that allows from yesterday's spend, so every ad set's limit was widened in proportion."} "Next conversion costs" is the marginal cost at the planned spend: the split evens these out as far as the limits allow, apart from a small exploration share that keeps testing ad sets the model is still unsure about.</p>
        </section>
      )}
      <section className="panel">
        <h2>What each ad set returns for its spend</h2>
        {camp.data.curves.length ? <Curves curves={camp.data.curves} lines={r?.ad_sets} /> : <p className="muted">No fitted curves yet; they appear after the first stats delivery is processed.</p>}
      </section>
      <section className="panel">
        <h2>Creatives to replace</h2>
        {risk.data?.creatives.length ? (
          <table>
            <thead><tr><th>Creative</th><th>Ad set</th><th className="r">Likelihood of wearing out within 7 days</th></tr></thead>
            <tbody>{risk.data.creatives.map((c) => (
              <tr key={c.creative_id}><td>{c.creative_id}</td><td>{c.ad_set_id}</td><td className="r">{c.state === "already_fatigued" ? "already 20%+ below its best CTR" : `${Math.round(c.probability * 100)}%`}</td></tr>
            ))}</tbody>
          </table>
        ) : <p className="muted">No creative is likely to wear out in the next week.</p>}
      </section>
    </div>
  );
}
