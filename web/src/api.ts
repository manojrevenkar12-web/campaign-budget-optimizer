export interface CampaignRow { id: string; name: string; daily_budget: number; ad_sets: number; last_day: string | null }
export interface AdSet { id: string; platform: string; name: string }
export interface Curve { ad_set_id: string; platform: string; spend: number[]; conversions: number[] }
export interface Campaign { id: string; name: string; daily_budget: number; ad_sets: AdSet[]; curves: Curve[] }
export interface Line {
  ad_set_id: string; platform: string; previous_spend: number | null; recommended_spend: number;
  expected_conversions: number; conversions_80pct: [number, number]; marginal_cost_per_conversion: number | null;
  at_lower_guardrail: boolean; at_upper_guardrail: boolean; cold_start: boolean; days_of_data: number;
}
export interface Recommendation {
  recommendation_id: string; campaign_id: string; day: string; policy: string; budget: number;
  expected_conversions: number; expected_cost_per_conversion: number | null; model_state_as_of: string | null;
  guardrails: { min_spend: number; min_share: number; max_change: number; relaxed: boolean }; ad_sets: Line[];
}
export interface AtRisk { creative_id: string; ad_set_id: string; day: string; probability: number; state: string; model_version: string }

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

const KEY = "cbo.key";
export const session = {
  get: () => { try { return sessionStorage.getItem(KEY); } catch { return null; } },
  set: (k: string | null) => { try { if (k) sessionStorage.setItem(KEY, k); else sessionStorage.removeItem(KEY); } catch { /* private mode */ } },
};

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: { Authorization: `Bearer ${session.get() ?? ""}`, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new ApiError(r.status, typeof data.detail === "string" ? data.detail : r.statusText);
  return data as T;
}

export const api = {
  campaigns: () => call<{ campaigns: CampaignRow[] }>("GET", "/v1/campaigns"),
  campaign: (id: string) => call<Campaign>("GET", `/v1/campaigns/${id}`),
  recommend: (id: string, day: string, budget?: number) => call<Recommendation>("POST", `/v1/campaigns/${id}/recommendations`, { day, ...(budget ? { budget } : {}) }),
  atRisk: (id: string) => call<{ creatives: AtRisk[] }>("GET", `/v1/campaigns/${id}/creatives/at-risk?threshold=0.5`),
};
