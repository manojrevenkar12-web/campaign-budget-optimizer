import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { App } from "./App";

function mockFetch(routes: Record<string, unknown>) {
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url.split("?")[0]}`;
    const body = routes[key];
    return new Response(JSON.stringify(body ?? { detail: "not found" }), { status: body ? 200 : 404 });
  }));
}

const rec = {
  recommendation_id: "r1", campaign_id: "c1", day: "2026-10-21", policy: "damped_thompson", budget: 3000,
  expected_conversions: 96, expected_cost_per_conversion: 31.25, model_state_as_of: "2026-10-20T06:00:00Z",
  guardrails: { min_spend: 10, min_share: 0.03, max_change: 0.3, relaxed: false },
  ad_sets: [
    { ad_set_id: "c1-as0", platform: "meta", previous_spend: 1000, recommended_spend: 1300, expected_conversions: 50, conversions_80pct: [45, 55], marginal_cost_per_conversion: 60, at_lower_guardrail: false, at_upper_guardrail: true, cold_start: false, days_of_data: 60 },
    { ad_set_id: "c1-as1", platform: "tiktok", previous_spend: 2000, recommended_spend: 1700, expected_conversions: 46, conversions_80pct: [40, 52], marginal_cost_per_conversion: 61, at_lower_guardrail: false, at_upper_guardrail: false, cold_start: false, days_of_data: 60 },
  ],
};

test("plans a campaign and explains the split", async () => {
  sessionStorage.setItem("cbo.key", "k");
  mockFetch({
    "GET /v1/campaigns": { campaigns: [{ id: "c1", name: "Spring sale", daily_budget: 3000, ad_sets: 2, last_day: "2026-10-20" }] },
    "GET /v1/campaigns/c1": { id: "c1", name: "Spring sale", daily_budget: 3000, ad_sets: [{ id: "c1-as0", platform: "meta", name: "" }, { id: "c1-as1", platform: "tiktok", name: "" }], curves: [] },
    "GET /v1/campaigns/c1/creatives/at-risk": { creatives: [{ creative_id: "cr9", ad_set_id: "c1-as0", day: "2026-10-20", probability: 0.82, state: "at_risk_scored", model_version: "3" }] },
    "POST /v1/campaigns/c1/recommendations": rec,
  });
  render(<QueryClientProvider client={new QueryClient()}><App /></QueryClientProvider>);
  fireEvent.click(await screen.findByText("Spring sale"));
  fireEvent.click(await screen.findByRole("button", { name: "Recommend split" }));
  expect(await screen.findByText("capped at +30%")).toBeInTheDocument();
  expect(screen.getByText("+30%")).toBeInTheDocument();
  expect(screen.getByText("−15%")).toBeInTheDocument();
  expect(screen.getByText("82%")).toBeInTheDocument();
});
