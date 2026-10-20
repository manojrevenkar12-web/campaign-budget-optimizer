from cbo.sim.evaluate import evaluate


def test_policies_run_and_the_oracle_bounds_them():
    res, s = evaluate(n_campaigns=4, days=20, policies=["even_split", "rule_of_thumb", "damped_thompson", "oracle"])
    assert set(res["policy"]) == {"even_split", "rule_of_thumb", "damped_thompson", "oracle"}
    best = s["oracle"]["expected_conversions"]
    assert all(v["expected_conversions"] <= best * 1.01 for v in s.values())
    # every policy spends the same budget
    spend = res.groupby("policy")["spend"].sum()
    assert spend.max() / spend.min() < 1.001
