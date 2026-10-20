import { niceMax, pctChange, signed } from "./format";

test("niceMax rounds up to a readable axis end", () => {
  expect(niceMax(87)).toBe(100);
  expect(niceMax(1234)).toBe(2000);
  expect(niceMax(0.3)).toBe(0.5);
});

test("spend changes are signed and handle new ad sets", () => {
  expect(signed(pctChange(1000, 1250))).toBe("+25%");
  expect(signed(pctChange(1000, 700))).toBe("−30%");
  expect(signed(pctChange(null, 500))).toBe("new");
});
