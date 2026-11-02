import { describe, expect, test } from "vitest";

import { createJsonMatchEvaluator } from "../json/match.js";

describe("json match with nothing to score", () => {
  test.each([
    { options: { aggregator: "average" }, outputs: {}, referenceOutputs: {} },
    {
      options: { aggregator: "average", excludeKeys: ["a"] },
      outputs: { a: 1 },
      referenceOutputs: { a: 2 },
    },
    {
      options: { aggregator: "average", listAggregator: "average" },
      outputs: [],
      referenceOutputs: [],
    },
    {
      options: { aggregator: "all", listAggregator: "average" },
      outputs: [],
      referenceOutputs: [],
    },
    {
      options: {
        aggregator: "average",
        listAggregator: "average",
        listMatchMode: "superset",
      },
      outputs: [{ a: 1 }],
      referenceOutputs: [],
    },
  ] as const)(
    "scores $options as a match",
    async ({ options, outputs, referenceOutputs }) => {
      const evaluator = createJsonMatchEvaluator({
        ...options,
        excludeKeys: "excludeKeys" in options ? [...options.excludeKeys] : [],
      });
      const result = await evaluator({
        outputs: outputs as Record<string, unknown>,
        referenceOutputs: referenceOutputs as Record<string, unknown>,
      });
      expect(result).toHaveLength(1);
      expect(result[0].score).toBe(1);
    }
  );
});
