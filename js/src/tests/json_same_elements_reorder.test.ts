import { describe, expect, test } from "vitest";

import { createJsonMatchEvaluator } from "../json/match.js";

// same_elements must be order-invariant when the two lists contain the same
// whole records, even if one record's fields are a subset of another's.
const cases: [string, unknown[], unknown[], number][] = [
  [
    "reference reordered",
    [{ total: 1 }, { total: 1, count: 2 }],
    [{ total: 1, count: 2 }, { total: 1 }],
    1,
  ],
  [
    "output reordered",
    [{ total: 1, count: 2 }, { total: 1 }],
    [{ total: 1 }, { total: 1, count: 2 }],
    1,
  ],
  [
    "wrong count still fails",
    [{ total: 1 }, { total: 1, count: 3 }],
    [{ total: 1, count: 2 }, { total: 1 }],
    0,
  ],
  ["empty record reordered", [{}, { total: 6 }], [{ total: 6 }, {}], 1],
  [
    "nested values reordered",
    [{ a: { b: [1] } }, { a: { b: [1] }, c: 2 }],
    [{ a: { b: [1] }, c: 2 }, { a: { b: [1] } }],
    1,
  ],
];

describe("json match same_elements reordered records", () => {
  test.each(cases)("%s", async (_name, outputs, referenceOutputs, expected) => {
    const evaluator = createJsonMatchEvaluator({
      listMatchMode: "same_elements",
      aggregator: "all",
      listAggregator: "all",
    });
    const result = await evaluator({ outputs, referenceOutputs });
    expect(result).toEqual([
      expect.objectContaining({ key: "json_match:all", score: expected }),
    ]);
  });

  test("records differing only in excluded keys still match", async () => {
    const evaluator = createJsonMatchEvaluator({
      listMatchMode: "same_elements",
      aggregator: "all",
      listAggregator: "all",
      excludeKeys: ["id"],
    });
    const result = await evaluator({
      outputs: [
        { a: 1, id: 9 },
        { a: 1, b: 2 },
      ],
      referenceOutputs: [
        { a: 1, b: 2 },
        { a: 1, id: 7 },
      ],
    });
    expect(result).toEqual([
      expect.objectContaining({ key: "json_match:all", score: 1 }),
    ]);
  });

  test("partial matches keep existing greedy scoring", async () => {
    // Not a complete multiset match, so the existing matcher is used unchanged.
    const evaluator = createJsonMatchEvaluator({
      listMatchMode: "same_elements",
      aggregator: "average",
      listAggregator: "average",
    });
    const result = await evaluator({
      outputs: [
        { a: 1, d: 2 },
        { a: 1, b: 2 },
      ],
      referenceOutputs: [
        { a: 1, b: 2 },
        { a: 1, b: 2, c: 3 },
      ],
    });
    expect(result).toEqual([
      expect.objectContaining({ key: "json_match:average", score: 0.5 }),
    ]);
  });
});
