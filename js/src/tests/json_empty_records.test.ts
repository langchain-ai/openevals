import { describe, expect, test } from "vitest";

import { createJsonMatchEvaluator } from "../json/match.js";

type ListMatchMode = "superset" | "subset" | "same_elements" | "ordered";

// Unmatched empty-object list items have no fields of their own, but must
// still be scored as unmatched records instead of silently vanishing.
const cases: [ListMatchMode, unknown[], unknown[], number][] = [
  ["same_elements", [{ total: 6 }, {}], [{ total: 6 }], 0],
  ["same_elements", [{ total: 6 }], [{ total: 6 }, {}], 0],
  ["same_elements", [{ total: 6 }, {}], [{ total: 6 }, {}], 1],
  ["same_elements", [{}], [], 0],
  ["same_elements", [], [{}], 0],
  ["same_elements", [{}], [{}], 1],
  ["superset", [{ total: 6 }], [{ total: 6 }, {}], 0],
  ["superset", [{ total: 6 }, {}], [{ total: 6 }], 1],
  ["subset", [{ total: 6 }, {}], [{ total: 6 }], 0],
  ["subset", [{ total: 6 }], [{ total: 6 }, {}], 1],
  ["ordered", [{ total: 6 }, {}], [{ total: 6 }], 0],
  ["ordered", [{ total: 6 }], [{ total: 6 }, {}], 0],
  ["ordered", [{ total: 6 }, {}], [{ total: 6 }, {}], 1],
];

describe("json match unmatched empty records", () => {
  test.each(cases)(
    "%s: %j vs %j scores %d",
    async (listMatchMode, outputs, referenceOutputs, expected) => {
      const evaluator = createJsonMatchEvaluator({
        aggregator: "all",
        listMatchMode,
      });
      const result = await evaluator({ outputs, referenceOutputs });
      const first = Array.isArray(result) ? result[0] : result;
      expect(first.key).toBe("json_match:all");
      expect(first.score).toBe(expected);
    }
  );
});
