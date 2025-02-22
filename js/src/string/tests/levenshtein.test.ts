import * as ls from "langsmith/vitest";
import { evaluate } from "langsmith/evaluation";
import { expect } from "vitest";

import { levenshteinDistance } from "../levenshtein.js";

ls.describe("Levenshtein Distance Tests", () => {
  ls.test("levenshtein with identical values", { inputs: {} }, async () => {
    const outputs = { a: 1, b: 2 };
    const referenceOutputs = { a: 1, b: 2 };
    const res = await levenshteinDistance({ outputs, referenceOutputs });

    expect(res.key).toBe("levenshtein_distance");
    expect(res.score).toBe(1.0);
  });

  ls.test("levenshtein with different values", { inputs: {} }, async () => {
    const outputs = { a: 1, b: 2 };
    const referenceOutputs = { a: 1, b: 3 };
    const res = await levenshteinDistance({ outputs, referenceOutputs });

    expect(res.key).toBe("levenshtein_distance");
    expect(res.score).toBeLessThan(1.0);
  });

  ls.test(
    "test works with evaluate",
    {
      inputs: { dataset: "exact match" },
    },
    async ({ inputs }) => {
      const evaluator = levenshteinDistance;
      const result = await evaluate(
        (inputs) => inputs,
        {
          data: inputs.dataset,
          evaluators: [
            evaluator
          ]
        }
      )
      expect(result).toBeDefined();
      expect(result.results.length).toBeGreaterThan(0);
      expect(result.results[0].evaluationResults.results[0].score).toBeDefined();
    }
  );
});
