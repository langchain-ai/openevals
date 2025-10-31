import * as ls from "langsmith/vitest";
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
});
