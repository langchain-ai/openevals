import * as ls from "langsmith/vitest";

import { expect } from "vitest";
import { exactMatch } from "../exact.js";

ls.describe("exact matcher", () => {
  ls.test(
    "matches identical inputs and outputs",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      expect(await exactMatch({ inputs, outputs })).toEqual({
        key: "exact_match",
        score: true,
      });
    }
  );

  ls.test(
    "fails with different values",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 3 };
      expect(await exactMatch({ inputs, outputs })).toEqual({
        key: "exact_match",
        score: false,
      });
    }
  );
});
