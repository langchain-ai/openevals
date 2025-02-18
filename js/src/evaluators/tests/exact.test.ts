import * as ls from "langsmith/vitest";

import { expect } from "vitest";
import { exactMatch } from "../exact.js";

ls.describe("exact matcher", () => {
  ls.test(
    "matches identical inputs and outputs",
    {
      inputs: {},
    },
    async () => {
      const outputs = { a: 1, b: 2 };
      const referenceOutputs = { a: 1, b: 2 };
      expect(await exactMatch({ outputs, referenceOutputs })).toEqual({
        key: "exact_match",
        score: true,
      });
    }
  );

  ls.test(
    "fails with different values",
    {
      inputs: {},
    },
    async () => {
      const outputs = { a: 1, b: 2 };
      const referenceOutputs = { a: 1, b: 3 };
      expect(await exactMatch({ outputs, referenceOutputs })).toEqual({
        key: "exact_match",
        score: false,
      });
    }
  );
});
