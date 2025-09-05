import * as ls from "langsmith/vitest";
import { evaluate } from "langsmith/evaluation";
import { expect, test } from "vitest";
import { exactMatch } from "../exact.js";

ls.describe("exact match", () => {
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
    "matches different order of inputs and outputs",
    {
      inputs: {},
    },
    async () => {
      const outputs = { a: 1, b: 2 };
      const referenceOutputs = { b: 2, a: 1 };
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
