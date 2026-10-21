import { describe, expect, test } from "vitest";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { RunnableLambda } from "@langchain/core/runnables";

import { createLLMAsJudge } from "../llm.js";

class StructuredOutputRecordingChatModel extends FakeListChatModel {
  recordedOptions: unknown[] = [];

  constructor() {
    super({ responses: [] });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  withStructuredOutput(_schema: any, config?: any): any {
    this.recordedOptions.push(config);
    return RunnableLambda.from(async () => ({
      score: true,
      reasoning: "equal",
    }));
  }
}

describe("createLLMAsJudge structuredOutputOptions", () => {
  test.each([
    [undefined, undefined],
    [{ method: "jsonSchema" }, { method: "jsonSchema" }],
  ])(
    "forwards %o to withStructuredOutput",
    async (structuredOutputOptions, expected) => {
      const judge = new StructuredOutputRecordingChatModel();
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        judge,
        structuredOutputOptions,
      });
      const result = await evaluator({
        inputs: { a: 1 },
        outputs: { a: 1 },
      });
      expect(result.score).toBe(true);
      expect(judge.recordedOptions).toEqual([expected]);
    }
  );
});
