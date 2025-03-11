import * as ls from "langsmith/vitest";

import { expect } from "vitest";

import { createLLMAsJudge } from "../llm.js";
import { HALLUCINATION_PROMPT } from "../prompts/hallucination.js";

ls.describe("LLM Judge Hallucination", () => {
  ls.test(
    "should pass hallucination check for non-hallucinated answer",
    {
      inputs: {
        question:
          "Who was the first president of the Star Republic of Oiewjoie?",
      },
    },
    async ({ inputs }) => {
      const outputs = { answer: "Bzkeoei Ahbeijo" };
      const context =
        "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo.";

      const llmAsJudge = createLLMAsJudge({
        prompt: HALLUCINATION_PROMPT,
        feedbackKey: "hallucination",
        model: "openai:o3-mini",
      });

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
        context,
        referenceOutputs: "",
      });
      expect(evalResult.score).toBeTruthy();
    }
  );

  ls.test(
    "should fail hallucination check for hallucinated answer",
    {
      inputs: {
        question:
          "Who was the first president of the Star Republic of Oiewjoie?",
      },
    },
    async ({ inputs }) => {
      const outputs = {
        answer: "John Adams",
      };
      const context =
        "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo.";

      const llmAsJudge = createLLMAsJudge({
        prompt: HALLUCINATION_PROMPT,
        feedbackKey: "hallucination",
        model: "openai:o3-mini",
      });

      await expect(llmAsJudge({ inputs, outputs })).rejects.toThrow();

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
        context,
        referenceOutputs: "",
      });
      expect(evalResult.score).toBeFalsy();
    }
  );
});
