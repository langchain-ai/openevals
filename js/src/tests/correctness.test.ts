import * as ls from "langsmith/vitest";

import { expect } from "vitest";

import { createLLMAsJudge } from "../llm.js";
import { CORRECTNESS_PROMPT } from "../prompts/correctness.js";

ls.describe("LLM Judge Correctness", () => {
  ls.test(
    "should pass correctness check for correct answer",
    {
      inputs: {
        question: "Who was the first president of the United States?",
      },
    },
    async ({ inputs }) => {
      const outputs = { answer: "George Washington" };

      const llmAsJudge = createLLMAsJudge({
        prompt: CORRECTNESS_PROMPT,
        feedbackKey: "correctness",
        model: "openai:o3-mini",
      });

      const evalResult = await llmAsJudge({ inputs, outputs });
      expect(evalResult.score).toBeTruthy();
    }
  );

  ls.test(
    "should fail correctness check for incorrect answer",
    {
      inputs: {
        question: "Who was the first president of the United States?",
      },
    },
    async ({ inputs }) => {
      const outputs = {
        answer: "John Adams",
      };

      const llmAsJudge = createLLMAsJudge({
        prompt: CORRECTNESS_PROMPT,
        feedbackKey: "correctness",
        model: "openai:o3-mini",
      });

      const evalResult = await llmAsJudge({ inputs, outputs });
      expect(evalResult.score).toBeFalsy();
    }
  );

  ls.test(
    "should pass correctness check for correct answer with reference outputs",
    {
      inputs: {
        question:
          "Who was the first president of the Star Republic of Oiewjoie?",
      },
    },
    async ({ inputs }) => {
      const outputs = { answer: "Bzkeoei Ahbeijo" };
      const referenceOutputs = { answer: "Bzkeoei Ahbeijo" };
      const llmAsJudge = createLLMAsJudge({
        prompt: CORRECTNESS_PROMPT,
        feedbackKey: "correctness",
        model: "openai:o3-mini",
      });

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
        referenceOutputs,
      });
      expect(evalResult.score).toBeTruthy();
    }
  );
});
