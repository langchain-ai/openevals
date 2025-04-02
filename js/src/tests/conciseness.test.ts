import * as ls from "langsmith/vitest";

import { expect } from "vitest";

import { createLLMAsJudge } from "../llm.js";
import { CONCISENESS_PROMPT } from "../prompts/conciseness.js";

ls.describe("LLM Judge Conciseness", () => {
  ls.test(
    "should pass conciseness check for concise answer",
    {
      inputs: {
        question: "How is the weather in San Francisco?",
      },
    },
    async ({ inputs }) => {
      const outputs = { answer: "Sunny and 90 degrees." };

      const llmAsJudge = createLLMAsJudge({
        prompt: CONCISENESS_PROMPT,
        feedbackKey: "conciseness",
        model: "openai:gpt-4o-mini",
      });

      const evalResult = await llmAsJudge({ inputs, outputs });
      expect(evalResult.score).toBeTruthy();
    }
  );

  ls.test(
    "should fail conciseness check for verbose answer",
    {
      inputs: {
        question: "How is the weather in San Francisco?",
      },
    },
    async ({ inputs }) => {
      const outputs = {
        answer:
          "Thanks for asking! The current weather in San Francisco is sunny and 90 degrees.",
      };

      const llmAsJudge = createLLMAsJudge({
        prompt: CONCISENESS_PROMPT,
        feedbackKey: "conciseness",
        model: "openai:gpt-4o-mini",
      });

      const evalResult = await llmAsJudge({ inputs, outputs });
      expect(evalResult.score).toBeFalsy();
    }
  );
});
