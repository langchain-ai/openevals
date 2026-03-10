import * as ls from "langsmith/vitest";

import { expect } from "vitest";

import { createLLMAsJudge } from "../llm.js";
import {
  CONCISENESS_PROMPT,
  CORRECTNESS_PROMPT,
  HALLUCINATION_PROMPT,
} from "../prompts/quality/index.js";

// ── CONCISENESS_PROMPT ────────────────────────────────────────────────────────

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
        model: "openai:gpt-5-mini",
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
        model: "openai:gpt-5-mini",
      });

      const evalResult = await llmAsJudge({ inputs, outputs });
      expect(evalResult.score).toBeFalsy();
    }
  );
});

// ── CORRECTNESS_PROMPT ────────────────────────────────────────────────────────

ls.describe("LLM Judge Correctness", () => {
  ls.test(
    "should pass correctness check for correct answer",
    {
      inputs: {
        question: "Who was the first president of the United States?",
      },
      referenceOutputs: {
        answer: "George Washington",
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = { answer: "George Washington" };

      const llmAsJudge = createLLMAsJudge({
        prompt: CORRECTNESS_PROMPT,
        feedbackKey: "correctness",
        model: "openai:gpt-5-mini",
      });

      await expect(llmAsJudge({ inputs, outputs })).rejects.toThrow();
      const evalResult = await llmAsJudge({
        inputs,
        outputs,
        referenceOutputs,
      });
      expect(evalResult.score).toBeTruthy();
    }
  );

  ls.test(
    "should fail correctness check for incorrect answer",
    {
      inputs: {
        question: "Who was the first president of the United States?",
      },
      referenceOutputs: {
        answer: "George Washington",
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = {
        answer: "John Adams",
      };

      const llmAsJudge = createLLMAsJudge({
        prompt: CORRECTNESS_PROMPT,
        feedbackKey: "correctness",
        model: "openai:gpt-5-mini",
      });

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
        referenceOutputs,
      });
      expect(evalResult.score).toBeFalsy();
    }
  );
});

// ── HALLUCINATION_PROMPT ──────────────────────────────────────────────────────

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
        model: "openai:gpt-5-mini",
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
        model: "openai:gpt-5-mini",
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
