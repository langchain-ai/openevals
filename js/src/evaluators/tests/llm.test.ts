import * as ls from "langsmith/vitest";
import { OpenAI } from "openai";

import { expect } from "vitest";
import { createLLMAsJudge } from "../llm.js";
import { ChatOpenAI } from "@langchain/openai";

ls.describe("llm as judge", () => {
  ls.test(
    "llm as judge OpenAI",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const client = new OpenAI();
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        judge: client,
        model: "gpt-4o-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.score).toBeDefined();
      expect(result.comment).toBeDefined();
    }
  );

  ls.test(
    "llm as judge OpenAI no reasoning",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const client = new OpenAI();
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        judge: client,
        model: "gpt-4o-mini",
        useReasoning: false,
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.score).toBeDefined();
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test(
    "llm as judge OpenAI not equal",
    {
      inputs: { a: 1, b: 3 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const client = new OpenAI();
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        judge: client,
        model: "gpt-4o-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.score).toBe(false);
      expect(result.comment).toBeDefined();
    }
  );

  ls.test(
    "llm as judge OpenAI not equal continuous",
    {
      inputs: { a: 1, b: 3 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const client = new OpenAI();
      const evaluator = createLLMAsJudge({
        prompt: "How equal are these two? {inputs} {outputs}",
        judge: client,
        model: "gpt-4o-mini",
        continuous: true,
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.score).toBeGreaterThan(0);
      expect(result.score).toBeLessThan(1);
      expect(result.comment).toBeDefined();
    }
  );

  ls.test(
    "llm as judge LangChain",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        judge: new ChatOpenAI({ model: "gpt-4o-mini" }),
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.score).toBe(true);
      expect(result.comment).toBeDefined();
    }
  );

  ls.test(
    "llm as judge init_chat_model",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        model: "openai:gpt-4o-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.score).toBe(true);
      expect(result.comment).toBeDefined();
    }
  );

  ls.test(
    "llm as judge few shot examples",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const evaluator = createLLMAsJudge({
        prompt: "Are these two foo? {inputs} {outputs}",
        fewShotExamples: [
          {
            inputs: { a: 1, b: 2 },
            outputs: { a: 1, b: 2 },
            score: 0.0,
          },
          { inputs: { a: 1, b: 3 }, outputs: { a: 1, b: 2 }, score: 1.0 },
        ],
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.score).toBe(false);
      expect(result.comment).toBeDefined();
    }
  );
});
