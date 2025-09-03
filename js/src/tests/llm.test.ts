import * as ls from "langsmith/vitest";
import { expect, test, expectTypeOf } from "vitest";
import { evaluate } from "langsmith/evaluation";
import { OpenAI } from "openai";
import { ChatOpenAI } from "@langchain/openai";

import { createLLMAsJudge } from "../llm.js";
import * as hub from "langchain/hub";
import { HumanMessage } from "@langchain/core/messages";

import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";

ls.describe("llm as judge", () => {
  ls.test(
    "prompt hub prompt",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const client = new OpenAI();
      const evaluator = createLLMAsJudge({
        prompt: await hub.pull("langchain-ai/equality-1-message"),
        judge: client,
        model: "openai:gpt-4o-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      // assert TS type is not unknown
      expectTypeOf(result.score).toEqualTypeOf<number | boolean>();
      expect(result.score).toBeDefined();
      expect(result.comment).toBeDefined();
    }
  );

  ls.test(
    "prompt hub structured prompt",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const evaluator = createLLMAsJudge({
        prompt: await hub.pull("jacob/simple-equality-structured"),
        model: "openai:gpt-4o-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.equality).toBe(true);
      expect(result.justification).toBeDefined();
    }
  );

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
        model: "openai:o3-mini",
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
        model: "openai:o3-mini",
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
        model: "openai:o3-mini",
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
        prompt:
          "How equal are these two? If there are two props and one is equal and the other is not, return 0.5. {inputs} {outputs}",
        judge: client,
        model: "openai:o3-mini",
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
    "llm as judge LangChain messages array",
    {
      inputs: { messages: [new HumanMessage(JSON.stringify({ a: 1, b: 2 }))] },
    },
    async ({ inputs }) => {
      const outputs = {
        messages: [new HumanMessage(JSON.stringify({ a: 1, b: 3 }))],
      };
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        judge: new ChatOpenAI({ model: "gpt-4o-mini" }),
      });
      const result = await evaluator({
        inputs: inputs.messages,
        outputs: outputs.messages,
      });
      expect(result).toBeDefined();
      expect(result.score).toBe(false);
    }
  );

  ls.test(
    "llm as judge LangChain messages object",
    {
      inputs: { messages: [new HumanMessage(JSON.stringify({ a: 1, b: 2 }))] },
    },
    async ({ inputs }) => {
      const outputs = {
        messages: [new HumanMessage(JSON.stringify({ a: 1, b: 3 }))],
      };
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        judge: new ChatOpenAI({ model: "gpt-4o-mini" }),
      });
      const result = await evaluator({
        inputs,
        outputs,
      });
      expect(result).toBeDefined();
      expect(result.score).toBe(false);
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
        model: "openai:o3-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.score).toBe(false);
      expect(result.comment).toBeDefined();
    }
  );

  ls.test(
    "llm as judge with custom JSON schema output schema",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        outputSchema: z.toJSONSchema(
          z.object({
            equality: z.boolean(),
            justification: z.string(),
          })
        ),
        model: "openai:gpt-4o-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.equality).toBe(true);
      expect(result.justification).toBeDefined();
    }
  );

  ls.test(
    "llm as judge with OpenAI client and custom JSON schema output schema",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        outputSchema: z.toJSONSchema(
          z.object({
            equality: z.boolean(),
            justification: z.string(),
          })
        ),
        judge: new OpenAI(),
        model: "gpt-4o-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.equality).toBe(true);
      expect(result.justification).toBeDefined();
    }
  );

  ls.test(
    "llm as judge with custom Zod output schema",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const evaluator = createLLMAsJudge({
        prompt: "Are these two equal? {inputs} {outputs}",
        outputSchema: z.object({
          equality: z.boolean(),
          justification: z.string(),
        }),
        model: "openai:gpt-4o-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.equality).toBe(true);
      expect(result.justification).toBeDefined();
    }
  );

  test("test llm as judge works with evaluate", async () => {
    const evaluator = createLLMAsJudge({
      prompt: "Are these two foo? {inputs} {outputs}",
      model: "openai:o3-mini",
    });
    const result = await evaluate((inputs) => inputs, {
      data: "exact match",
      evaluators: [evaluator],
    });
    expect(result).toBeDefined();
    expect(result.results.length).toBeGreaterThan(0);
  }, 60000);

  ls.test(
    "llm as judge with mustache prompt",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const prompt = ChatPromptTemplate.fromMessages(
        [
          [
            "system",
            "You are an expert at determining if two objects are equal.",
          ],
          ["user", "Are these two equal? {{inputs}} {{outputs}}"],
        ],
        { templateFormat: "mustache" }
      );
      const evaluator = createLLMAsJudge({
        prompt,
        model: "openai:gpt-4o-mini",
      });
      const result = await evaluator({ inputs, outputs });
      expect(result).toBeDefined();
      expect(result.score).toBe(true);
    }
  );
});
