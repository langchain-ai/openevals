import * as ls from "langsmith/vitest";
import { expect, expectTypeOf, beforeAll } from "vitest";
import { OpenAI } from "openai";
import { ChatOpenAI } from "@langchain/openai";

import { createLLMAsJudge } from "../llm.js";
import * as hub from "langchain/hub";
import { HumanMessage } from "@langchain/core/messages";

import { z } from "zod";
import { ChatPromptTemplate, HumanMessagePromptTemplate, StructuredPrompt } from "@langchain/core/prompts";
import { Client } from "langsmith";

ls.describe("llm as judge", () => {
  beforeAll(async () => {
    // Setup required prompts in LangChain Hub before running tests
    const client = new Client();

    // Create test-equality prompt
    const testEqualityPrompt = ChatPromptTemplate.fromMessages([
      ["system", "You are an expert LLM as judge."],
      ["human", "Are these two equal? {inputs} {outputs}"],
    ]);

    try {
      await client.pushPrompt("test-equality", { object: testEqualityPrompt });
      console.log("Created test-equality prompt");
    } catch (error) {
      console.log(`test-equality prompt may already exist: ${error}`);
    }

    // Create equality-1-message prompt
    const equality1MessagePrompt = ChatPromptTemplate.fromMessages([
      ["human", "Are these two equal? {inputs} {outputs}"],
    ]);

    try {
      await client.pushPrompt("equality-1-message", {
        object: equality1MessagePrompt,
      });
      console.log("Created equality-1-message prompt");
    } catch (error) {
      console.log(`equality-1-message prompt may already exist: ${error}`);
    }

    // Create simple-equality-structured prompt
    const structuredEqualityPrompt = new StructuredPrompt({
      inputVariables: ["inputs", "outputs"],
      promptMessages: [
        HumanMessagePromptTemplate.fromTemplate(
          `Are these equal?

<item1>
{inputs}
</item1>

<item2>
{outputs}
</item2>`,
        ),
      ],
      schema: {
        title: "score",
        description: "Get a score",
        type: "object",
        properties: {
          equality: {
            type: "boolean",
            description: "Whether the two items are equal",
          },
          justification: {
            type: "string",
            description: "Justification for your decision above",
          },
        },
        required: ["equality", "justification"],
        strict: true,
        additionalProperties: false,
      },
    });

    try {
      await client.pushPrompt("simple-equality-structured", {
        object: structuredEqualityPrompt,
      });
      console.log("Created simple-equality-structured prompt");
    } catch (error) {
      console.log(
        `simple-equality-structured prompt may already exist: ${error}`
      );
    }
  });

  ls.test(
    "prompt hub prompt",
    {
      inputs: { a: 1, b: 2 },
    },
    async ({ inputs }) => {
      const outputs = { a: 1, b: 2 };
      const client = new OpenAI();
      const evaluator = createLLMAsJudge({
        prompt: await hub.pull("equality-1-message"),
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
        prompt: await hub.pull("simple-equality-structured"),
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
