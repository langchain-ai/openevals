import { Sandbox } from "@e2b/code-interpreter";
import * as ls from "langsmith/vitest";

import { beforeAll, expect } from "vitest";

import { createE2BTypeScriptEvaluator } from "../typescript.js";

ls.describe("E2B TypeScript Evaluator", () => {
  let sandbox: Sandbox;
  beforeAll(async () => {
    sandbox = await Sandbox.create();
  });

  ls.test.each([
    {
      inputs: {
        question: "Generate a function that returns the sum of two numbers.",
      },
      outputs: {
        content: `import { add } from 'lodash';

function sum(a: number, b: number): number {
  return add(a, b);
}`,
      },
      expectedScore: true,
    },
    {
      inputs: {
        question: "Generate a bad LangGraph app.",
      },
      outputs: {
        content: `import { StateGraph } from '@langchain/langgraph';

await StateGraph.invoke({})`,
      },
      expectedScore: false,
    },
    {
      inputs: {
        question: "Generate a LangGraph app.",
      },
      outputs: {
        content: `import { Annotation, StateGraph } from '@langchain/langgraph';

const StateAnnotation = Annotation.Root({
  joke: Annotation<string>,
  topic: Annotation<string>,
});

const graph = new StateGraph(StateAnnotation)
  .addNode("joke", () => ({}))
  .compile();
  
await graph.invoke({
  joke: "foo",
  topic: "history",
});
`,
      },
      expectedScore: true,
    },
  ])("should respect imports", async ({ inputs, outputs, expectedScore }) => {
    const evaluator = createE2BTypeScriptEvaluator({
      sandbox,
    });
    const evalResult = await evaluator({ inputs, outputs });
    expect(evalResult.score).toBe(expectedScore);
  });
});
