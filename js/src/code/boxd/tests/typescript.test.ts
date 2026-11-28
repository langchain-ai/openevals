import type { Boxd } from "@boxd-sh/sdk";
import * as ls from "langsmith/vitest";

import { afterAll, beforeAll, expect } from "vitest";

import { createBoxdTypeScriptEvaluator } from "../typescript.js";
import { acquireTestMachine } from "./machine.js";

ls.describe("boxd TypeScript Evaluator", () => {
  let boxdClient: Boxd;
  let machineId: string;
  let release: () => Promise<void> = async () => {};
  beforeAll(async () => {
    if (!process.env.BOXD_API_KEY) {
      return;
    }
    ({ boxdClient, machineId, release } = await acquireTestMachine());
  }, 240_000);
  afterAll(async () => {
    await release();
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
    if (!process.env.BOXD_API_KEY) {
      return;
    }
    const evaluator = createBoxdTypeScriptEvaluator({
      boxdClient,
      machineId,
      sandboxProjectDirectory: "openevals-typescript",
    });
    const evalResult = await evaluator({ inputs, outputs });
    expect(evalResult.score).toBe(expectedScore);
  });
});
