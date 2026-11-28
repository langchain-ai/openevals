import type { Boxd } from "@boxd-sh/sdk";
import * as ls from "langsmith/vitest";

import { afterAll, beforeAll, expect } from "vitest";

import { createBoxdExecutionEvaluator } from "../execution.js";
import { acquireTestMachine } from "./machine.js";

ls.describe("boxd Execution Evaluator", () => {
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
      expectedScore: false,
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
        question: "Generate an ok LangGraph app.",
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
      expectedScore: false,
    },
    {
      inputs: {
        question: "Generate a good LangGraph app.",
      },
      outputs: {
        content: `import { Annotation, StateGraph } from '@langchain/langgraph';

  const StateAnnotation = Annotation.Root({
    joke: Annotation<string>,
    topic: Annotation<string>,
  });

  const graph = new StateGraph(StateAnnotation)
    .addNode("test", () => ({}))
    .addEdge("__start__", "test")
    .compile();
    
  await graph.invoke({
    joke: "foo",
    topic: "history",
  });
  `,
      },
      expectedScore: true,
    },
  ])("should execute code", async ({ outputs, expectedScore }) => {
    if (!process.env.BOXD_API_KEY) {
      return;
    }
    const evaluator = createBoxdExecutionEvaluator({
      boxdClient,
      machineId,
      sandboxProjectDirectory: "openevals-execution",
    });
    const evalResult = await evaluator({ outputs });
    expect(evalResult.score).toBe(expectedScore);
  });
});
