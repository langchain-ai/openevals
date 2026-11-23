import { describe, expect, it } from "vitest";

import { createTrajectoryMatchEvaluator } from "../match.js";

const toolCall = (args: Record<string, unknown>) => ({
  function: { name: "search", arguments: JSON.stringify(args) },
});

const trajectory = (...toolCalls: ReturnType<typeof toolCall>[]) => [
  { role: "user", content: "Find the flights." },
  { role: "assistant", content: "", tool_calls: toolCalls },
];

const narrow = toolCall({ origin: "SFO" });
const broad = toolCall({ origin: "SFO", destination: "JFK" });

describe.each(["strict", "unordered", "subset", "superset"] as const)(
  "tool call assignment in %s mode",
  (trajectoryMatchMode) => {
    it("does not let call order decide a superset args match", async () => {
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
        toolArgsMatchMode: "superset",
      });
      const result = await evaluator({
        outputs: trajectory(broad, narrow),
        referenceOutputs: trajectory(narrow, broad),
      });
      expect(result.score).toBe(true);
    });

    it("does not let call order decide a subset args match", async () => {
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
        toolArgsMatchMode: "subset",
      });
      const result = await evaluator({
        outputs: trajectory(narrow, broad),
        referenceOutputs: trajectory(broad, narrow),
      });
      expect(result.score).toBe(true);
    });
  }
);

describe.each(["strict", "unordered", "superset"] as const)(
  "reusing a tool call in %s mode",
  (trajectoryMatchMode) => {
    it("does not let one output call satisfy two reference calls", async () => {
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
        toolArgsMatchMode: "superset",
      });
      const result = await evaluator({
        outputs: trajectory(broad, toolCall({ origin: "LAX" })),
        referenceOutputs: trajectory(narrow, narrow),
      });
      expect(result.score).toBe(false);
    });
  }
);

describe("matcher overrides", () => {
  it("use the same assignment", async () => {
    const evaluator = createTrajectoryMatchEvaluator({
      trajectoryMatchMode: "unordered",
      toolArgsMatchOverrides: {
        search: (output, reference) =>
          Object.entries(reference).every(
            ([key, value]) => output[key] === value
          ),
      },
    });
    const result = await evaluator({
      outputs: trajectory(broad, narrow),
      referenceOutputs: trajectory(narrow, broad),
    });
    expect(result.score).toBe(true);
  });
});
