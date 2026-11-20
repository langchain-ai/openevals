import { describe, expect, it } from "vitest";

import { createTrajectoryMatchEvaluator } from "../match.js";
import type { ToolArgsMatchMode } from "../../types.js";

const trajectory = (city: string) => [
  { role: "user", content: "What is the weather in SF?" },
  {
    role: "assistant",
    content: "",
    tool_calls: [
      {
        function: {
          name: "get_weather",
          arguments: JSON.stringify({ city }),
        },
      },
    ],
  },
];

const trajectoryMatchModes = [
  "strict",
  "unordered",
  "subset",
  "superset",
] as const;

describe("trajectory tool args match overrides", () => {
  it.each(
    trajectoryMatchModes.flatMap((trajectoryMatchMode) =>
      ["strict", "Exact", "exactly"].map((override) => [
        trajectoryMatchMode,
        override,
      ])
    )
  )(
    "throws in %s mode for the unknown override mode %s",
    async (trajectoryMatchMode, override) => {
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
        toolArgsMatchOverrides: {
          get_weather: override as ToolArgsMatchMode,
        },
      });
      await expect(
        evaluator({
          outputs: trajectory("NYC"),
          referenceOutputs: trajectory("SF"),
        })
      ).rejects.toThrow("Invalid tool args match mode");
    }
  );

  it.each([
    ["exact", false],
    ["ignore", true],
  ] as const)(
    "still applies the known override mode %s",
    async (override, expected) => {
      const evaluator = createTrajectoryMatchEvaluator({
        toolArgsMatchOverrides: { get_weather: override },
      });
      const result = await evaluator({
        outputs: trajectory("NYC"),
        referenceOutputs: trajectory("SF"),
      });
      expect(result.score).toBe(expected);
    }
  );
});
