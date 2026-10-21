import { describe, expect, test } from "vitest";

import { createTrajectoryMatchEvaluator } from "../match.js";
import type { ChatCompletionMessage } from "../../types.js";

const trajectory = (args: Record<string, unknown>): ChatCompletionMessage[] => [
  { role: "user", content: "What time is my flight?" },
  {
    role: "assistant",
    content: "",
    tool_calls: [
      {
        type: "function",
        id: "1",
        function: {
          name: "get_flight_info",
          arguments: JSON.stringify(args),
        },
      },
    ],
  },
];

describe("trajectory tool args matcher argument order", () => {
  test.each(["strict", "unordered", "subset", "superset"] as const)(
    "custom matcher receives output args first in %s mode",
    async (trajectoryMatchMode) => {
      const calls: [Record<string, unknown>, Record<string, unknown>][] = [];
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
        toolArgsMatchOverrides: {
          get_flight_info: (outputArgs, referenceArgs) => {
            calls.push([outputArgs, referenceArgs]);
            return true;
          },
        },
      });
      const result = await evaluator({
        outputs: trajectory({ source: "output" }),
        referenceOutputs: trajectory({ source: "reference" }),
      });

      expect(result.score).toBe(true);
      expect(calls.length).toBeGreaterThan(0);
      for (const [outputArgs, referenceArgs] of calls) {
        expect(outputArgs).toEqual({ source: "output" });
        expect(referenceArgs).toEqual({ source: "reference" });
      }
    }
  );

  test.each(["unordered", "subset", "superset"] as const)(
    "subset tool args mode is consistent in %s mode",
    async (trajectoryMatchMode) => {
      // The output args are a subset of the reference args.
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
        toolArgsMatchMode: "subset",
      });
      const result = await evaluator({
        outputs: trajectory({ flight_no: "LX0112" }),
        referenceOutputs: trajectory({ flight_no: "LX0112", foo: "bar" }),
      });

      expect(result.score).toBe(true);
    }
  );
});
