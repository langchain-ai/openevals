import { describe, expect, it } from "vitest";

import { createTrajectoryMatchEvaluator } from "../match.js";

const trajectory = (args: Record<string, unknown>) => [
  { role: "user", content: "What time is my flight?" },
  {
    role: "assistant",
    content: "",
    tool_calls: [
      {
        function: {
          name: "get_flight_info",
          arguments: JSON.stringify(args),
        },
      },
    ],
  },
];

describe("trajectory tool args matching", () => {
  it.each(["strict", "unordered", "subset", "superset"] as const)(
    "passes the output args first to a custom matcher in %s mode",
    async (trajectoryMatchMode) => {
      const calls: [unknown, unknown][] = [];
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
      for (const call of calls) {
        expect(call).toEqual([{ source: "output" }, { source: "reference" }]);
      }
    }
  );

  it.each(["unordered", "subset", "superset"] as const)(
    "applies the subset tool args mode consistently in %s mode",
    async (trajectoryMatchMode) => {
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
