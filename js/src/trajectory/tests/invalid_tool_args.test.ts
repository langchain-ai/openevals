import { describe, expect, it } from "vitest";

import { createTrajectoryMatchEvaluator } from "../match.js";

const trajectory = (args: string) => [
  { role: "user", content: "What time is my flight?" },
  {
    role: "assistant",
    content: "",
    tool_calls: [
      {
        function: {
          name: "get_flight_info",
          arguments: args,
        },
      },
    ],
  },
];

const reference = trajectory(JSON.stringify({ flight: "AA1" }));

describe.each(["strict", "unordered", "subset", "superset"] as const)(
  "tool call args that are not a JSON object in %s mode",
  (trajectoryMatchMode) => {
    it.each(
      ['{"flight": "AA1"', "[1]", "null"].flatMap((args) =>
        (["exact", "subset", "superset"] as const).map(
          (toolArgsMatchMode) => [args, toolArgsMatchMode] as const
        )
      )
    )(
      "fails the match for %s with %s args",
      async (args, toolArgsMatchMode) => {
        const evaluator = createTrajectoryMatchEvaluator({
          trajectoryMatchMode,
          toolArgsMatchMode,
        });
        const result = await evaluator({
          outputs: trajectory(args),
          referenceOutputs: reference,
        });
        expect(result.score).toBe(false);
      }
    );

    it("matches when args are ignored", async () => {
      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode,
        toolArgsMatchMode: "ignore",
      });
      const result = await evaluator({
        outputs: trajectory('{"flight": "AA1"'),
        referenceOutputs: reference,
      });
      expect(result.score).toBe(true);
    });

    it.each(["", "  "])("treats %j as empty args", async (args) => {
      const evaluator = createTrajectoryMatchEvaluator({ trajectoryMatchMode });
      const result = await evaluator({
        outputs: trajectory(args),
        referenceOutputs: trajectory("{}"),
      });
      expect(result.score).toBe(true);
    });
  }
);
