import * as ls from "langsmith/vitest";

import { expect } from "vitest";
import { trajectoryStrictMatch } from "../strict.js";

const EXACT_MATCH_REFERENCE_OUTPUTS = {
  messages: [
    {
      role: "user",
      content: "What is the weather in SF?",
    },
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          function: {
            name: "get_weather",
            arguments: JSON.stringify({ city: "San Francisco" }),
          },
        },
      ],
    },
    {
      role: "tool",
      content: "It's 80˚ and sunny in San Francisco.",
    },
    {
      role: "assistant",
      content: "The weather in San Francisco is 80˚ and sunny.",
    },
  ],
};

ls.describe("trajectory", () => {
  ls.test.each([
    {
      inputs: {},
      referenceOutputs: EXACT_MATCH_REFERENCE_OUTPUTS,
      evaluator: trajectoryStrictMatch,
      feedbackKey: "trajectory_strict_match",
    },
  ])(
    "trajectory exact match",
    async ({ referenceOutputs, evaluator, feedbackKey }) => {
      const outputs = [
        {
          role: "user",
          content: "What is the weather in SF?",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "SF" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 80 degrees and sunny in SF.",
        },
        {
          role: "assistant",
          content: "The weather in SF is 80 degrees and sunny.",
        },
      ];
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs!,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(true);
      expect(result.comment).toBeUndefined();
    }
  );
});
