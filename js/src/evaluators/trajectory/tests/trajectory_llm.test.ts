import * as ls from "langsmith/vitest";
import { expect } from "vitest";

import { createTrajectoryLLMAsJudge } from "../llm.js";
import { ChatCompletionMessage } from "../../types.js";

ls.describe("Trajectory LLM", () => {
  ls.test(
    "should match trajectories",
    {
      inputs: {},
    },
    async () => {
      const evaluator = createTrajectoryLLMAsJudge({});
      const inputs = {};
      const outputs: ChatCompletionMessage[] = [
        { role: "user", content: "What is the weather in SF?" },
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
        { role: "tool", content: "It's 80 degrees and sunny in SF." },
        {
          role: "assistant",
          content: "The weather in SF is 80 degrees and sunny.",
        },
      ];

      const referenceOutputs: ChatCompletionMessage[] = [
        { role: "user", content: "What is the weather in SF?" },
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
          content: "It's 80 degrees and sunny in San Francisco.",
        },
        { role: "assistant", content: "The weather in SF is 80˚ and sunny." },
      ];

      const evalResult = await evaluator({
        inputs,
        outputs,
        referenceOutputs,
      });

      expect(evalResult.key).toBe("trajectory_accuracy");
      expect(evalResult.score).toBe(true);
    }
  );

  ls.test(
    "should match trajectories with inverse rubric",
    { inputs: {} },
    async () => {
      const evaluator = createTrajectoryLLMAsJudge({});
      const inputs = {};
      const outputs: ChatCompletionMessage[] = [
        { role: "user", content: "What is the weather in SF?" },
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
        { role: "tool", content: "It's 80 degrees and sunny in SF." },
        {
          role: "assistant",
          content: "The weather in SF is 80 degrees and sunny.",
        },
      ];

      const referenceOutputs: ChatCompletionMessage[] = [
        { role: "user", content: "What is the weather in SF?" },
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
          content: "It's 80 degrees and sunny in San Francisco.",
        },
        { role: "assistant", content: "The weather in SF is 80˚ and sunny." },
      ];

      const evalResult = await evaluator({
        inputs,
        outputs,
        referenceOutputs,
        rubric:
          "We are looking for bad trajectories, so score should be 0 if the trajectory contains reasonable steps for the agent to answer the input, and 1 if not.",
      });

      expect(evalResult.key).toBe("trajectory_accuracy");
      expect(evalResult.score).toBe(false);
    }
  );
});
