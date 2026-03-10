import * as ls from "langsmith/vitest";
import { expect } from "vitest";

import { createTrajectoryLLMAsJudge } from "../llm.js";
import {
  TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
  TRAJECTORY_ACCURACY_PROMPT,
} from "../../prompts/trajectory/index.js";

ls.describe("trajectory llm", () => {
  ls.test.each([{ inputs: {} }])("trajectory match", async () => {
    const evaluator = createTrajectoryLLMAsJudge({
      prompt: TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
      model: "openai:o3-mini",
    });
    const outputs = [
      { role: "user", content: "What is the weather in SF?" },
      {
        role: "assistant",
        tool_calls: [
          {
            function: {
              name: "get_weather",
              arguments: JSON.stringify({ city: "SF" }),
            },
          },
        ],
        content: "",
      },
      { role: "tool", content: "It's 80 degrees and sunny in SF." },
      { role: "assistant", content: "The weather in SF is 80 degrees and sunny." },
    ];
    const referenceOutputs = [
      { role: "user", content: "What is the weather in SF?" },
      {
        role: "assistant",
        tool_calls: [
          {
            function: {
              name: "get_weather",
              arguments: JSON.stringify({ city: "San Francisco" }),
            },
          },
        ],
        content: "",
      },
      { role: "tool", content: "It's 80 degrees and sunny in San Francisco." },
      { role: "assistant", content: "The weather in SF is 80˚ and sunny." },
    ];
    const result = await evaluator({ outputs, referenceOutputs });
    expect(result.key).toBe("trajectory_accuracy");
    expect(result.score).toBe(true);
  });

  ls.test.each([{ inputs: {} }])("trajectory no reference", async () => {
    const evaluator = createTrajectoryLLMAsJudge({
      prompt: TRAJECTORY_ACCURACY_PROMPT,
      model: "openai:o3-mini",
    });
    const outputs = [
      { role: "user", content: "What is the weather in SF?" },
      {
        role: "assistant",
        tool_calls: [
          {
            function: {
              name: "get_weather",
              arguments: JSON.stringify({ city: "SF" }),
            },
          },
        ],
        content: "",
      },
      { role: "tool", content: "It's 80 degrees and sunny in SF." },
      { role: "assistant", content: "The weather in SF is 80 degrees and sunny." },
    ];
    const result = await evaluator({ outputs });
    expect(result.key).toBe("trajectory_accuracy");
    expect(result.score).toBe(true);
  });

  ls.test.each([{ inputs: {} }])("trajectory no reference bad trajectory", async () => {
    const evaluator = createTrajectoryLLMAsJudge({
      prompt: TRAJECTORY_ACCURACY_PROMPT,
      model: "openai:o3-mini",
    });
    const outputs = [
      { role: "user", content: "What are some good restaurants in SF?" },
      {
        role: "assistant",
        tool_calls: [
          {
            function: {
              name: "get_weather",
              arguments: JSON.stringify({ city: "SF" }),
            },
          },
        ],
        content: "",
      },
      { role: "tool", content: "It's 80 degrees and sunny in SF." },
      { role: "assistant", content: "The weather in SF is 80 degrees and sunny." },
    ];
    const result = await evaluator({ outputs });
    expect(result.key).toBe("trajectory_accuracy");
    expect(result.score).toBe(false);
  });
});
