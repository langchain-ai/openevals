import * as ls from "langsmith/vitest";
import { expect } from "vitest";

import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { trajectoryStrictMatch } from "../strict.js";
import { trajectoryUnorderedMatch } from "../unordered.js";
import { trajectorySuperset } from "../superset.js";
import { trajectorySubset } from "../subset.js";

ls.describe("trajectory", () => {
  ls.test.each([
    {
      inputs: {},
      evaluator: trajectoryStrictMatch,
      feedbackKey: "trajectory_strict_match",
    },
    {
      inputs: {},
      evaluator: trajectoryUnorderedMatch,
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      evaluator: trajectorySuperset,
      feedbackKey: "trajectory_superset",
    },
    {
      inputs: {},
      evaluator: trajectorySubset,
      feedbackKey: "trajectory_subset",
    },
  ])("trajectory exact match", async ({ evaluator, feedbackKey }) => {
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
    const referenceOutputs = [
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
    ];
    const result = await evaluator({
      outputs,
      referenceOutputs,
    });
    expect(result).toBeDefined();
    expect(result.key).toBe(feedbackKey);
    expect(result.score).toBe(true);
    expect(result.comment).toBeUndefined();
  });

  ls.test.each([
    {
      inputs: {},
      evaluator: trajectoryStrictMatch,
      feedbackKey: "trajectory_strict_match",
    },
    {
      inputs: {},
      evaluator: trajectoryUnorderedMatch,
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      evaluator: trajectorySuperset,
      feedbackKey: "trajectory_superset",
    },
    {
      inputs: {},
      evaluator: trajectorySubset,
      feedbackKey: "trajectory_subset",
    },
  ])("different tool message order", async ({ evaluator, feedbackKey }) => {
    const outputs = [
      {
        role: "user",
        content: "What is the weather in SF and London?",
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
          {
            function: {
              name: "get_weather",
              arguments: JSON.stringify({ city: "London" }),
            },
          },
        ],
      },
      {
        role: "tool",
        content: "It's 80 degrees and sunny in SF.",
      },
      {
        role: "tool",
        content: "It's 90 degrees and rainy in London.",
      },
      {
        role: "assistant",
        content:
          "The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
      },
    ];
    const referenceOutputs = [
      {
        role: "user",
        content: "What is the weather in SF and London?",
      },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            function: {
              name: "get_weather",
              arguments: JSON.stringify({ city: "London" }),
            },
          },
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
        content: "It's 90 degrees and rainy in London.",
      },
      {
        role: "tool",
        content: "It's 80 degrees and sunny in SF.",
      },
      {
        role: "assistant",
        content:
          "The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
      },
    ];
    const result = await evaluator({
      outputs,
      referenceOutputs,
    });
    expect(result).toBeDefined();
    expect(result.key).toBe(feedbackKey);
    expect(result.score).toBe(true);
    expect(result.comment).toBeUndefined();
  });

  ls.test.each([
    {
      inputs: {},
      evaluator: trajectoryStrictMatch,
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectoryUnorderedMatch,
      feedbackKey: "trajectory_unordered_match",
      score: true,
    },
    {
      inputs: {},
      evaluator: trajectorySuperset,
      feedbackKey: "trajectory_superset",
      score: true,
    },
    {
      inputs: {},
      evaluator: trajectorySubset,
      feedbackKey: "trajectory_subset",
      score: true,
    },
  ])("different message count", async ({ evaluator, feedbackKey, score }) => {
    const outputs = [
      {
        role: "user",
        content: "What is the weather in SF and London?",
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
        content: "",
        tool_calls: [
          {
            function: {
              name: "get_weather",
              arguments: JSON.stringify({ city: "London" }),
            },
          },
        ],
      },
      {
        role: "tool",
        content: "It's 90 degrees and rainy in London.",
      },
      {
        role: "assistant",
        content:
          "The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
      },
    ];
    const referenceOutputs = [
      {
        role: "user",
        content: "What is the weather in SF and London?",
      },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            function: {
              name: "get_weather",
              arguments: JSON.stringify({ city: "London" }),
            },
          },
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
        content: "It's 90 degrees and rainy in London.",
      },
      {
        role: "tool",
        content: "It's 80 degrees and sunny in SF.",
      },
      {
        role: "assistant",
        content:
          "The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
      },
    ];
    const result = await evaluator({
      outputs,
      referenceOutputs,
    });
    expect(result).toBeDefined();
    expect(result.key).toBe(feedbackKey);
    expect(result.score).toBe(score);
    expect(result.comment).toBeUndefined();
  });

  ls.test.each([
    {
      inputs: {},
      evaluator: trajectoryStrictMatch,
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectoryUnorderedMatch,
      feedbackKey: "trajectory_unordered_match",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectorySuperset,
      feedbackKey: "trajectory_superset",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectorySubset,
      feedbackKey: "trajectory_subset",
      score: true,
    },
  ])(
    "trajectory subset tool call",
    async ({ evaluator, feedbackKey, score }) => {
      const outputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
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
          content:
            "The weather in SF is 80 degrees and sunny. In London, it's 9000 degrees and hallucinating.",
        },
      ];
      const referenceOutputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "London" }),
              },
            },
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
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "tool",
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "assistant",
          content:
            "The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        },
      ];
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(score);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      evaluator: trajectoryStrictMatch,
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectoryUnorderedMatch,
      feedbackKey: "trajectory_unordered_match",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectorySuperset,
      feedbackKey: "trajectory_superset",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectorySubset,
      feedbackKey: "trajectory_subset",
      score: false,
    },
  ])("different called tools", async ({ evaluator, feedbackKey, score }) => {
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
    const referenceOutputs = [
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
              name: "accuweather_forecast",
              arguments: JSON.stringify({ city: "San Francisco" }),
            },
          },
        ],
      },
      {
        role: "tool",
        content: "It's 80 degrees and sunny in San Francisco.",
      },
      {
        role: "assistant",
        content: "The weather in SF is 80˚ and sunny.",
      },
    ];
    const result = await evaluator({
      outputs,
      referenceOutputs,
    });
    expect(result).toBeDefined();
    expect(result.key).toBe(feedbackKey);
    expect(result.score).toBe(score);
    expect(result.comment).toBeUndefined();
  });

  ls.test.only.each([
    {
      inputs: {},
      evaluator: trajectoryStrictMatch,
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectoryUnorderedMatch,
      feedbackKey: "trajectory_unordered_match",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectorySuperset,
      feedbackKey: "trajectory_superset",
      score: true,
    },
    {
      inputs: {},
      evaluator: trajectorySubset,
      feedbackKey: "trajectory_subset",
      score: false,
    },
  ])(
    "trajectory with extra tool calls",
    async ({ evaluator, feedbackKey, score }) => {
      const outputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
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
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "London" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content: "It's 80 degrees and sunny in San Francisco.",
        },
        {
          role: "tool",
          content: "It's 90 degrees and rainy in London.",
        },
        {
          role: "assistant",
          content:
            "The weather in SF is 80˚ and sunny. In London, it's 90˚ and rainy.",
        },
      ];
      const referenceOutputs = [
        {
          role: "user",
          content: "What is the weather in SF and London?",
        },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              function: {
                name: "get_weather",
                arguments: JSON.stringify({ city: "SF and London" }),
              },
            },
          ],
        },
        {
          role: "tool",
          content:
            "It's 80 degrees and sunny in SF, and 90 degrees and rainy in London.",
        },
        {
          role: "assistant",
          content:
            "The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        },
      ];
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(score);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      evaluator: trajectoryStrictMatch,
      feedbackKey: "trajectory_strict_match",
    },
    {
      inputs: {},
      evaluator: trajectoryUnorderedMatch,
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      evaluator: trajectorySuperset,
      feedbackKey: "trajectory_superset",
    },
    {
      inputs: {},
      evaluator: trajectorySubset,
      feedbackKey: "trajectory_subset",
    },
  ])(
    "trajectory match with langchain messages",
    async ({ evaluator, feedbackKey }) => {
      const outputs = [
        new HumanMessage("What is the weather in SF?"),
        new AIMessage({
          content: "",
          tool_calls: [
            {
              id: "1234",
              name: "get_weather",
              args: { city: "SF" },
            },
          ],
        }),
        new ToolMessage({
          tool_call_id: "1234",
          content: "It's 80 degrees and sunny in SF.",
        }),
        new AIMessage("The weather in SF is 80 degrees and sunny."),
      ];
      const referenceOutputs = [
        new HumanMessage("What is the weather in SF?"),
        new AIMessage({
          content: "Let me check that for you!",
          tool_calls: [
            {
              id: "4321",
              name: "get_weather",
              args: { city: "San Francisco" },
            },
          ],
        }),
        new ToolMessage({
          tool_call_id: "4321",
          content: "It's 80 degrees and sunny in San Francisco.",
        }),
        new AIMessage("The weather in SF is 80˚ and sunny."),
      ];
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(true);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test.each([
    {
      inputs: {},
      evaluator: trajectoryStrictMatch,
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      evaluator: trajectoryUnorderedMatch,
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      evaluator: trajectorySuperset,
      feedbackKey: "trajectory_superset",
    },
    {
      inputs: {},
      evaluator: trajectorySubset,
      feedbackKey: "trajectory_subset",
    },
  ])(
    "trajectory match with langchain messages failure",
    async ({ evaluator, feedbackKey }) => {
      const outputs = [
        new HumanMessage("What is the weather in SF?"),
        new AIMessage({
          content: "",
          tool_calls: [
            {
              id: "1234",
              name: "get_weather",
              args: { city: "SF" },
            },
          ],
        }),
        new ToolMessage({
          tool_call_id: "1234",
          content: "It's 80 degrees and sunny in SF.",
        }),
        new AIMessage("The weather in SF is 80 degrees and sunny."),
      ];
      const referenceOutputs = [
        new HumanMessage("What is the weather in SF?"),
        new AIMessage({
          content: "Let me check that for you!",
          tool_calls: [
            {
              id: "4321",
              name: "accuweather_forecast",
              args: { city: "San Francisco" },
            },
          ],
        }),
        new ToolMessage({
          tool_call_id: "4321",
          content: "It's 80 degrees and sunny in San Francisco.",
        }),
        new AIMessage("The weather in SF is 80˚ and sunny."),
      ];
      const result = await evaluator({
        outputs,
        referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(false);
      expect(result.comment).toBeUndefined();
    }
  );
});
