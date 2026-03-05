/* eslint-disable @typescript-eslint/no-explicit-any */
import * as ls from "langsmith/vitest";
import { expect } from "vitest";

import { HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { createTrajectoryMatchEvaluator } from "../match.js";
import { FlexibleChatCompletionMessage } from "../../types.js";

ls.describe("trajectory", () => {
  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
    },
  ])("trajectory exact match", async ({ trajectoryMatchMode, feedbackKey }) => {
    const evaluator = createTrajectoryMatchEvaluator({
      trajectoryMatchMode,
    });
    const result = await evaluator({
      outputs: [
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
        { role: "tool", content: "It's 80 degrees and sunny in SF." },
        {
          role: "assistant",
          content: "The weather in SF is 80 degrees and sunny.",
        },
      ],
      referenceOutputs: [
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
        { role: "tool", content: "It's 80˚ and sunny in San Francisco." },
        {
          role: "assistant",
          content: "The weather in San Francisco is 80˚ and sunny.",
        },
      ],
    });
    expect(result).toBeDefined();
    expect(result.key).toBe(feedbackKey);
    expect(result.score).toBe(true);
    expect(result.comment).toBeUndefined();
  });

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
      score: true,
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
      score: true,
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
      score: true,
    },
  ])(
    "different message count",
    async ({ trajectoryMatchMode, feedbackKey, score }) => {
      const outputs = [
        { role: "user", content: "What is the weather in SF and London?" },
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
        { role: "tool", content: "It's 90 degrees and rainy in London." },
        {
          role: "assistant",
          content:
            "The weather in SF is 80 degrees and sunny. In London, it's 90 degrees and rainy.",
        },
      ] satisfies FlexibleChatCompletionMessage[];
      const referenceOutputs = [
        { role: "user", content: "What is the weather in SF and London?" },
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
        { role: "tool", content: "It's 90 degrees and rainy in London." },
        { role: "tool", content: "It's 80 degrees and sunny in SF." },
        {
          role: "assistant",
          content:
            "The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        },
      ] satisfies FlexibleChatCompletionMessage[];
      const evaluator = createTrajectoryMatchEvaluator({ trajectoryMatchMode });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(score);
    }
  );

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
      score: false,
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
      score: true,
    },
  ])(
    "trajectory subset tool call",
    async ({ trajectoryMatchMode, feedbackKey, score }) => {
      const outputs = [
        { role: "user", content: "What is the weather in SF and London?" },
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
          content:
            "The weather in SF is 80 degrees and sunny. In London, it's 9000 degrees and hallucinating.",
        },
      ] satisfies FlexibleChatCompletionMessage[];
      const referenceOutputs = [
        { role: "user", content: "What is the weather in SF and London?" },
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
        { role: "tool", content: "It's 90 degrees and rainy in London." },
        { role: "tool", content: "It's 90 degrees and rainy in London." },
        {
          role: "assistant",
          content:
            "The weather in London is 90˚ and rainy. In SF, it's 80˚ and sunny.",
        },
      ] satisfies FlexibleChatCompletionMessage[];
      const evaluator = createTrajectoryMatchEvaluator({ trajectoryMatchMode });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(score);
    }
  );

  ls.test.each([
    {
      inputs: {},
      trajectoryMatchMode: "strict",
      feedbackKey: "trajectory_strict_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "unordered",
      feedbackKey: "trajectory_unordered_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "superset",
      feedbackKey: "trajectory_superset_match",
    },
    {
      inputs: {},
      trajectoryMatchMode: "subset",
      feedbackKey: "trajectory_subset_match",
    },
  ])(
    "trajectory match with langchain messages",
    async ({ trajectoryMatchMode, feedbackKey }) => {
      const outputs = [
        new HumanMessage("What is the weather in SF?"),
        new AIMessage({
          content: "",
          tool_calls: [
            { id: "1234", name: "get_weather", args: { city: "San Francisco" } },
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
            { id: "4321", name: "get_weather", args: { city: "San Francisco" } },
          ],
        }),
        new ToolMessage({
          tool_call_id: "4321",
          content: "It's 80 degrees and sunny in San Francisco.",
        }),
        new AIMessage("The weather in SF is 80˚ and sunny."),
      ];
      const evaluator = createTrajectoryMatchEvaluator({ trajectoryMatchMode });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result.key).toBe(feedbackKey);
      expect(result.score).toBe(true);
    }
  );

  ls.test.each([
    { inputs: {}, toolArgsMatchMode: "exact", score: false },
    { inputs: {}, toolArgsMatchMode: "ignore", score: true },
  ])("trajectory match strict params", async ({ toolArgsMatchMode, score }) => {
    const outputs = [
      new HumanMessage("What is the weather in SF?"),
      new AIMessage({
        content: "",
        tool_calls: [{ id: "1234", name: "get_weather", args: { city: "SF" } }],
      }),
      new ToolMessage({ content: "It's 80 degrees and sunny.", tool_call_id: "1234" }),
      new AIMessage("The weather in SF is 80 degrees and sunny."),
    ];
    const referenceOutputs = [
      new HumanMessage("What is the weather in SF?"),
      new AIMessage({
        content: "",
        tool_calls: [
          { id: "1234", name: "get_weather", args: { city: "San Francisco" } },
        ],
      }),
      new ToolMessage({ content: "It's 80 degrees and sunny.", tool_call_id: "1234" }),
      new AIMessage("The weather in SF is 80 degrees and sunny."),
    ];
    const evaluator = createTrajectoryMatchEvaluator({
      trajectoryMatchMode: "strict",
      toolArgsMatchMode: toolArgsMatchMode as any,
    });
    const result = await evaluator({ outputs, referenceOutputs });
    expect(result).toEqual({ key: "trajectory_strict_match", score, comment: undefined });
  });

  ls.test.each([
    { inputs: {}, toolArgsMatchMode: "exact", score: false },
    { inputs: {}, toolArgsMatchMode: "ignore", score: true },
    { inputs: {}, toolArgsMatchMode: "subset", score: false },
    { inputs: {}, toolArgsMatchMode: "superset", score: true },
  ])("tool_args_match_mode superset", async ({ toolArgsMatchMode, score }) => {
    const outputs = [
      { role: "user", content: "Hi there, what time is my flight?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            id: "123",
            function: {
              name: "get_flight_info",
              arguments: JSON.stringify({ is_cool: true, flight_no: "LX0112" }),
            },
          },
        ],
      },
      { role: "assistant", content: "Your flight is at 10:00 AM." },
    ] satisfies FlexibleChatCompletionMessage[];
    const referenceOutputs = [
      { role: "user", content: "Hi there, what time is my flight?" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            id: "321",
            function: {
              name: "get_flight_info",
              arguments: JSON.stringify({ flight_no: "LX0112" }),
            },
          },
        ],
      },
      { role: "assistant", content: "Your flight is at 10:00 AM." },
    ] satisfies FlexibleChatCompletionMessage[];
    const evaluator = createTrajectoryMatchEvaluator({
      toolArgsMatchMode: toolArgsMatchMode as any,
    });
    const result = await evaluator({ outputs, referenceOutputs });
    expect(result.score).toBe(score);
  });

  ls.test.each([
    { inputs: {} },
  ])(
    "trajectory match with nested field overrides",
    async () => {
      const outputs = [
        { role: "user", content: "Hi there, what time is my flight?" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              type: "function",
              id: "4a286aff",
              function: {
                name: "lookup_policy",
                arguments: JSON.stringify({
                  query: "flight upgrades",
                  time: { start: "2025-03-22T18:34:40Z", end: "2025-03-22T20:34:40Z" },
                }),
              },
            },
          ],
        },
        { role: "assistant", content: "No upgrades available." },
      ] satisfies FlexibleChatCompletionMessage[];
      const referenceOutputs = [
        { role: "user", content: "Hi there, what time is my flight?" },
        {
          role: "assistant",
          content: "",
          tool_calls: [
            {
              type: "function",
              id: "cb2f81d3",
              function: {
                name: "lookup_policy",
                arguments: JSON.stringify({
                  query: "foo",
                  time: { start: "2025-03-22T18:34:40Z", end: "baz" },
                }),
              },
            },
          ],
        },
        { role: "assistant", content: "Upgrades possible." },
      ] satisfies FlexibleChatCompletionMessage[];

      const evaluatorNoOverrides = createTrajectoryMatchEvaluator({
        trajectoryMatchMode: "strict",
      });
      const resultNoOverrides = await evaluatorNoOverrides({ outputs, referenceOutputs });
      expect(resultNoOverrides.score).toBe(false);

      const evaluator = createTrajectoryMatchEvaluator({
        trajectoryMatchMode: "strict",
        toolArgsMatchOverrides: { lookup_policy: ["time.start"] },
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result.score).toBe(true);
    }
  );
});
