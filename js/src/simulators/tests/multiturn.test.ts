import { expect } from "vitest";
import * as ls from "langsmith/vitest";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { initChatModel } from "langchain/chat_models/universal";
import { tool } from "@langchain/core/tools";
import { OpenAI } from "openai";
import { z } from "zod";

import { createMultiturnSimulator } from "../multiturn.js";
import { createLLMSimulatedUser } from "../prebuilts.js";
import { createLLMAsJudge } from "../../llm.js";
import type { MultiturnSimulatorTrajectory } from "../../types.js";

ls.describe("Multiturn simulator", () => {
  ls.test(
    "multiturn_failure",
    {
      inputs: {
        messages: [{ role: "user", content: "Please give me a refund." }],
      },
    },
    async ({ inputs }) => {
      // Create a function that returns a refund denial
      const giveRefund = tool(
        async () => {
          return "Refunds are not permitted.";
        },
        {
          name: "give_refund",
          description: "Give a refund to the user.",
          schema: z.object({}),
        }
      );

      // Create a React-style agent
      const app = createReactAgent({
        llm: await initChatModel("openai:gpt-4.1-mini"),
        tools: [giveRefund],
        prompt:
          "You are an overworked customer service agent. If the user is rude, be polite only once, then be rude back and tell them to stop wasting your time.",
        checkpointer: new MemorySaver(),
      });

      const user = createLLMSimulatedUser({
        system:
          "You are an angry user who is frustrated with the service and keeps making additional demands.",
        model: "openai:gpt-4.1-nano",
      });

      const trajectoryEvaluator = createLLMAsJudge({
        model: "openai:gpt-4o-mini",
        prompt:
          "Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedbackKey: "satisfaction",
      });

      const simulator = createMultiturnSimulator({
        app,
        user,
        trajectoryEvaluators: [trajectoryEvaluator],
        maxTurns: 5,
      });

      const result = await simulator({
        initialTrajectory: inputs,
        runnableConfig: { configurable: { thread_id: "1" } },
      });

      expect(result.evaluatorResults[0].score).toBe(false);
    }
  );

  ls.test(
    "multiturn_success",
    {
      inputs: {
        messages: [{ role: "user", content: "Give me a refund!" }],
      },
    },
    async ({ inputs }) => {
      // Create a function that returns a refund approval
      const giveRefund = tool(
        async () => {
          return "Refunds granted.";
        },
        {
          name: "give_refund",
          description: "Give a refund to the user.",
          schema: z.object({}),
        }
      );

      // Create a React-style agent
      const app = createReactAgent({
        llm: await initChatModel("openai:gpt-4.1-nano"),
        tools: [giveRefund],
        checkpointer: new MemorySaver(),
      });

      const user = createLLMSimulatedUser({
        system: "You are a happy and reasonable person who wants a refund.",
        model: "openai:gpt-4.1-nano",
      });

      const trajectoryEvaluator = createLLMAsJudge({
        model: "openai:gpt-4o-mini",
        prompt:
          "Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedbackKey: "satisfaction",
      });

      const simulator = createMultiturnSimulator({
        app,
        user,
        trajectoryEvaluators: [trajectoryEvaluator],
        maxTurns: 5,
      });

      const result = await simulator({
        initialTrajectory: inputs,
        runnableConfig: { configurable: { thread_id: "1" } },
      });

      expect(result.evaluatorResults[0].score).toBe(true);
    }
  );

  ls.test(
    "multiturn_preset_responses",
    {
      inputs: {
        messages: [{ role: "user", content: "Give me a refund!" }],
      },
    },
    async ({ inputs }) => {
      // Create a function that returns a refund approval
      const giveRefund = tool(
        async () => {
          return "Refunds granted.";
        },
        {
          name: "give_refund",
          description: "Give a refund to the user.",
          schema: z.object({}),
        }
      );

      // Create a React-style agent
      const app = createReactAgent({
        llm: await initChatModel("openai:gpt-4.1-nano"),
        tools: [giveRefund],
        checkpointer: new MemorySaver(),
      });

      const trajectoryEvaluator = createLLMAsJudge({
        model: "openai:gpt-4o-mini",
        prompt:
          "Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedbackKey: "satisfaction",
      });

      const simulator = createMultiturnSimulator({
        app,
        user: [
          "All work and no play makes Jack a dull boy 1.",
          "All work and no play makes Jack a dull boy 2.",
          "All work and no play makes Jack a dull boy 3.",
          "All work and no play makes Jack a dull boy 4.",
        ],
        trajectoryEvaluators: [trajectoryEvaluator],
        maxTurns: 5,
      });

      const result = await simulator({
        initialTrajectory: inputs,
        runnableConfig: { configurable: { thread_id: "1" } },
      });

      expect(result.trajectory.messages[2].content).toBe(
        "All work and no play makes Jack a dull boy 1."
      );
      expect(result.trajectory.messages[4].content).toBe(
        "All work and no play makes Jack a dull boy 2."
      );
      expect(result.trajectory.messages[6].content).toBe(
        "All work and no play makes Jack a dull boy 3."
      );
      expect(result.trajectory.messages[8].content).toBe(
        "All work and no play makes Jack a dull boy 4."
      );
    }
  );

  ls.test(
    "multiturn_message_with_openai",
    {
      inputs: {
        messages: [{ role: "user", content: "Give me a cracker!" }],
      },
    },
    async ({ inputs }) => {
      const client = new OpenAI();

      // Create a custom app function
      const app = async ({ messages }: MultiturnSimulatorTrajectory) => {
        const res = await client.chat.completions.create({
          model: "gpt-4.1-nano",
          messages: [
            {
              role: "system",
              content:
                "You are an angry parrot named Polly who is angry at everything. Squawk a lot.",
            },
            ...(messages as any),
          ],
        });
        return { messages: res.choices[0].message };
      };

      const user = createLLMSimulatedUser({
        system:
          "You are an angry parrot named Anna who is angry at everything. Squawk a lot.",
        model: "openai:gpt-4.1-nano",
      });

      const trajectoryEvaluator = createLLMAsJudge({
        model: "openai:gpt-4o-mini",
        prompt:
          "Based on the below conversation, are the parrots angry?\n{outputs}",
        feedbackKey: "anger",
      });

      const simulator = createMultiturnSimulator({
        app,
        user,
        trajectoryEvaluators: [trajectoryEvaluator],
        maxTurns: 5,
      });

      const result = await simulator({
        initialTrajectory: inputs,
      });

      expect(result.evaluatorResults[0].score).toBe(true);
    }
  );

  ls.test(
    "multiturn_stopping_condition",
    {
      inputs: {
        messages: [{ role: "user", content: "Give me a refund!" }],
      },
    },
    async ({ inputs }) => {
      // Create a function that returns a refund approval
      const giveRefund = tool(
        async () => {
          return "Refunds granted.";
        },
        {
          name: "give_refund",
          description: "Give a refund to the user.",
          schema: z.object({}),
        }
      );

      // Create a React-style agent
      const app = createReactAgent({
        llm: await initChatModel("openai:gpt-4.1-nano"),
        tools: [giveRefund],
        checkpointer: new MemorySaver(),
      });

      const user = createLLMSimulatedUser({
        system: "You are a happy and reasonable person who wants a refund.",
        model: "openai:gpt-4.1-nano",
      });

      const trajectoryEvaluator = createLLMAsJudge({
        model: "openai:gpt-4o-mini",
        prompt:
          "Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedbackKey: "satisfaction",
      });

      const client = new OpenAI();

      // Create a stopping condition
      const stoppingCondition = async (
        currentTrajectory: MultiturnSimulatorTrajectory
      ): Promise<boolean> => {
        const res = await client.chat.completions.create({
          model: "gpt-4.1-nano",
          messages: [
            {
              role: "system",
              content:
                "Your job is to determine if a refund has been granted in the following conversation. Respond only with JSON with a single boolean key named 'refund_granted'.",
            },
            ...(currentTrajectory.messages as any),
          ],
          response_format: { type: "json_object" },
        });

        const content = res.choices[0].message.content;
        return JSON.parse(content as string).refund_granted;
      };

      const simulator = createMultiturnSimulator({
        app,
        user,
        trajectoryEvaluators: [trajectoryEvaluator],
        stoppingCondition,
        maxTurns: 10,
      });

      const result = await simulator({
        initialTrajectory: inputs,
        runnableConfig: { configurable: { thread_id: "1" } },
      });

      expect(result.evaluatorResults[0].score).toBe(true);
      expect(result.trajectory.messages.length).toBeLessThan(20);
    }
  );
});
