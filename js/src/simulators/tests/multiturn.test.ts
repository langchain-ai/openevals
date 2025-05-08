import { expect } from "vitest";
import * as ls from "langsmith/vitest";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { initChatModel } from "langchain/chat_models/universal";
import { tool } from "@langchain/core/tools";
import { OpenAI } from "openai";
import { z } from "zod";

import { runMultiturnSimulation } from "../multiturn.js";
import { createLLMSimulatedUser, _isInternalMessage } from "../prebuilts.js";
import { createLLMAsJudge } from "../../llm.js";
import type { ChatCompletionMessage } from "../../index.js";

ls.describe("Multiturn simulator", () => {
  ls.test(
    "multiturn_failure",
    {
      inputs: {
        messages: [],
      },
    },
    async () => {
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
      const agent = createReactAgent({
        llm: await initChatModel("openai:gpt-4.1-mini"),
        tools: [giveRefund],
        prompt:
          "You are an overworked customer service agent. If the user is rude, be polite only once, then be rude back and tell them to stop wasting your time.",
      });

      const app = async ({ inputs }: { inputs: ChatCompletionMessage }) => {
        const res = await agent.invoke({
          messages: [inputs],
        });
        return res.messages[res.messages.length - 1];
      };

      const user = createLLMSimulatedUser({
        system:
          "You are an angry user who wants a refund and keeps making additional demands.",
        model: "openai:gpt-4.1-nano",
      });

      const trajectoryEvaluator = createLLMAsJudge({
        model: "openai:gpt-4o-mini",
        prompt:
          "Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedbackKey: "satisfaction",
      });

      const result = await runMultiturnSimulation({
        app,
        user,
        trajectoryEvaluators: [trajectoryEvaluator],
        maxTurns: 5,
      });

      expect(result.evaluatorResults[0].score).toBe(false);
    }
  );

  ls.test(
    "multiturn_success",
    {
      inputs: {
        messages: [],
      },
    },
    async () => {
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
      const agent = createReactAgent({
        llm: await initChatModel("openai:gpt-4.1-nano"),
        tools: [giveRefund],
        checkpointer: new MemorySaver(),
      });

      const app = async ({
        inputs,
        threadId,
      }: {
        inputs: ChatCompletionMessage;
        threadId: string;
      }) => {
        const res = await agent.invoke(
          {
            messages: [inputs],
          },
          {
            configurable: {
              threadId,
            },
          }
        );
        return res.messages[res.messages.length - 1];
      };

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

      const result = await runMultiturnSimulation({
        app,
        user,
        trajectoryEvaluators: [trajectoryEvaluator],
        maxTurns: 5,
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
      const agent = createReactAgent({
        llm: await initChatModel("openai:gpt-4.1-nano"),
        tools: [giveRefund],
      });

      const app = async ({ inputs }: { inputs: ChatCompletionMessage }) => {
        const res = await agent.invoke({
          messages: [inputs],
        });
        return res.messages[res.messages.length - 1];
      };

      const trajectoryEvaluator = createLLMAsJudge({
        model: "openai:gpt-4o-mini",
        prompt:
          "Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedbackKey: "satisfaction",
      });

      const result = await runMultiturnSimulation({
        app,
        user: [
          { role: "user", content: "Give me a refund!", foo: "bar" },
          "All work and no play makes Jack a dull boy 1.",
          "All work and no play makes Jack a dull boy 2.",
          "All work and no play makes Jack a dull boy 3.",
          "All work and no play makes Jack a dull boy 4.",
        ],
        trajectoryEvaluators: [trajectoryEvaluator],
        maxTurns: 5,
      });

      const filteredTrajectory = result.trajectory.filter(
        (m) => !_isInternalMessage(m as any)
      );

      expect(filteredTrajectory[2].content).toBe(
        "All work and no play makes Jack a dull boy 1."
      );
      expect(filteredTrajectory[4].content).toBe(
        "All work and no play makes Jack a dull boy 2."
      );
      expect(filteredTrajectory[6].content).toBe(
        "All work and no play makes Jack a dull boy 3."
      );
      expect(filteredTrajectory[8].content).toBe(
        "All work and no play makes Jack a dull boy 4."
      );
    }
  );

  ls.test.only(
    "multiturn_message_with_openai",
    {
      inputs: {
        messages: [{ role: "user" as const, content: "I want a refund!" }],
      },
    },
    async ({ inputs }) => {
      const client = new OpenAI();

      // Create a custom app function
      const app = async ({ inputs }: { inputs: ChatCompletionMessage }) => {
        const res = await client.chat.completions.create({
          model: "gpt-4.1-nano",
          messages: [
            {
              role: "system",
              content:
                "You are a patient and understanding customer service agent",
            },
            inputs,
          ],
        });
        return res.choices[0].message;
      };

      const user = createLLMSimulatedUser({
        system:
          "You are an aggressive and hostile customer who wants a refund for their car.",
        model: "openai:gpt-4.1-nano",
        fixedResponses: inputs.messages,
      });

      const trajectoryEvaluator = createLLMAsJudge({
        model: "openai:gpt-4o-mini",
        prompt:
          "Based on the below conversation, was the user satisfied?\n{outputs}",
        feedbackKey: "satisfaction",
      });

      const result = await runMultiturnSimulation({
        app,
        user,
        trajectoryEvaluators: [trajectoryEvaluator],
        maxTurns: 5,
      });

      expect(result.evaluatorResults[0].score).toBe(false);
    }
  );

  ls.test(
    "multiturn_stopping_condition",
    {
      inputs: {
        messages: [{ role: "user" as const, content: "Give me a refund!" }],
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
      const agent = createReactAgent({
        llm: await initChatModel("openai:gpt-4.1-nano"),
        tools: [giveRefund],
      });

      const app = async ({ inputs }: { inputs: ChatCompletionMessage }) => {
        const res = await agent.invoke({
          messages: [inputs],
        });
        return res.messages[res.messages.length - 1];
      };

      const user = createLLMSimulatedUser({
        system: "You are a happy and reasonable person who wants a refund.",
        model: "openai:gpt-4.1-nano",
        fixedResponses: inputs.messages,
      });

      const trajectoryEvaluator = createLLMAsJudge({
        model: "openai:gpt-4o-mini",
        prompt:
          "Based on the below conversation, has the user been satisfied?\n{outputs}",
        feedbackKey: "satisfaction",
      });

      const client = new OpenAI();

      // Create a stopping condition
      const stoppingCondition = async (params: {
        trajectory: ChatCompletionMessage[];
        turnCounter: number;
        threadId: string;
      }): Promise<boolean> => {
        const res = await client.chat.completions.create({
          model: "gpt-4.1-nano",
          messages: [
            {
              role: "system",
              content:
                "Your job is to determine if a refund has been granted in the following conversation. Respond only with JSON with a single boolean key named 'refund_granted'.",
            },
            ...(params.trajectory as any),
          ],
          response_format: { type: "json_object" },
        });

        const content = res.choices[0].message.content;
        return JSON.parse(content as string).refund_granted;
      };

      const result = await runMultiturnSimulation({
        app,
        user,
        trajectoryEvaluators: [trajectoryEvaluator],
        stoppingCondition,
        maxTurns: 10,
      });

      expect(result.evaluatorResults[0].score).toBe(true);
      expect(result.trajectory.length).toBeLessThan(20);
    }
  );
});
