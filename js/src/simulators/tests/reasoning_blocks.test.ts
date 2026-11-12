import { describe, expect, test } from "vitest";
import { AIMessage } from "@langchain/core/messages";

import { runMultiturnSimulation } from "../multiturn.js";
import type { ChatCompletionMessage } from "../../index.js";

const REASONING = {
  type: "reasoning",
  id: "rs_1",
  summary: [{ type: "summary_text", text: "Checking refund policy." }],
};

async function run(appReply: AIMessage | ChatCompletionMessage) {
  const seen: ChatCompletionMessage[][] = [];
  const res = await runMultiturnSimulation({
    app: async () => appReply,
    user: ["Give me a refund!"],
    stoppingCondition: async (params: {
      trajectory: ChatCompletionMessage[];
    }) => {
      seen.push(params.trajectory);
      return true;
    },
    maxTurns: 1,
  });
  return { trajectory: res.trajectory as ChatCompletionMessage[], seen };
}

describe("multiturn simulation reasoning blocks", () => {
  test("drops reasoning blocks from the trajectory", async () => {
    const { trajectory, seen } = await run(
      new AIMessage({
        content: [REASONING, { type: "text", text: "Refund granted." }],
      })
    );
    const last = trajectory[trajectory.length - 1];
    expect(JSON.stringify(last.content)).not.toContain("reasoning");
    expect(JSON.stringify(last.content)).toContain("Refund granted.");
    const lastSeen = seen[0][seen[0].length - 1];
    expect(JSON.stringify(lastSeen.content)).not.toContain("reasoning");
  });

  test("drops function_call blocks but keeps tool calls", async () => {
    const { trajectory } = await run(
      new AIMessage({
        content: [
          REASONING,
          {
            type: "function_call",
            call_id: "call_1",
            name: "give_refund",
            arguments: "{}",
          },
        ],
        tool_calls: [{ name: "give_refund", args: {}, id: "call_1" }],
      })
    );
    const last = trajectory[trajectory.length - 1] as ChatCompletionMessage & {
      tool_calls?: { id: string }[];
    };
    expect(JSON.stringify(last.content ?? "")).not.toContain("function_call");
    expect(last.tool_calls?.[0]?.id).toBe("call_1");
  });

  test("leaves messages without reasoning unchanged", async () => {
    const reply = {
      role: "assistant",
      content: "Refund granted.",
      id: "m1",
    } as ChatCompletionMessage;
    const { trajectory } = await run(reply);
    expect(trajectory[trajectory.length - 1]).toEqual(reply);
  });
});
