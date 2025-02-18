import { ChatCompletionMessage } from "../types.js";

function _normalizeToolCall(toolCall: Record<string, any>): {
  name: string;
  args: string;
} {
  if (
    "function" in toolCall &&
    toolCall.function != null &&
    typeof toolCall.function === "object"
  ) {
    return {
      name: toolCall.function.name,
      args: toolCall.function.arguments,
    };
  }
  return toolCall as { name: string; args: string };
}

function _extractToolCalls(
  messages: ChatCompletionMessage[]
): { name: string; args: string }[] {
  const toolCalls: { name: string; args: string }[] = [];
  for (const message of messages) {
    if (message.tool_calls) {
      toolCalls.push(...message.tool_calls.map(_normalizeToolCall));
    }
  }
  return toolCalls;
}

export function isTrajectorySuperset(
  outputs: ChatCompletionMessage[],
  referenceOutputs: ChatCompletionMessage[]
): boolean {
  const outputToolCalls = _extractToolCalls(outputs);
  const referenceToolCalls = _extractToolCalls(referenceOutputs);
  const outputToolCounts = new Map<string, number>();
  const referenceToolCounts = new Map<string, number>();

  for (const call of outputToolCalls) {
    outputToolCounts.set(call.name, (outputToolCounts.get(call.name) ?? 0) + 1);
  }
  for (const call of referenceToolCalls) {
    referenceToolCounts.set(
      call.name,
      (referenceToolCounts.get(call.name) ?? 0) + 1
    );
  }

  const allTools = new Set([
    ...outputToolCounts.keys(),
    ...referenceToolCounts.keys(),
  ]);
  for (const name of allTools) {
    if (
      (outputToolCounts.get(name) ?? 0) < (referenceToolCounts.get(name) ?? 0)
    ) {
      return false;
    }
  }
  return true;
}
