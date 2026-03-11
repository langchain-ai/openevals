import {
  ChatCompletionMessage,
  ToolArgsMatchMode,
  ToolArgsMatchOverrides,
  ToolArgsMatcher,
} from "../types.js";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _normalizeToolCall(toolCall: Record<string, any>): {
  name: string;
  args: Record<string, unknown>;
} {
  if (
    "function" in toolCall &&
    toolCall.function != null &&
    typeof toolCall.function === "object" &&
    typeof toolCall.function.arguments === "string"
  ) {
    return {
      name: toolCall.function.name,
      args: JSON.parse(toolCall.function.arguments),
    };
  }
  return toolCall as { name: string; args: Record<string, unknown> };
}

function _extractToolCalls(
  messages: ChatCompletionMessage[]
): { name: string; args: Record<string, unknown> }[] {
  const toolCalls: { name: string; args: Record<string, unknown> }[] = [];
  for (const message of messages) {
    if (message.tool_calls) {
      toolCalls.push(...message.tool_calls.map(_normalizeToolCall));
    }
  }
  return toolCalls;
}

export async function _isTrajectorySuperset(
  outputs: ChatCompletionMessage[],
  referenceOutputs: ChatCompletionMessage[],
  toolArgsMatchMode: ToolArgsMatchMode,
  toolArgsMatchOverrides?: ToolArgsMatchOverrides
): Promise<boolean> {
  const outputToolCalls = _extractToolCalls(outputs);
  const referenceToolCalls = _extractToolCalls(referenceOutputs);

  // Keep track of which reference tool calls have been matched
  const matchedReferenceCalls = new Set<number>();

  // For each reference tool call, find a matching output tool call
  for (const refCall of referenceToolCalls) {
    const refName = refCall.name;
    const refArgs = refCall.args;

    let foundMatch = false;
    for (let outIdx = 0; outIdx < outputToolCalls.length; outIdx++) {
      const outCall = outputToolCalls[outIdx];
      const outName = outCall.name;

      // Names must match
      if (refName !== outName) {
        continue;
      }

      // If we're already using this output call for a different match, skip
      if (matchedReferenceCalls.has(outIdx)) {
        continue;
      }

      // Check tool args according to match mode
      const matcher = _getMatcherForToolName(
        refName,
        toolArgsMatchMode,
        toolArgsMatchOverrides
      );

      const outArgs = outCall.args;
      if (await matcher(outArgs, refArgs)) {
        matchedReferenceCalls.add(outIdx);
        foundMatch = true;
        break;
      }
    }

    // If we didn't find a match for this reference call, we're not a superset
    if (!foundMatch) {
      return false;
    }
  }

  return true;
}

// Deep equality check function
function _deepEqual(a: unknown, b: unknown): boolean {
  if (a == null && b == null) return true;
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object" || !a || !b) return false;

  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((val, index) => _deepEqual(val, b[index]));
  }

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) return false;

  return (
    keysA.every((key) => keysB.includes(key)) &&
    keysB.every((key) => keysA.includes(key)) &&
    keysA.every((key) =>
      _deepEqual(
        (a as Record<string, unknown>)[key],
        (b as Record<string, unknown>)[key]
      )
    )
  );
}

function _exactMatch(
  toolCall: Record<string, unknown>,
  referenceToolCall: Record<string, unknown>
): boolean {
  return _deepEqual(toolCall, referenceToolCall);
}

function _ignoreMatch(
  _toolCall: Record<string, unknown>,
  _referenceToolCall: Record<string, unknown>
): boolean {
  return true;
}

function _subsetMatch(
  toolCall: Record<string, unknown>,
  referenceToolCall: Record<string, unknown>
): boolean {
  // Every key-value pair in toolCall must exist in referenceToolCall with the same value
  return Object.entries(toolCall).every(
    ([key, value]) =>
      key in referenceToolCall && _deepEqual(referenceToolCall[key], value)
  );
}

function _supersetMatch(
  toolCall: Record<string, unknown>,
  referenceToolCall: Record<string, unknown>
): boolean {
  // Every key-value pair in referenceToolCall must exist in toolCall with the same value
  return Object.entries(referenceToolCall).every(
    ([key, value]) => key in toolCall && _deepEqual(toolCall[key], value)
  );
}

function _getMatcherForComparisonMode(
  mode: ToolArgsMatchMode
): ToolArgsMatcher {
  if (mode === "exact") {
    return _exactMatch;
  } else if (mode === "subset") {
    return _subsetMatch;
  } else if (mode === "superset") {
    return _supersetMatch;
  } else {
    return _ignoreMatch;
  }
}

function _getPartialMatcherOnKeys(keys: string[]): ToolArgsMatcher {
  const getNestedValue = (
    d: Record<string, unknown>,
    keyPath: string
  ): unknown => {
    let current: unknown = d;
    for (const part of keyPath.split(".")) {
      if (current && typeof current === "object" && part in current) {
        current = current[part as keyof typeof current];
      } else {
        return undefined;
      }
    }
    return current;
  };

  return (
    outputCall: Record<string, unknown>,
    referenceCall: Record<string, unknown>
  ): boolean => {
    return keys.every((key) => {
      const nestedOutputValue = getNestedValue(outputCall, key);
      const nestedReferenceValue = getNestedValue(referenceCall, key);
      return _deepEqual(nestedOutputValue, nestedReferenceValue);
    });
  };
}

export function _getMatcherForToolName(
  toolCallName: string,
  toolArgsMatchMode: ToolArgsMatchMode,
  toolArgsMatchOverrides?: ToolArgsMatchOverrides
): ToolArgsMatcher {
  let matcher = _getMatcherForComparisonMode(toolArgsMatchMode);

  if (toolArgsMatchOverrides && toolCallName in toolArgsMatchOverrides) {
    const override = toolArgsMatchOverrides[toolCallName];

    if (typeof override === "string") {
      matcher = _getMatcherForComparisonMode(override);
    } else if (typeof override === "function") {
      matcher = override;
    } else if (Array.isArray(override)) {
      matcher = _getPartialMatcherOnKeys(override);
    }
  }

  return matcher;
}

export function _chatCompletionMessagesToString(
  messages: ChatCompletionMessage[]
): string {
  function formatMessage(message: ChatCompletionMessage): string {
    let content = message.content ?? "";

    // Handle tool/function calls
    if (message.tool_calls) {
      const toolCallsStr = message.tool_calls
        .map((call: { function: { name: string; arguments: string } }) => {
          const func = call.function ?? {};
          return `<tool_call>\n<name>${func.name ?? ""}</name>\n<arguments>${func.arguments ?? ""}</arguments>\n</tool_call>`;
        })
        .join("\n");

      content = content ? `${content}\n${toolCallsStr}` : toolCallsStr;
    }

    // Handle tool call results
    if (message.tool_call_id) {
      content = `<tool_result>\n<id>${message.tool_call_id}</id>\n<content>${content}</content>\n</tool_result>`;
    }

    return `<${message.role ?? ""}>\n${content}\n</${message.role ?? ""}>`;
  }

  return messages.map(formatMessage).join("\n\n");
}
