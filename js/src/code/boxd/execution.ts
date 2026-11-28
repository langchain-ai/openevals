import type { Boxd } from "@boxd-sh/sdk";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";

import { _createBaseCodeEvaluator } from "../base.js";
import { SingleResultScorerReturnType } from "../../types.js";
import {
  EXTRACT_IMPORT_NAMES,
  TYPESCRIPT_EVALUATOR_SEPARATOR,
} from "../e2b/sandbox/files.js";
import {
  PACKAGE_JSON_FILE_WITH_TSX,
  TSX_VERSION,
  TYPESCRIPT_VERSION,
} from "./sandbox/files.js";
import {
  DEFAULT_TIMEOUT_MS,
  _runInMachine,
  _validateTimeoutMs,
} from "./machine.js";

const BOXD_COMMAND = [
  `[ -f package.json ] || cp openevals_package.json package.json`,
  `npm install --prefer-offline --no-audit --no-fund --loglevel=error`,
  `ls node_modules/typescript >/dev/null 2>&1 || npm install typescript@${TYPESCRIPT_VERSION} --prefer-offline --no-audit --no-fund --loglevel=error`,
  `ls node_modules/tsx >/dev/null 2>&1 || npm install tsx@${TSX_VERSION} --prefer-offline --no-audit --no-fund --loglevel=error`,
  `node extract_import_names.mjs | xargs -r -n1 sh -c 'ls "node_modules/$1" >/dev/null 2>&1 || npm install "$1" --prefer-offline --no-audit --no-fund --loglevel=error' _`,
  `echo '${TYPESCRIPT_EVALUATOR_SEPARATOR}'`,
  `npx tsx outputs.ts`,
].join(" && ");

/**
 * Creates an evaluator that runs TypeScript code in a boxd machine and returns
 * whether it ran without errors. Imported packages are installed first.
 *
 * @param config - Configuration object
 * @param config.boxdClient - boxd client used to reach the machine.
 * @param config.machineId - Id of the boxd machine to run the code in.
 * @param config.environmentVariables - Environment variables to set for the command in the machine.
 * @param config.sandboxProjectDirectory - Directory within the machine to use for the evaluation,
 *   relative to the home directory or absolute. Defaults to "openevals".
 *   Allows you to add additional project files ahead of time. An existing package.json
 *   there is kept; TypeScript 5 and the packages the code imports are added to it.
 * @param config.timeoutMs - Milliseconds to wait for the install and run. When it elapses,
 *   the command is stopped and the boxd client throws. Defaults to 300000.
 * @param config.codeExtractionStrategy - Strategy to extract code from the output:
 *   - "none": Use the raw output as-is
 *   - "llm": Use an LLM to extract code from the output
 *   - "markdown_code_blocks": Extract code from markdown code blocks
 * @param config.codeExtractor - Custom function to extract code from the output. Will override `codeExtractionStrategy` if provided.
 * @param config.model - LLM model name (only allowed when codeExtractionStrategy is "llm")
 * @param config.client - LangChain chat model instance (only allowed when codeExtractionStrategy is "llm")
 * @returns A function that runs TypeScript code and returns whether it succeeded
 */
export function createBoxdExecutionEvaluator(config: {
  boxdClient: Boxd;
  machineId: string;
  environmentVariables?: Record<string, string>;
  sandboxProjectDirectory?: string;
  timeoutMs?: number;
  codeExtractionStrategy?: "none" | "llm" | "markdown_code_blocks";
  codeExtractor?: (outputs: string | Record<string, unknown>) => string;
  model?: string;
  client?: BaseChatModel;
}) {
  if (
    config?.codeExtractionStrategy !== "llm" &&
    (config?.model || config?.client)
  ) {
    throw new Error(
      "model and client may only be passed if codeExtractionStrategy is 'llm'"
    );
  }
  _validateTimeoutMs(config.timeoutMs);
  const _scorer = async (params: { outputs: string }) => {
    const res = await _runInMachine({
      boxdClient: config.boxdClient,
      machineId: config.machineId,
      cwd: config.sandboxProjectDirectory ?? "openevals",
      files: {
        "outputs.ts": params.outputs,
        "openevals_package.json": PACKAGE_JSON_FILE_WITH_TSX,
        "extract_import_names.mjs": EXTRACT_IMPORT_NAMES,
      },
      command: BOXD_COMMAND,
      env: config.environmentVariables,
      timeoutMs: config.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    });
    if (res.exitCode === 0) {
      return [true, ""] as const;
    }
    return [false, res.stderr || res.stdout] as const;
  };
  return _createBaseCodeEvaluator<SingleResultScorerReturnType>({
    scorer: _scorer,
    runName: "boxd_execution_evaluator",
    feedbackKey: "execution_succeeded",
    codeExtractionStrategy: config.codeExtractionStrategy,
    codeExtractor: config.codeExtractor,
    model: config.model,
    client: config.client,
  });
}
