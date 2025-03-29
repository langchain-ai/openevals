import { Sandbox } from "@e2b/code-interpreter";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";

import { _createBaseCodeEvaluator } from "../base.js";
import { SingleResultScorerReturnType } from "../../types.js";
import {
  PACKAGE_JSON_FILE,
  TYPESCRIPT_EVALUATOR_FILE,
  EXTRACT_IMPORT_NAMES,
  TYPESCRIPT_EVALUATOR_SEPARATOR,
} from "./sandbox/files.js";

const E2B_COMMAND = [
  `echo '${PACKAGE_JSON_FILE}' > package.json`,
  `echo '${TYPESCRIPT_EVALUATOR_FILE}' > check_typescript.js`,
  `echo '${EXTRACT_IMPORT_NAMES}' > extract_import_names.js`,
  `ls node_modules/typescript 2>/dev/null || npm i`,
  `node extract_import_names.js | xargs -r -I {} sh -c 'ls node_modules/{} 2>/dev/null || npm install {} --prefer-offline'`,
  `node check_typescript.js`,
].join("&&");

/**
 * Creates an evaluator that checks TypeScript code for syntax and type errors,
 * including those used in dependencies, using E2B's sandboxing.
 *
 * @param config - Optional configuration object
 * @param config.sandbox - E2B sandbox instance to use for the evaluation.
 * @param config.sandboxProjectDirectory - Directory within the sandbox to use for the evaluation.
 *   Allows you to add additional sandbox project files ahead of time.
 * @param config.codeExtractionStrategy - Strategy to extract code from the output:
 *   - "none": Use the raw output as-is
 *   - "llm": Use an LLM to extract code from the output
 *   - "markdown_code_blocks": Extract code from markdown code blocks
 * @param config.codeExtractor - Custom function to extract code from the output. Will override `codeExtractionStrategy` if provided.
 * @param config.model - LLM model name (only allowed when codeExtractionStrategy is "llm")
 * @param config.client - LangChain chat model instance (only allowed when codeExtractionStrategy is "llm")
 * @returns A function that evaluates TypeScript code and returns whether it's valid
 */
export function createE2BTypeScriptEvaluator(config: {
  sandbox: Sandbox;
  sandboxProjectDirectory?: string;
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
  const _scorer = async (params: { outputs: string }) => {
    const cwd = config.sandboxProjectDirectory ?? "openevals";
    const sandbox = config.sandbox;
    try {
      await sandbox.files.write(`${cwd}/outputs.ts`, params.outputs);
      const res = await sandbox.commands.run(E2B_COMMAND, {
        cwd,
      });
      if (res.stdout.includes(TYPESCRIPT_EVALUATOR_SEPARATOR)) {
        return JSON.parse(res.stdout.split(TYPESCRIPT_EVALUATOR_SEPARATOR)[1]);
      }
      return [false, res.stdout];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (e: any) {
      if (e.result && typeof e.result.stderr === "string") {
        return [false, e.result.stderr];
      }
      throw e;
    }
  };
  return _createBaseCodeEvaluator<SingleResultScorerReturnType>({
    scorer: _scorer,
    runName: "e2b_typescript_evaluator",
    feedbackKey: "typescript_succeeded",
    ...config,
  });
}
