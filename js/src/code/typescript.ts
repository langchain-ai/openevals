import { BaseChatModel } from "@langchain/core/language_models/chat_models";

import ts from "typescript";
import * as fs from "fs";
import os from "node:os";
import { v4 as uuidv4 } from "uuid";

import { SingleResultScorerReturnType } from "../types.js";

import { _createBaseCodeEvaluator } from "./base.js";

const _typescriptScorer = (params: {
  outputs: string;
  [key: string]: unknown;
}) => {
  const { outputs } = params;
  try {
    const tempFilepath = os.tmpdir() + `/temp-${uuidv4()}.ts`;
    fs.writeFileSync(tempFilepath, outputs);
    const program = ts.createProgram([tempFilepath], {
      module: ts.ModuleKind.NodeNext,
      moduleResolution: ts.ModuleResolutionKind.NodeNext,
      target: ts.ScriptTarget.ES2021,
      alwaysStrict: true,
      skipLibCheck: true,
    });
    const diagnostics = ts
      .getPreEmitDiagnostics(program)
      .filter(
        (diagnostic) =>
          !ts
            .flattenDiagnosticMessageText(diagnostic.messageText, "")
            .includes("Cannot find module")
      );
    const issueStrings: string[] = [];
    if (diagnostics.length === 0) {
      return true;
    } else {
      diagnostics.forEach((diagnostic) => {
        if (diagnostic.file) {
          const { line, character } =
            diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start!);
          const message = ts.flattenDiagnosticMessageText(
            diagnostic.messageText,
            "\n"
          );
          issueStrings.push(`(${line + 1},${character + 1}): ${message}`);
        } else {
          issueStrings.push(
            ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n")
          );
        }
      });
      if (issueStrings.length) {
        const issues = issueStrings.join("\n");
        return [false, issues] as const;
      }
      return true;
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } catch (e: any) {
    return [false, e.message] as const;
  }
};

export function createTypeScriptEvaluator(config?: {
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
  const _scorer = (params: { outputs: string; [key: string]: unknown }) => {
    return _typescriptScorer(params);
  };
  return _createBaseCodeEvaluator<SingleResultScorerReturnType>({
    scorer: _scorer,
    runName: "typescript_evaluator",
    feedbackKey: "typescript_succeeded",
    ...config,
  });
}
