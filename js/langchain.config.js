import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * @param {string} relativePath
 * @returns {string}
 */
function abs(relativePath) {
  return resolve(dirname(fileURLToPath(import.meta.url)), relativePath);
}

export const config = {
  internals: [
    /node\:/,
    /js-tiktoken/,
    /langsmith/,
    /\@langchain\/core\/runnables/,
    /\@langchain\/core\/prompts/,
    /\@langchain\/core\/messages/,
    /langchain\/chat_models\/universal/,
    /\@langchain\/core\/utils\/json_schema/,
  ],
  entrypoints: {
    index: "index",
    "code/typescript": "code/typescript",
    "code/e2b": "code/e2b/index",
    llm: "llm",
    prompts: "prompts/index",
    types: "types",
    utils: "utils",
  },
  requiresOptionalDependency: [
    "code/typescript",
    "code/e2b",
  ],
  tsConfigPath: resolve("./tsconfig.json"),
  packageSuffix: "core",
  cjsSource: "./dist-cjs",
  cjsDestination: "./dist",
  abs,
}