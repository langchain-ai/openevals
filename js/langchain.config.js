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
  ],
  entrypoints: {
    index: "index",
    "code/typescript": "code/typescript",
    llm: "llm",
    types: "types",
    utils: "utils",
  },
  tsConfigPath: resolve("./tsconfig.json"),
  packageSuffix: "core",
  cjsSource: "./dist-cjs",
  cjsDestination: "./dist",
  abs,
}