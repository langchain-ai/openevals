import * as ls from "langsmith/vitest";

import { expect } from "vitest";

import { createCodeLLMAsJudge } from "../llm.js";
import { CODE_CORRECTNESS_PROMPT } from "../../prompts/code_correctness.js";

ls.describe("Code LLM As Judge", () => {
  ls.test.each([
    {
      inputs: {
        question: "Generate a function that returns the sum of two numbers.",
      },
      outputs: { content: "function sum(a, b) { return a + b; }" },
    },
    {
      inputs: {
        question: "Generate a working web server in Express.js.",
      },
      outputs: {
        content:
          "import express from 'express';\nconst app = express();\napp.get('/', (req, res) => { res.send('Hello World'); });\napp.listen(3000, () => { console.log('Server is running on port 3000'); });",
      },
    },
    {
      inputs: {
        question: "Generate a working web server in Express.js.",
      },
      outputs: {
        content:
          "import express from 'express';\nconst app = express();\napp.wget('/', (req, res) => { res.send('Hello World'); });\napp.listen(3000, () => { console.log('Server is running on port 3000'); });",
      },
      expectedScore: false,
    },
    {
      inputs: {
        question: `Add proper TypeScript types to the following code:

\`\`\`typescript
function add(a, b) { return a + b; }
\`\`\`
`,
      },
      outputs: {
        content: `
\`\`\`typescript
function add(a: number, b: number): boolean {
  return a + b;
}
\`\`\`
`,
      },
      expectedScore: false,
    },
  ])(
    "should pass basic type check",
    async ({ inputs, outputs, expectedScore }) => {
      const evaluator = createCodeLLMAsJudge({
        prompt: CODE_CORRECTNESS_PROMPT,
        model: "openai:o4-mini",
      });

      const evalResult = await evaluator({ inputs, outputs });
      expect(evalResult.score).toBe(expectedScore ?? true);
    }
  );
});
