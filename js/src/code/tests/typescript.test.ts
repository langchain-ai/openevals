import * as ls from "langsmith/vitest";

import { expect } from "vitest";

import { createTypeScriptEvaluator } from "../typescript.js";

ls.describe("TypeScript Evaluator", () => {
  ls.test.each([
    {
      inputs: {
        question: "Generate a function that returns the sum of two numbers.",
      },
      outputs: { content: "function sum(a, b) { return a + b; }" },
    },
    {
      inputs: {
        question: "Generate a function that returns the sum of two numbers.",
      },
      outputs: {
        content:
          "import { sum } from './sum';\nfunction add(a, b) { return sum(a, b); }",
      },
    },
    {
      inputs: {
        question: "Generate a function that returns the sum of two numbers.",
      },
      outputs: {
        content:
          "import { sum } from './sum';\nfunction sum(a, b) { const res = sum(a, b); res++; return res; }",
      },
      expectedScore: false,
    },
    {
      inputs: {
        question: "Generate a function that returns the sum of two numbers.",
      },
      outputs: {
        content: `
Sure! I'll help you write a function that returns the sum of two numbers. Initialize it like this:

\`\`\`typescript
function add(a, b) { return a + b; }
\`\`\`

Then, you can run it like this:

\`\`\`typescript
const res = add(1, 2);
console.log(res);
\`\`\`
`,
      },
      codeExtractionStrategy: "markdown_code_blocks",
    },
    {
      inputs: {
        question: "Generate a function that returns the sum of two numbers.",
      },
      outputs: {
        content: `
Sure! I'll help you write a function that returns the sum of two numbers. Initialize it like this:

\`\`\`typescript
function add(a, b) { return a + b; }
\`\`\`

Then, you can run it like this:

\`\`\`typescript
const res = add(1, 2);
console.log(res);
\`\`\`
`,
      },
      codeExtractionStrategy: "llm",
      model: "openai:o3-mini",
    },
  ])(
    "should pass basic type check",
    async ({
      inputs,
      outputs,
      expectedScore,
      codeExtractionStrategy,
      model,
    }) => {
      const evaluator = createTypeScriptEvaluator({
        codeExtractionStrategy,
        model,
      });

      const evalResult = await evaluator({ inputs, outputs });
      expect(evalResult.score).toBe(expectedScore ?? true);
    }
  );
});
