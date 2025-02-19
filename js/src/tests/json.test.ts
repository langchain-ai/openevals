import * as ls from "langsmith/vitest";
import { expect } from "vitest";

import { OpenAI } from "openai";
import { ChatOpenAI } from "@langchain/openai";

import { createJsonMatchEvaluator } from "../json/match.js";

ls.describe("json", () => {
  ls.test(
    "test json match base",
    {
      inputs: { a: 1, b: 2 },
      referenceOutputs: { a: 1, b: 2 },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.length).toBe(2);
      expect(result[0].key).toBe("a");
      expect(result[0].score).toBe(1.0);
      expect(result[1].key).toBe("b");
      expect(result[1].score).toBe(1.0);
    }
  );

  ls.test(
    "test json match mix",
    {
      inputs: { a: "Mango, Bananas", b: 2 },
      referenceOutputs: { a: "Bananas, Mango", b: 1 },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "average",
        rubric: {
          a: "Does the answer mention all the fruits in the reference answer?",
        },
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0.5);
    }
  );

  ls.test(
    "test json match average",
    {
      inputs: { a: 1, b: 2 },
      referenceOutputs: { a: 1, b: 1 },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "average",
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0.5);
    }
  );

  ls.test(
    "test json match exclude",
    {
      inputs: { a: 1, b: 2 },
      referenceOutputs: { a: 1, b: 1 },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "average",
        excludeKeys: ["b"],
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(1.0);
    }
  );

  ls.test(
    "test json match all",
    {
      inputs: { a: 1, b: 2 },
      referenceOutputs: { a: 1, b: 1 },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "all",
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0);
    }
  );

  ls.test(
    "test json match rubric",
    {
      inputs: {
        name: "Harrison Chase",
        description:
          "CEO of LangChain, used to work at Kensho + Robust Intelligence.",
      },
      referenceOutputs: {
        name: "Harrison Chase",
        description:
          "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence.",
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "all",
        rubric: {
          description:
            "Is the correct title and company mentioned, as well as all previous companies?",
        },
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(1);
    }
  );

  ls.test(
    "test json match rubric wrong",
    {
      inputs: {
        name: "Harrison Chase",
        description: "CEO of LangChain, used to work at Kensho.",
      },
      referenceOutputs: {
        name: "Harrison Chase",
        description:
          "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence.",
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "all",
        rubric: {
          description:
            "Is the correct title and company mentioned, as well as all previous companies?",
        },
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0);
    }
  );

  ls.test(
    "test json match rubric with reasoning",
    {
      inputs: {
        description: "CEO of LangChain, used to work at Kensho.",
      },
      referenceOutputs: {
        description:
          "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence.",
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        rubric: {
          description:
            "Is the correct title and company mentioned, as well as all previous companies?",
        },
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0);
      expect(result.comment).toBeDefined();
    }
  );

  ls.test(
    "test json match rubric without reasoning",
    {
      inputs: {
        description: "CEO of LangChain, used to work at Kensho.",
      },
      referenceOutputs: {
        description:
          "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence.",
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        rubric: {
          description:
            "Is the correct title and company mentioned, as well as all previous companies?",
        },
        useReasoning: false,
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0);
      expect(result.comment).toBeUndefined();
    }
  );

  ls.test(
    "test json match rubric with reasoning individual keys",
    {
      inputs: {
        name: "Harrison Chase",
        description: "CEO of LangChain, used to work at Kensho.",
      },
      referenceOutputs: {
        name: "Harrison Chase",
        description:
          "Harrison chase is the CEO of LangChain. He used to work at Kensho and Robust Intelligence.",
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        rubric: {
          description:
            "Is the correct title and company mentioned, as well as all previous companies?",
        },
      });
      const result = await evaluator({ outputs, referenceOutputs });
      expect(result).toBeDefined();
      expect(result.length).toBe(2);
      expect(result[0].key).toBe("name");
      expect(result[0].score).toBe(1.0);
      expect(result[1].key).toBe("description");
      expect(result[1].score).toBe(0);
      expect(result[1].comment).toBeDefined();
    }
  );

  ls.test(
    "test json match list all none",
    {
      inputs: {
        inputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.length).toBe(2);
      expect(result[0].key).toBe("a");
      expect(result[0].score).toBe(1.0);
      expect(result[1].key).toBe("b");
      expect(result[1].score).toBe(1.0);
    }
  );

  ls.test(
    "test json match list average none",
    {
      inputs: {
        inputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.length).toBe(2);
      expect(result[0].key).toBe("a");
      expect(result[0].score).toBe(1.0);
      expect(result[1].key).toBe("b");
      expect(result[1].score).toBe(0.5);
    }
  );

  ls.test(
    "test json match list all all",
    {
      inputs: {
        inputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "all",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(1.0);
    }
  );

  ls.test(
    "test json match list average all",
    {
      inputs: {
        inputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "all",
        listAggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0.5);
    }
  );

  ls.test(
    "test json match list all average",
    {
      inputs: {
        inputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0.0);
    }
  );

  ls.test(
    "test json match list average average",
    {
      inputs: {
        inputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "average",
        listAggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0.75);
    }
  );

  ls.test(
    "test json match list mismatch all none",
    {
      inputs: {
        inputs: [
          { a: 1, d: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2, c: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
      });
      let result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      result = result.sort((a, b) => a.key.localeCompare(b.key));
      expect(result).toBeDefined();
      expect(result.length).toBe(4);
      expect(result[0].key).toBe("a");
      expect(result[0].score).toBe(1.0);
      expect(result[1].key).toBe("b");
      expect(result[1].score).toBe(0);
      expect(result[2].key).toBe("c");
      expect(result[2].score).toBe(0);
      expect(result[3].key).toBe("d");
      expect(result[3].score).toBe(0);
    }
  );

  ls.test(
    "test json match list mismatch average none",
    {
      inputs: {
        inputs: [
          { a: 1, d: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2, c: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
      });
      let result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      result = result.sort((a, b) => a.key.localeCompare(b.key));
      expect(result).toBeDefined();
      expect(result.length).toBe(4);
      expect(result[0].key).toBe("a");
      expect(result[0].score).toBe(1.0);
      expect(result[1].key).toBe("b");
      expect(result[1].score).toBe(0.5);
      expect(result[2].key).toBe("c");
      expect(result[2].score).toBe(0);
      expect(result[3].key).toBe("d");
      expect(result[3].score).toBe(0);
    }
  );

  ls.test(
    "test json match list mismatch all all",
    {
      inputs: {
        inputs: [
          { a: 1, d: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2, c: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "all",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0);
    }
  );

  ls.test(
    "test json match list mismatch average all",
    {
      inputs: {
        inputs: [
          { a: 1, d: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2, c: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "all",
        listAggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0);
    }
  );

  ls.test(
    "test json match list mismatch all average",
    {
      inputs: {
        inputs: [
          { a: 1, d: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2, c: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0);
    }
  );

  ls.test(
    "test json match list mismatch average average",
    {
      inputs: {
        inputs: [
          { a: 1, d: 2 },
          { a: 1, b: 2 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2 },
          { a: 1, b: 2, c: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        aggregator: "average",
        listAggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0.5);
    }
  );

  ls.test(
    "test json match list rubric",
    {
      inputs: { inputs: [{ a: "Strawberries, Melons, Bananas" }] },
      referenceOutputs: {
        referenceOutputs: [{ a: "Bananas, Strawberries, Melons" }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        rubric: {
          a: "Does the answer mention all the fruits in the reference answer?",
        },
        listAggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("a");
      expect(result.score).toBe(1);
    }
  );

  ls.test(
    "test json match list mismatch output missing",
    {
      inputs: {
        inputs: [
          { a: 1, b: 2, d: 3 },
          { a: 1, b: 2, c: 3 },
          { a: 1, b: 2, d: 3 },
        ],
      },
      referenceOutputs: {
        referenceOutputs: [
          { a: 1, b: 2, d: 3 },
          { a: 1, b: 2, c: 3 },
          { a: 1, b: 2, c: 3 },
        ],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(5 / 6);
    }
  );

  ls.test(
    "test json match mode exact extra reference",
    {
      inputs: { inputs: [{ a: 1 }, { a: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ a: 1 }, { a: 1 }, { a: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(2 / 3);
    }
  );

  ls.test(
    "test json match mode exact extra output",
    {
      inputs: { inputs: [{ a: 1 }, { a: 1 }, { a: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ a: 1 }, { a: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(2 / 3);
    }
  );

  ls.test(
    "test json match mode exact unordered",
    {
      inputs: { inputs: [{ a: 1, d: 2, e: 2 }, { b: 1 }, { c: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ b: 1, d: 2, e: 2 }, { a: 1 }, { c: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
        excludeKeys: ["d", "e"],
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(1);
    }
  );

  ls.test(
    "test json match mode subset outputs",
    {
      inputs: { inputs: [{ a: 1 }, { b: 1 }, { c: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ b: 1 }, { a: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
        listMatchMode: "superset",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(1);
    }
  );

  ls.test(
    "test json match mode subset reference",
    {
      inputs: { inputs: [{ a: 1 }, { b: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ b: 1 }, { c: 1 }, { a: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
        listMatchMode: "subset",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(1);
    }
  );

  ls.test(
    "test json match mode subset reference",
    {
      inputs: { inputs: [{ a: 1 }, { b: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ b: 1 }, { c: 1 }, { a: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
        listMatchMode: "subset",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(1);
    }
  );

  ls.test(
    "test json match mode order wrong",
    {
      inputs: { inputs: [{ a: 1 }, { b: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ b: 1 }, { a: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
        listMatchMode: "ordered",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(0);
    }
  );

  ls.test(
    "test json match mode order right",
    {
      inputs: { inputs: [{ a: 1 }, { b: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ a: 1 }, { b: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
        listMatchMode: "ordered",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(1);
    }
  );

  ls.test(
    "test json match mode order wrong language",
    {
      inputs: { inputs: [{ a: 1, c: "The Dog" }, { b: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ a: 1, c: "El Perro" }, { b: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
        rubric: { c: "Are the answers the same, language independent?" },
        listMatchMode: "ordered",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(1);
    }
  );

  ls.test(
    "test json match mode order",
    {
      inputs: { inputs: [{ a: 1 }, { b: 1 }, { c: 1 }] },
      referenceOutputs: {
        referenceOutputs: [{ a: 1 }, { b: 1 }, { d: 1 }],
      },
    },
    async ({ inputs, referenceOutputs }) => {
      const outputs = inputs.inputs;
      const client = new OpenAI();
      const evaluator = createJsonMatchEvaluator({
        judge: client,
        model: "openai:o3-mini",
        listAggregator: "average",
        aggregator: "average",
        rubric: { c: "Are the answers the same, language independent?" },
        listMatchMode: "ordered",
      });
      const result = await evaluator({
        outputs,
        referenceOutputs: referenceOutputs.referenceOutputs,
      });
      expect(result).toBeDefined();
      expect(result.key).toBe("structured_match_score");
      expect(result.score).toBe(2 / 3);
    }
  );
});
