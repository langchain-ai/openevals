import * as ls from "langsmith/vitest";

import { expect } from "vitest";

import { createLLMAsJudge } from "../llm.js";
import { RAG_HELPFULNESS_PROMPT } from "../prompts/rag/helpfulness.js";
import { RAG_GROUNDEDNESS_PROMPT } from "../prompts/rag/groundedness.js";
import { RAG_RETRIEVAL_RELEVANCE_PROMPT } from "../prompts/rag/retrieval_relevance.js";



ls.describe("LLM as Judge RAG", () => {
  ls.test(
    "should test LLM judge RAG helpfulness",
    {
      inputs: {
        question: "Where was the first president of foobarland born?",
      },
    },
    async ({ inputs }) => {
      const outputs = {
        answer:
          "Bagatur Askaryan, the first president of FoobarLand, was born in Langchainville.",
      };

      const llmAsJudge = createLLMAsJudge({
        prompt: RAG_HELPFULNESS_PROMPT,
        feedbackKey: "helpfulness",
        model: "openai:gpt-5-mini",
      });

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
      });
      expect(evalResult.score).toBeTruthy();
    }
  );

  ls.test(
    "should test LLM judge RAG helpfulness not correct",
    {
      inputs: {
        question: "Where was the first president of foobarland born?",
      },
    },
    async ({ inputs }) => {
      const outputs = {
        answer: "The first president of foobarland was bagatur",
      };

      const llmAsJudge = createLLMAsJudge({
        prompt: RAG_HELPFULNESS_PROMPT,
        feedbackKey: "helpfulness",
        model: "openai:gpt-5-mini",
      });

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
      });
      expect(evalResult.score).toBeFalsy();
    }
  );

  ls.test(
    "should test LLM judge RAG groundedness",
    {
      inputs: {},
    },
    async () => {
      const retrievalEvaluator = createLLMAsJudge({
        prompt: RAG_GROUNDEDNESS_PROMPT,
        feedbackKey: "groundedness",
        model: "openai:gpt-5-mini",
      });

      const context = {
        documents: [
          "FoobarLand is a new country located on the dark side of the moon",
          "Space dolphins are native to FoobarLand",
          "FoobarLand is a constitutional democracy whose first president was Bagatur Askaryan",
          "The current weather in FoobarLand is 80 degrees and clear.",
        ],
      };

      const outputs = {
        answer: "The first president of FoobarLand was Bagatur Askaryan.",
      };

      const evalResult = await retrievalEvaluator({
        context,
        outputs,
      });
      expect(evalResult.score).toBeTruthy();
    }
  );

  ls.test(
    "should test LLM judge RAG retrieval relevance",
    {
      inputs: {
        question: "Where was the first president of FoobarLand born?",
      },
    },
    async ({ inputs }) => {
      const retrievalRelevanceEvaluator = createLLMAsJudge({
        prompt: RAG_RETRIEVAL_RELEVANCE_PROMPT,
        feedbackKey: "retrieval_relevance",
        model: "openai:gpt-5-mini",
      });

      const context = {
        documents: [
          "The Eiffel Tower was constructed between 1887 and 1889 in Paris, France.",
          "Photosynthesis is the process by which plants convert sunlight into energy.",
          "The Amazon River is the largest river in the world by discharge volume.",
          "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        ],
      };

      const evalResult = await retrievalRelevanceEvaluator({
        inputs,
        context,
      });
      expect(evalResult.score).toBeFalsy();
    }
  );
});
