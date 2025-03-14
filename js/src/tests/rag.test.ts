import * as ls from "langsmith/vitest";

import { expect } from "vitest";

import { createLLMAsJudge } from "../llm.js";
import { RAG_HALLUCATION_PROMPT } from "../prompts/rag_hallucination.js";
import { RETRIEVAL_HELPFULNESS_PROMPT} from "../prompts/rag_retrieval.js"; 

ls.describe("LLM Judge Hallucination", () => {
  ls.test(
    "should pass hallucination check for non-hallucinated answer",
    {
      inputs: {
        question:
          "Who was the first president of the Star Republic of Oiewjoie?",
      },
    },
    async ({ inputs }) => {
      const outputs = { answer: "Bzkeoei Ahbeijo" };
      const context =
        "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo.";

      const llmAsJudge = createLLMAsJudge({
        prompt: RAG_HALLUCATION_PROMPT,
        feedbackKey: "hallucination",
        model: "openai:o3-mini",
      });

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
        context,
        referenceOutputs: "",
      });
      expect(evalResult.score).toBeTruthy();
    }
  );

  ls.test(
    "should fail hallucination check for hallucinated answer",
    {
      inputs: {
        question:
          "Who was the first president of the Star Republic of Oiewjoie?",
      },
    },
    async ({ inputs }) => {
      const outputs = {
        answer: "John Adams",
      };
      const context =
        "The Star Republic of Oiewjoie is a country that exists in the universe. The first president of the Star Republic of Oiewjoie was Bzkeoei Ahbeijo.";

      const llmAsJudge = createLLMAsJudge({
        prompt: RAG_HALLUCATION_PROMPT,
        feedbackKey: "hallucination",
        model: "openai:o3-mini",
      });

      await expect(llmAsJudge({ inputs, outputs })).rejects.toThrow();

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
        context,
        referenceOutputs: "",
      });
      expect(evalResult.score).toBeFalsy();
    }
  );


  ls.test(
    "should pass retrieval for relevant docs",
    {
      inputs: {
        question:
          "Where was the first president of foobarland born?",
      },
    },
    async ({ inputs }) => {
      const outputs = [
        {
            title: "foobarland president",
            content: "the first president of foobarland was bagatur"
        },
        {
            title: "bagatur bio",
            content: "bagutur was born in langchainland"
        }
      ]

      const llmAsJudge = createLLMAsJudge({
        prompt: RETRIEVAL_HELPFULNESS_PROMPT,
        feedbackKey: "hallucination",
        model: "openai:o3-mini",
      });

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
      });
      expect(evalResult.score).toBeTruthy();
    }
  );

  ls.test(
    "should fail retrieval for irrelevant docs",
    {
      inputs: {
        question:
          "Where was the first president of foobarland born?",
      },
    },
    async ({ inputs }) => {
      const outputs = [
        {
            title: "foobarland president",
            content: "the first president of foobarland was bagatur"
        },
        {
            title: "bagatur bio",
            content: "bagutur is a big fan of PR reviews"
        }
      ]

      const llmAsJudge = createLLMAsJudge({
        prompt: RETRIEVAL_HELPFULNESS_PROMPT,
        feedbackKey: "hallucination",
        model: "openai:o3-mini",
      });

      const evalResult = await llmAsJudge({
        inputs,
        outputs,
      });
      expect(evalResult.score).toBeFalsy();
    }
  );
});
