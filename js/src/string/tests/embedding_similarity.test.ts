import * as ls from "langsmith/vitest";
import { expect } from "vitest";
import { evaluate } from "langsmith/evaluation";
import { OpenAIEmbeddings } from "@langchain/openai";

import { createEmbeddingSimilarityEvaluator } from "../embedding_similarity.js";

ls.describe("Embedding Similarity Tests", () => {
  ls.test(
    "embedding similarity with identical values",
    { inputs: {} },
    async () => {
      const outputs = { a: 1, b: 2 };
      const referenceOutputs = { a: 1, b: 2 };
      const embeddingSimilarity = createEmbeddingSimilarityEvaluator({
        embeddings: new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
      });
      const res = await embeddingSimilarity({ outputs, referenceOutputs });

      expect(res.key).toBe("embedding_similarity");
      expect(res.score).toBe(1.0);
    }
  );

  ls.test(
    "embedding similarity with different values",
    { inputs: {} },
    async () => {
      const outputs = { a: 1, b: 2 };
      const referenceOutputs = { a: 1, b: 3 };
      const embeddingSimilarity = createEmbeddingSimilarityEvaluator({
        embeddings: new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
      });
      const res = await embeddingSimilarity({ outputs, referenceOutputs });

      expect(res.key).toBe("embedding_similarity");
      expect(res.score).toBeLessThan(1.0);
    }
  );
});
