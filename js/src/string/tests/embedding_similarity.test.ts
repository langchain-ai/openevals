import * as ls from "langsmith/vitest";
import { expect } from "vitest";
import { createHash } from "node:crypto";
import { Embeddings } from "@langchain/core/embeddings";
import { OpenAIEmbeddings } from "@langchain/openai";

import { createEmbeddingSimilarityEvaluator } from "../embedding_similarity.js";

// Offline embeddings: identical text maps to an identical vector.
class HashEmbeddings extends Embeddings {
  constructor() {
    super({});
  }

  async embedQuery(text: string): Promise<number[]> {
    return Array.from(createHash("sha256").update(text).digest());
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map((text) => this.embedQuery(text)));
  }
}

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

  ls.test(
    "embedding similarity ignores object key order",
    { inputs: {} },
    async () => {
      const outputs = {
        city: "Paris",
        unit: "C",
        details: { lat: 48.85, lon: 2.35 },
      };
      const referenceOutputs = {
        unit: "C",
        details: { lon: 2.35, lat: 48.85 },
        city: "Paris",
      };
      const embeddingSimilarity = createEmbeddingSimilarityEvaluator({
        embeddings: new HashEmbeddings(),
      });
      const res = await embeddingSimilarity({ outputs, referenceOutputs });

      expect(res.key).toBe("embedding_similarity");
      expect(res.score).toBe(1.0);
    }
  );
});
