import { Embeddings } from "@langchain/core/embeddings";

import { EvaluatorResult } from "../types.js";
import { _runEvaluator } from "../utils.js";

// Helper functions for vector calculations
const handleEmbeddingOutputs = (
  algorithm: string,
  receivedEmbedding: number[],
  expectedEmbedding: number[]
): number => {
  const dotProduct = (v1: number[], v2: number[]): number =>
    v1.reduce((sum, a, i) => sum + a * v2[i], 0);

  const vectorMagnitude = (v: number[]): number =>
    Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));

  const cosineSimilarity = (v1: number[], v2: number[]): number => {
    const dotProd = dotProduct(v1, v2);
    const magnitude1 = vectorMagnitude(v1);
    const magnitude2 = vectorMagnitude(v2);
    return dotProd / (magnitude1 * magnitude2);
  };

  // Calculate similarity based on chosen algorithm
  const similarity =
    algorithm === "cosine"
      ? cosineSimilarity(receivedEmbedding, expectedEmbedding)
      : dotProduct(receivedEmbedding, expectedEmbedding);

  return Number(similarity.toFixed(2));
};

interface EmbeddingSimilarityOptions {
  embeddings: Embeddings;
  algorithm?: "cosine" | "dot_product";
}

/**
 * Creates an evaluator that compares the actual output and reference output for similarity by text embedding distance.
 * @param {Object} options - The configuration options
 * @param {Embeddings} options.embeddings - The embeddings model to use for similarity comparison
 * @param {('cosine'|'dot_product')} [options.algorithm='cosine'] - The algorithm to use for embedding similarity
 * @returns An evaluator that returns a score representing the embedding similarity
 */
export const createEmbeddingSimilarityEvaluator = ({
  embeddings,
  algorithm = "cosine",
}: EmbeddingSimilarityOptions) => {
  if (algorithm !== "cosine" && algorithm !== "dot_product") {
    throw new Error(
      `Unsupported algorithm: ${algorithm}. Only 'cosine' and 'dot_product' are supported.`
    );
  }

  return async (params: {
    outputs: unknown;
    referenceOutputs?: unknown;
  }): Promise<EvaluatorResult> => {
    const { outputs, referenceOutputs } = params;
    if (outputs == null || referenceOutputs == null) {
      throw new Error(
        "Embedding similarity requires both outputs and referenceOutputs"
      );
    }

    const outputString =
      typeof outputs === "string" ? outputs : JSON.stringify(outputs);
    const referenceOutputString =
      typeof referenceOutputs === "string"
        ? referenceOutputs
        : JSON.stringify(referenceOutputs);

    const getScore = async (): Promise<number> => {
      const receivedEmbedding = await embeddings.embedQuery(outputString);
      const expectedEmbedding = await embeddings.embedQuery(
        referenceOutputString
      );

      return handleEmbeddingOutputs(
        algorithm,
        receivedEmbedding,
        expectedEmbedding
      );
    };

    return _runEvaluator(
      "embedding_similarity",
      getScore,
      "embedding_similarity"
    );
  };
};
