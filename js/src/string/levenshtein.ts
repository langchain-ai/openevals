import { EvaluatorResult } from "../types.js";
import { _runEvaluator } from "../utils.js";

function scorer(outputs: unknown, referenceOutputs: unknown): number {
  if (outputs === null || referenceOutputs === null) {
    throw new Error(
      "Levenshtein distance requires both outputs and reference_outputs"
    );
  }

  // Convert to strings if needed
  const outputStr =
    typeof outputs === "string" ? outputs : JSON.stringify(outputs);
  const referenceStr =
    typeof referenceOutputs === "string"
      ? referenceOutputs
      : JSON.stringify(referenceOutputs);

  // Create a matrix of size (m+1)x(n+1) where m and n are the string lengths
  const m = outputStr.length;
  const n = referenceStr.length;
  const dp: number[][] = Array(m + 1)
    .fill(null)
    .map(() => Array(n + 1).fill(0));

  // Initialize first row and column
  for (let i = 0; i <= m; i++) {
    dp[i][0] = i;
  }
  for (let j = 0; j <= n; j++) {
    dp[0][j] = j;
  }

  // Fill the matrix
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (outputStr[i - 1] === referenceStr[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1, // deletion
          dp[i][j - 1] + 1, // insertion
          dp[i - 1][j - 1] + 1 // substitution
        );
      }
    }
  }

  // Calculate the distance and normalize it to a score between 0 and 1
  const distance = dp[m][n];
  const maxLength = Math.max(m, n);
  return maxLength > 0 ? 1.0 - distance / maxLength : 1.0;
}

/**
 * Evaluates the actual output and reference output for similarity by Levenshtein distance.
 * @param options - The options object containing outputs and reference outputs
 * @returns EvaluatorResult containing match result with score between 0.0 and 1.0,
 * where 1.0 indicates an exact match and lower values indicate greater differences
 */
export async function levenshteinDistance(params: {
  outputs: unknown;
  referenceOutputs: unknown;
}): Promise<EvaluatorResult> {
  const { outputs, referenceOutputs } = params;

  const getScore = () => scorer(outputs, referenceOutputs);

  return _runEvaluator(
    "levenshtein_distance",
    getScore,
    "levenshtein_distance"
  );
}
