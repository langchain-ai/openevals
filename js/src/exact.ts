import { _runEvaluator } from "./utils.js";

const _scorer = (params: { outputs: unknown; referenceOutputs?: unknown }) => {
  const { outputs, referenceOutputs } = params;
  if (outputs === null || referenceOutputs === null) {
    throw new Error("Exact match requires both outputs and referenceOutputs");
  }

  const processNestedStructures = (value: unknown): unknown => {
    if (value === undefined) {
      return null; // Convert undefined to null for consistent handling
    }
    if (Array.isArray(value)) {
      return value.map(processNestedStructures);
    }
    if (typeof value === "object" && value !== null) {
      return Object.fromEntries(
        Object.entries(value)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([k, v]) => [k, processNestedStructures(v)])
      );
    }
    return value;
  };

  const outputsJson = JSON.stringify(processNestedStructures(outputs));
  const referenceOutputsJson = JSON.stringify(
    processNestedStructures(referenceOutputs)
  );
  return outputsJson === referenceOutputsJson;
};

/**
 * Performs exact matching between input and output values.
 * @param outputs outputs to compare
 * @param referenceOutputs Reference outputs to compare
 * @returns
 */
export const exactMatch = async (params: {
  outputs: unknown;
  referenceOutputs?: unknown;
}) => {
  return _runEvaluator("exact_match", _scorer, "exact_match", params);
};
