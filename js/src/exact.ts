import { _runEvaluator } from "./utils.js";

const _scorer = (params: { outputs: unknown; referenceOutputs: unknown }) => {
  const { outputs, referenceOutputs } = params;
  if (outputs === null || referenceOutputs === null) {
    throw new Error("Exact match requires both outputs and referenceOutputs");
  }
  const outputsJson = JSON.stringify(outputs, null, 2);
  const referenceOutputsJson = JSON.stringify(referenceOutputs, null, 2);
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
  referenceOutputs: unknown;
}) => {
  return _runEvaluator("exact_match", _scorer, "exact_match", params);
};
