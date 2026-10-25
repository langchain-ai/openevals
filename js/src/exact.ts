import { _runEvaluator, _sortKeysDeep } from "./utils.js";

const _scorer = (params: { outputs: unknown; referenceOutputs?: unknown }) => {
  const { outputs, referenceOutputs } = params;
  if (
    outputs === null ||
    outputs === undefined ||
    referenceOutputs === null ||
    referenceOutputs === undefined
  ) {
    throw new Error("Exact match requires both outputs and referenceOutputs");
  }

  const outputsJson = JSON.stringify(_sortKeysDeep(outputs));
  const referenceOutputsJson = JSON.stringify(_sortKeysDeep(referenceOutputs));
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
