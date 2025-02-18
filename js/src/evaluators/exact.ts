import { _runEvaluator } from "./utils.js";

const _scorer = (params: { inputs: unknown; outputs: unknown }) => {
  const { inputs, outputs } = params;
  if (inputs === null || outputs === null) {
    throw new Error("Exact match requires both inputs and outputs");
  }
  const inputsJson = JSON.stringify(inputs, null, 2);
  const outputsJson = JSON.stringify(outputs, null, 2);
  return inputsJson === outputsJson;
};

/**
 * Performs exact matching between input and output values.
 * @param inputs Inputs to compare
 * @param outputs Outputs to compare
 * @returns
 */
export const exactMatch = async (params: {
  inputs: unknown;
  outputs: unknown;
}) => {
  return _runEvaluator("exact_match", _scorer, "exact_match", params);
};
