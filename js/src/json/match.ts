import { BaseChatModel } from "@langchain/core/language_models/chat_models";

import { ModelClient, MultiResultScorerReturnType } from "../types.js";
import { _createLLMAsJudgeScorer } from "../llm.js";
import { _runEvaluator } from "../utils.js";
import { getCurrentRunTree } from "langsmith/traceable";

type AggregatorType = "average" | "all" | undefined;
type ListMatchMode = "superset" | "subset" | "same_elements" | "ordered";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type RecordStringAny = Record<string, any>;

const SYSTEM_PROMPT = `You are an LLM that evaluates the accuracy of structured outputs.
Make sure to evaluate each key the users ask you to evaluate separately. Assign the score
for each key based on its own criteria - DO NOT convolute the scores of different keys.
Also only evaluate the output vs. the reference output based on the criteria. DO NOT EVALUATE
BASED ON ANYTHING ELSE. If the output does not match the reference output in some way that
is not mentioned in the criteria that is not a problem and you should ignore those discrepancies.
Only focus on finding discrepancies based on the criteria. If there is a None value being compared
to a non-None value, you should assign a score of 0.`;

const USER_PROMPT = `Please evaluate the accuracy of the following output keys according to these criteria:
{rubric}
<Outputs>
{outputs}
</Outputs>
<Expected Outputs>
{reference_outputs}
</Expected Outputs>`;

function _prepareParameters({
  outputs,
  referenceOutputs,
  rubric,
  excludeKeys,
  useReasoning,
  listMatchMode = "same_elements",
}: {
  outputs: RecordStringAny[];
  referenceOutputs: RecordStringAny[];
  rubric: Record<string, string>;
  excludeKeys: string[];
  useReasoning: boolean;
  listMatchMode: ListMatchMode;
}): {
  processedOutputs: RecordStringAny;
  processedReferenceOutputs: RecordStringAny;
  jsonSchema: RecordStringAny;
  scores: RecordStringAny;
  formattedRubric: string;
  useListReducer: boolean;
} {
  const jsonSchema: RecordStringAny = {
    type: "object",
    title: "json_match",
    description: "Scores measuring the accuracy of structured outputs",
    properties: {},
    required: [],
    additionalProperties: false,
  };

  const scores: RecordStringAny = {};
  let formattedRubric = "";
  let useListReducer = false;
  let processedOutputs: RecordStringAny = {};
  let processedReferenceOutputs: RecordStringAny = {};

  if (Array.isArray(outputs)) {
    useListReducer = true;
    if (!Array.isArray(referenceOutputs)) {
      throw new Error(
        "If outputs is a list, referenceOutputs must also be a list"
      );
    }

    const outputsToUse: RecordStringAny = {};
    const referenceOutputsToUse: RecordStringAny = {};

    if (listMatchMode === "ordered") {
      outputs.forEach((output, i) => {
        Object.entries(output).forEach(([key, value]) => {
          outputsToUse[`${key}_${i}`] = value;
        });
      });
      referenceOutputs.forEach((refOutput, i) => {
        Object.entries(refOutput).forEach(([key, value]) => {
          referenceOutputsToUse[`${key}_${i}`] = value;
        });
      });
    } else if (listMatchMode === "superset") {
      const availableOutputs = Array.from(Array(outputs.length).keys());
      const matchedReferences = new Set<number>();

      referenceOutputs.forEach((refItem, i) => {
        let bestMatchScore = -1;
        let bestMatchIdx: number | undefined;

        availableOutputs.forEach((outIdx) => {
          const outputItem = outputs[outIdx];
          let matchScore = 0;

          Object.keys(refItem).forEach((key) => {
            if (
              key in outputItem &&
              !excludeKeys.includes(key) &&
              !(key in rubric)
            ) {
              matchScore += Number(refItem[key] === outputItem[key]);
            }
          });

          if (matchScore > bestMatchScore) {
            bestMatchScore = matchScore;
            bestMatchIdx = outIdx;
          }
        });

        if (bestMatchIdx !== undefined) {
          Object.entries(outputs[bestMatchIdx]).forEach(([key, value]) => {
            outputsToUse[`${key}_${i}`] = value;
          });

          Object.entries(refItem).forEach(([key, value]) => {
            referenceOutputsToUse[`${key}_${i}`] = value;
          });

          availableOutputs.splice(availableOutputs.indexOf(bestMatchIdx), 1);
          matchedReferences.add(i);
        } else {
          Object.entries(refItem).forEach(([key, value]) => {
            referenceOutputsToUse[`${key}_${i}`] = value;
          });
        }
      });
    } else {
      const availableReferences = Array.from(
        Array(referenceOutputs.length).keys()
      );
      const matchedOutputs = new Set<number>();

      outputs.forEach((outputItem, i) => {
        let bestMatchIdx: number | undefined;
        let bestMatchScore = -1;

        availableReferences.forEach((refIdx) => {
          const refItem = referenceOutputs[refIdx];
          let matchScore = 0;

          Object.keys(outputItem).forEach((key) => {
            if (
              key in refItem &&
              !excludeKeys.includes(key) &&
              !(key in rubric)
            ) {
              matchScore += Number(outputItem[key] === refItem[key]);
            }
          });

          if (matchScore > bestMatchScore) {
            bestMatchScore = matchScore;
            bestMatchIdx = refIdx;
          }
        });

        if (bestMatchIdx !== undefined) {
          Object.entries(outputItem).forEach(([key, value]) => {
            outputsToUse[`${key}_${i}`] = value;
          });

          Object.entries(referenceOutputs[bestMatchIdx]).forEach(
            ([key, value]) => {
              referenceOutputsToUse[`${key}_${i}`] = value;
            }
          );

          availableReferences.splice(
            availableReferences.indexOf(bestMatchIdx),
            1
          );
          matchedOutputs.add(i);
        } else {
          Object.entries(outputItem).forEach(([key, value]) => {
            outputsToUse[`${key}_${i}`] = value;
          });
        }
      });

      if (listMatchMode === "same_elements") {
        availableReferences.forEach((refIdx, index) => {
          const refItem = referenceOutputs[refIdx];
          const dummyIdx = outputs.length + index;
          Object.entries(refItem).forEach(([key, value]) => {
            referenceOutputsToUse[`${key}_${dummyIdx}`] = value;
          });
        });
      }
    }

    processedOutputs = outputsToUse;
    processedReferenceOutputs = referenceOutputsToUse;
  } else {
    processedOutputs = outputs;
    processedReferenceOutputs = referenceOutputs;
  }

  Object.entries(processedOutputs).forEach(([rawKey, value]) => {
    const key = useListReducer
      ? rawKey.substring(0, rawKey.lastIndexOf("_"))
      : rawKey;

    if (excludeKeys.includes(key)) {
      return;
    }

    if (!(rawKey in processedReferenceOutputs)) {
      scores[rawKey] = 0;
      return;
    }

    if (!(key in rubric) && processedReferenceOutputs[rawKey] === value) {
      scores[rawKey] = 1;
    } else if (!(key in rubric)) {
      scores[rawKey] = 0;
    } else {
      const keyCriteria = rubric[key];
      formattedRubric += `Key: ${key}, Criteria: ${keyCriteria}\n`;

      if (!useReasoning) {
        jsonSchema.properties[rawKey] = {
          type: "boolean",
          description: `Does the output for key ${key}, follow the criteria? ${keyCriteria}`,
        };
        jsonSchema.required.push(rawKey);
      } else {
        jsonSchema.properties[rawKey] = {
          type: "object",
          properties: {
            reasoning: {
              type: "string",
              description: `Reasoning for the score you assigned to key ${key}`,
            },
            score: {
              type: "boolean",
              description: `Does the output for key ${key}, follow the criteria? ${keyCriteria}`,
            },
          },
          required: ["score", "reasoning"],
          additionalProperties: false,
        };
        jsonSchema.required.push(rawKey);
      }
    }
  });

  Object.entries(processedReferenceOutputs).forEach(([rawKey, _]) => {
    const key = useListReducer
      ? rawKey.substring(0, rawKey.lastIndexOf("_"))
      : rawKey;
    if (!excludeKeys.includes(key) && !(rawKey in processedOutputs)) {
      scores[rawKey] = 0;
    }
  });

  return {
    processedOutputs,
    processedReferenceOutputs,
    jsonSchema,
    scores,
    formattedRubric,
    useListReducer,
  };
}

function _aggregateResults({
  scores,
  useListReducer,
  aggregator,
  listAggregator,
  scoreKey,
}: {
  scores: RecordStringAny;
  useListReducer: boolean;
  aggregator: AggregatorType;
  listAggregator: "average" | "all";
  scoreKey: string;
}): MultiResultScorerReturnType {
  if (useListReducer) {
    const indexGroupedScores: Record<string, RecordStringAny> = {};

    Object.entries(scores).forEach(([k, v]) => {
      const index = k.substring(k.lastIndexOf("_") + 1);
      const baseKey = k.substring(0, k.lastIndexOf("_"));

      if (!indexGroupedScores[index]) {
        indexGroupedScores[index] = {};
      }
      indexGroupedScores[index][baseKey] = v;
    });

    let indexScores: RecordStringAny = {};

    if (aggregator === "average") {
      Object.entries(indexGroupedScores).forEach(([index, group]) => {
        if (Object.keys(group).length) {
          const total = Object.values(group).reduce(
            (sum, v) =>
              sum + (typeof v === "object" ? Number(v.score) : Number(v)),
            0
          );
          indexScores[index] = total / Object.keys(group).length;
        }
      });
    } else if (aggregator === "all") {
      Object.entries(indexGroupedScores).forEach(([index, group]) => {
        if (Object.keys(group).length) {
          const hasNonOne = Object.values(group).some(
            (v) => (typeof v === "object" ? Number(v.score) : Number(v)) !== 1
          );
          indexScores[index] = hasNonOne ? 0 : 1;
        }
      });
    } else {
      indexScores = indexGroupedScores;
    }

    if (listAggregator === "average") {
      if (Object.values(indexScores).every((v) => typeof v === "number")) {
        const score = Object.keys(indexScores).length
          ? Object.values(indexScores).reduce(
              (a, b) => Number(a) + Number(b),
              0
            ) / Object.keys(indexScores).length
          : 0;
        return { [[scoreKey, aggregator].join(":")]: score };
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const scoresAggregatedAcrossList: Record<string, any[]> = {};
      Object.values(indexScores).forEach((group) => {
        Object.entries(group).forEach(([key, value]) => {
          if (!scoresAggregatedAcrossList[key]) {
            scoresAggregatedAcrossList[key] = [];
          }
          scoresAggregatedAcrossList[key].push(value);
        });
      });

      const result: Record<string, number> = {};
      Object.entries(scoresAggregatedAcrossList).forEach(([key, values]) => {
        if (values.length) {
          result[[scoreKey, key].join(":")] =
            values.reduce(
              (sum, v) =>
                sum + (typeof v === "object" ? Number(v.score) : Number(v)),
              0
            ) / values.length;
        }
      });

      return result;
    } else if (listAggregator === "all") {
      if (Object.values(indexScores).every((v) => typeof v === "number")) {
        return {
          [[scoreKey, aggregator].join(":")]: Object.values(indexScores).some(
            (v) => v !== 1
          )
            ? 0
            : 1,
        };
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const scoresAggregatedAcrossList: Record<string, any[]> = {};
      Object.values(indexScores).forEach((group) => {
        Object.entries(group).forEach(([key, value]) => {
          if (!scoresAggregatedAcrossList[key]) {
            scoresAggregatedAcrossList[key] = [];
          }
          scoresAggregatedAcrossList[key].push(value);
        });
      });

      const result: Record<string, number> = {};
      Object.entries(scoresAggregatedAcrossList).forEach(([key, values]) => {
        result[[scoreKey, key].join(":")] = values.some((v) => v !== 1) ? 0 : 1;
      });

      return result;
    }
  }

  if (aggregator === "average") {
    const score = Object.keys(scores).length
      ? Object.values(scores).reduce(
          (sum, v) =>
            sum + (typeof v === "object" ? Number(v.score) : Number(v)),
          0
        ) / Object.keys(scores).length
      : 0;
    return { [[scoreKey, aggregator].join(":")]: score };
  } else if (aggregator === "all") {
    const score = Object.values(scores).some(
      (v) => (typeof v === "object" ? Number(v.score) : Number(v)) !== 1
    )
      ? 0
      : 1;
    return { [[scoreKey, aggregator].join(":")]: score };
  } else {
    const results: RecordStringAny = {};
    Object.entries(scores).forEach(([key, value]) => {
      if (typeof value === "object") {
        results[[scoreKey, key].join(":")] = {
          score: Number(value.score),
          reasoning: value.reasoning,
        };
      } else {
        results[[scoreKey, key].join(":")] = Number(value);
      }
    });
    return results;
  }
}

/**
 * Create an evaluator to evaluate the accuracy of structured outputs.
 *
 * @param options The configuration options
 * @param options.aggregator - The aggregation method to use for combining the keys of each structured object.
 *        If undefined, will return a single EvaluatorResult for each key that appears in either
 *        the outputs or the reference_outputs or both. If "average", will return a single EvaluatorResult that
 *        is the average of the feedback for each key. If "all", will return a single EvaluatorResult that
 *        is a combined AND statement of the feedback for each key. If "all"/"average" the feedback key
 *        returned will be called "json_match"
 * @param options.listAggregator - The aggregation method to use when evaluating a list of outputs.
 *        Defaults to "all". If "all", the score for a single feedback key will be a combined AND statement
 *        of the scores for that key across all elements of the list. If "average", the score will be the
 *        average of the scores for that key across all elements of the list
 * @param options.rubric - The rubric to use for the judge. Each entry is a key/value pair where the key
 *        is the structured output key and the value is the criteria for the LLM to evaluate that key
 *        against the reference output
 * @param options.excludeKeys - The keys to exclude from the evaluation. Use this if there are keys in your
 *        structured output you don't care about evaluating. Every key not in excludeKeys or rubric will be
 *        evaluated for exact match with the reference output
 * @param options.judge - The judge to use for the evaluation
 * @param options.model - The model to use for the evaluation
 * @param options.useReasoning - Whether to use reasoning for the keys in rubric. Defaults to true
 * @param options.listMatchMode - The mode to use for matching list elements. Defaults to "same_elements".
 *        If "same_elements", matches every element of outputs with reference_outputs and vice versa.
 *        If "subset", matches elements of outputs with reference_outputs.
 *        If "superset", matches elements of reference_outputs with outputs.
 *        If "ordered", matches elements by their index position
 * @returns A function that takes outputs and reference_outputs and returns an EvaluatorResult or list of EvaluatorResults
 */
export const createJsonMatchEvaluator = ({
  aggregator,
  listAggregator = "all",
  rubric = {},
  excludeKeys = [],
  judge,
  model,
  useReasoning = true,
  listMatchMode = "same_elements",
}: {
  aggregator?: AggregatorType;
  listAggregator?: "average" | "all";
  rubric?: Record<string, string>;
  excludeKeys?: string[];
  judge?: ModelClient | BaseChatModel;
  model?: string;
  useReasoning?: boolean;
  listMatchMode?: ListMatchMode;
}) => {
  if ((judge || model) && Object.keys(rubric).length === 0) {
    throw new Error("rubric must be provided if judge or model is provided");
  } else if (!judge && !model && Object.keys(rubric).length !== 0) {
    throw new Error("judge or model must be provided if rubric is provided");
  }

  const wrappedEvaluator = async ({
    outputs,
    referenceOutputs,
  }: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    outputs?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    referenceOutputs?: any;
  }) => {
    const {
      processedOutputs,
      processedReferenceOutputs,
      jsonSchema,
      scores,
      formattedRubric,
      useListReducer,
    } = _prepareParameters({
      outputs,
      referenceOutputs,
      rubric,
      excludeKeys,
      useReasoning,
      listMatchMode,
    });

    // Identify which keys need LLM evaluation
    const llmKeys =
      Object.keys(formattedRubric ?? {}).length > 0
        ? new Set(Object.keys(jsonSchema.properties))
        : new Set<string>();

    // Special handling when aggregator is specified - aggregate all keys at once
    if (aggregator !== undefined) {
      async function aggregateScorer(): Promise<MultiResultScorerReturnType> {
        // Get LLM scores if needed
        if (llmKeys.size > 0) {
          const outputKeys = Object.keys(jsonSchema.properties)
            .map((key) => `${key}: ${processedOutputs[key]}`)
            .join("\n");
          const expectedOutputKeys = Object.keys(jsonSchema.properties)
            .map((key) => `${key}: ${processedReferenceOutputs[key]}`)
            .join("\n");

          const scorerFn = _createLLMAsJudgeScorer({
            prompt: USER_PROMPT,
            system: SYSTEM_PROMPT,
            schema: jsonSchema,
            judge,
            model,
          });

          const llmScores = await scorerFn({
            outputs: outputKeys,
            referenceOutputs: expectedOutputKeys,
            rubric: formattedRubric,
          });
          Object.assign(scores, llmScores);
        }

        // Aggregate
        return _aggregateResults({
          scoreKey: "json_match",
          scores,
          useListReducer,
          aggregator,
          listAggregator,
        });
      }

      return _runEvaluator(
        "json_match_evaluator",
        aggregateScorer,
        "json_match",
        {
          inputs: outputs,
          referenceOutputs,
        }
      );
    }

    // Group raw keys by their base key for processing
    const rawKeysByBase: Record<string, string[]> = {};
    const allRawKeys = new Set([...Object.keys(scores), ...llmKeys]);

    for (const rawKey of allRawKeys) {
      const baseKey =
        useListReducer && rawKey.includes("_") && rawKey.lastIndexOf("_") > 0
          ? rawKey.substring(0, rawKey.lastIndexOf("_"))
          : rawKey;
      if (!rawKeysByBase[baseKey]) {
        rawKeysByBase[baseKey] = [];
      }
      rawKeysByBase[baseKey].push(rawKey);
    }

    // Process each base key
    const allResults = [];
    for (const baseKey of Object.keys(rawKeysByBase).sort()) {
      const rawKeys = rawKeysByBase[baseKey];
      const needsLlm = rawKeys.some((rk) => llmKeys.has(rk));

      if (needsLlm) {
        // Create scorer that calls LLM for these keys
        async function keyScorer(): Promise<MultiResultScorerReturnType> {
          // Create schema for just these keys
          const keySchema = {
            type: "object",
            title: "structured_match_score",
            description: "Scores measuring the accuracy of structured outputs",
            properties: Object.fromEntries(
              rawKeys
                .filter((rk) => jsonSchema.properties[rk])
                .map((rk) => [rk, jsonSchema.properties[rk]])
            ),
            required: rawKeys.filter((rk) => jsonSchema.properties[rk]),
            additionalProperties: false,
          };

          // Create LLM scorer
          const scorerFn = _createLLMAsJudgeScorer({
            prompt: USER_PROMPT,
            system: SYSTEM_PROMPT,
            schema: keySchema,
            judge,
            model,
          });

          // Format outputs
          const outputStrs = rawKeys
            .filter((rk) => processedOutputs[rk] !== undefined)
            .map((rk) => `${rk}: ${processedOutputs[rk]}`);
          const expectedStrs = rawKeys
            .filter((rk) => processedReferenceOutputs[rk] !== undefined)
            .map((rk) => `${rk}: ${processedReferenceOutputs[rk]}`);

          const keyCriteria = rubric[baseKey] || "";
          const formattedKeyRubric = keyCriteria
            ? `Key: ${baseKey}, Criteria: ${keyCriteria}\n`
            : "";

          // Call LLM
          const llmScores = await scorerFn({
            outputs: outputStrs.join("\n"),
            referenceOutputs: expectedStrs.join("\n"),
            rubric: formattedKeyRubric,
          });

          // Combine with non-LLM scores
          const allKeyScores: RecordStringAny = {
            ...Object.fromEntries(
              rawKeys
                .filter((rk) => !llmKeys.has(rk))
                .map((rk) => [rk, scores[rk] ?? 0])
            ),
            ...(typeof llmScores === "object" && llmScores != null
              ? llmScores
              : {}),
          };

          // Aggregate across list items if needed
          if (useListReducer && rawKeys.length > 1) {
            // Fill in missing indices with 0 scores
            const allIndices = new Set<number>();
            for (const key of Object.keys(scores)) {
              if (key.includes("_")) {
                const idx = key.substring(key.lastIndexOf("_") + 1);
                try {
                  allIndices.add(parseInt(idx));
                } catch (e) {
                  // ignore non-numeric indices
                }
              }
            }

            // Add 0 scores for missing indices
            for (const idx of allIndices) {
              const expectedKey = `${baseKey}_${idx}`;
              if (!(expectedKey in allKeyScores)) {
                allKeyScores[expectedKey] = 0;
              }
            }

            const aggregated = _aggregateResults({
              scoreKey: "json_match",
              scores: allKeyScores,
              useListReducer: true,
              aggregator: undefined,
              listAggregator,
            });

            // Convert boolean scores to numbers and preserve reasoning
            const result: RecordStringAny = {};
            for (const [key, value] of Object.entries(aggregated)) {
              if (
                typeof value === "object" &&
                value != null &&
                "score" in value
              ) {
                result[key] = {
                  score: Number(value.score),
                  reasoning: value.reasoning,
                };
              } else {
                result[key] =
                  typeof value === "boolean" ? Number(value) : value;
              }
            }
            return result;
          } else {
            // Single key - return with json_match prefix
            const value = Object.values(allKeyScores)[0];
            const sourceRunId = getCurrentRunTree(true)?.id;
            if (
              typeof value === "object" &&
              value != null &&
              "score" in value
            ) {
              return {
                [`json_match:${baseKey}`]: {
                  score: Number(value.score),
                  reasoning: value.reasoning,
                  sourceRunId,
                },
              };
            } else {
              return {
                [`json_match:${baseKey}`]: {
                  score: typeof value === "boolean" ? Number(value) : value,
                  sourceRunId,
                },
              };
            }
          }
        }

        const results = await _runEvaluator(
          "json_match_evaluator",
          keyScorer,
          `json_match:${baseKey}`,
          {
            inputs: processedOutputs[baseKey] ?? outputs,
            referenceOutputs:
              processedReferenceOutputs[baseKey] ?? referenceOutputs,
          }
        );
        if (Array.isArray(results)) {
          allResults.push(...results);
        } else {
          allResults.push(results);
        }
      } else {
        // Non-LLM keys - just aggregate existing scores
        async function keyScorer(): Promise<MultiResultScorerReturnType> {
          const keyScores: RecordStringAny = Object.fromEntries(
            rawKeys.map((rk) => [rk, scores[rk]])
          );

          // Aggregate across list items if needed
          if (useListReducer && rawKeys.length > 1) {
            // Fill in missing indices with 0 scores
            const allIndices = new Set<number>();
            for (const key of Object.keys(scores)) {
              if (key.includes("_")) {
                const idx = key.substring(key.lastIndexOf("_") + 1);
                try {
                  allIndices.add(parseInt(idx));
                } catch (e) {
                  // ignore non-numeric indices
                }
              }
            }

            // Add 0 scores for missing indices
            for (const idx of allIndices) {
              const expectedKey = `${baseKey}_${idx}`;
              if (!(expectedKey in keyScores)) {
                keyScores[expectedKey] = 0;
              }
            }

            const aggregated = _aggregateResults({
              scoreKey: "json_match",
              scores: keyScores,
              useListReducer: true,
              aggregator: undefined,
              listAggregator,
            });

            // Convert boolean scores to numbers
            const result: RecordStringAny = {};
            for (const [key, value] of Object.entries(aggregated)) {
              result[key] = typeof value === "boolean" ? Number(value) : value;
            }
            return result;
          } else {
            // Single key - return with json_match prefix
            const value = keyScores[rawKeys[0]];
            return {
              [`json_match:${baseKey}`]: {
                score: typeof value === "boolean" ? Number(value) : value,
                sourceRunId: getCurrentRunTree(true)?.id,
              },
            };
          }
        }

        const results = await _runEvaluator(
          "json_match_evaluator",
          keyScorer,
          `json_match:${baseKey}`,
          {
            inputs: processedOutputs[baseKey] ?? outputs,
            referenceOutputs:
              processedReferenceOutputs[baseKey] ?? referenceOutputs,
          }
        );
        if (Array.isArray(results)) {
          allResults.push(...results);
        } else {
          allResults.push(results);
        }
      }
    }

    return allResults;
  };

  return wrappedEvaluator;
};
