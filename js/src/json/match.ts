import { ModelClient } from './types/model-client';
import { BaseChatModel } from './types/base-chat-model';
import { _createLLMAsJudgeScorer } from '../llm.js';
import { _runEvaluator } from '../utils.js';

type AggregatorType = 'average' | 'all' | undefined;
type ListMatchMode = 'superset' | 'subset' | 'same_elements' | 'ordered';

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
    listMatchMode = 'same_elements'
}: {
    outputs: Record<string, any>[];
    referenceOutputs: Record<string, any>[];
    rubric: Record<string, string>;
    excludeKeys: string[];
    useReasoning: boolean;
    listMatchMode: ListMatchMode;
}): [any, any, any, Record<string, any>, string, boolean] {
    const jsonSchema: Record<string, any> = {
        type: 'object',
        title: 'structured_match_score',
        description: 'Scores measuring the accuracy of structured outputs',
        properties: {},
        required: [],
        additionalProperties: false,
    };

    let scores: Record<string, any> = {};
    let formattedRubric = '';
    let useListReducer = false;
    let processedOutputs: Record<string, any> = {};
    let processedReferenceOutputs: Record<string, any> = {};

    if (Array.isArray(outputs)) {
        useListReducer = true;
        if (!Array.isArray(referenceOutputs)) {
            throw new Error('If outputs is a list, referenceOutputs must also be a list');
        }

        let outputsToUse: Record<string, any> = {};
        let referenceOutputsToUse: Record<string, any> = {};

        if (listMatchMode === 'ordered') {
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
        } else if (listMatchMode === 'superset') {
            const availableOutputs = Array.from(Array(outputs.length).keys());
            const matchedReferences = new Set<number>();

            referenceOutputs.forEach((refItem, i) => {
                let bestMatchScore = -1;
                let bestMatchIdx: number | null = null;

                availableOutputs.forEach(outIdx => {
                    const outputItem = outputs[outIdx];
                    let matchScore = 0;

                    Object.keys(refItem).forEach(key => {
                        if (key in outputItem && !excludeKeys.includes(key) && !(key in rubric)) {
                            matchScore += Number(refItem[key] === outputItem[key]);
                        }
                    });

                    if (matchScore > bestMatchScore) {
                        bestMatchScore = matchScore;
                        bestMatchIdx = outIdx;
                    }
                });

                if (bestMatchIdx !== null) {
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
            const availableReferences = Array.from(Array(referenceOutputs.length).keys());
            const matchedOutputs = new Set<number>();

            outputs.forEach((outputItem: any, i: number) => {
                let bestMatchIdx: number | null = null;
                let bestMatchScore = -1;

                availableReferences.forEach(refIdx => {
                    const refItem = referenceOutputs[refIdx];
                    let matchScore = 0;

                    Object.keys(outputItem).forEach(key => {
                        if (key in refItem && !excludeKeys.includes(key) && !(key in rubric)) {
                            matchScore += Number(outputItem[key] === refItem[key]);
                        }
                    });

                    if (matchScore > bestMatchScore) {
                        bestMatchScore = matchScore;
                        bestMatchIdx = refIdx;
                    }
                });

                if (bestMatchIdx !== null) {
                    Object.entries(outputItem).forEach(([key, value]) => {
                        outputsToUse[`${key}_${i}`] = value;
                    });

                    Object.entries(referenceOutputs[bestMatchIdx]).forEach(([key, value]) => {
                        referenceOutputsToUse[`${key}_${i}`] = value;
                    });

                    availableReferences.splice(availableReferences.indexOf(bestMatchIdx), 1);
                    matchedOutputs.add(i);
                } else {
                    Object.entries(outputItem).forEach(([key, value]) => {
                        outputsToUse[`${key}_${i}`] = value;
                    });
                }
            });

            if (listMatchMode === 'same_elements') {
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
        const key = useListReducer ? rawKey.substring(0, rawKey.lastIndexOf('_')) : rawKey;
        
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
                    type: 'boolean',
                    description: `Does the output for key ${key}, follow the criteria? ${keyCriteria}`,
                };
                jsonSchema.required.push(rawKey);
            } else {
                jsonSchema.properties[rawKey] = {
                    type: 'object',
                    properties: {
                        reasoning: {
                            type: 'string',
                            description: `Reasoning for the score you assigned to key ${key}`,
                        },
                        score: {
                            type: 'boolean',
                            description: `Does the output for key ${key}, follow the criteria? ${keyCriteria}`,
                        },
                    },
                    required: ['score', 'reasoning'],
                    additionalProperties: false,
                };
                jsonSchema.required.push(rawKey);
            }
        }
    });

    Object.entries(processedReferenceOutputs).forEach(([rawKey, _]) => {
        const key = useListReducer ? rawKey.substring(0, rawKey.lastIndexOf('_')) : rawKey;
        if (!excludeKeys.includes(key) && !(rawKey in processedOutputs)) {
            scores[rawKey] = 0;
        }
    });

    return [processedOutputs, processedReferenceOutputs, jsonSchema, scores, formattedRubric, useListReducer];
}

function _aggregateResults({
    scores,
    useListReducer,
    aggregator,
    listAggregator,
}: {
    scores: Record<string, any>;
    useListReducer: boolean;
    aggregator: AggregatorType;
    listAggregator: 'average' | 'all';
}): Record<string, any> | number | [number, string] {
    if (useListReducer) {
        const indexGroupedScores: Record<string, Record<string, any>> = {};
        
        Object.entries(scores).forEach(([k, v]) => {
            const index = k.substring(k.lastIndexOf('_') + 1);
            const baseKey = k.substring(0, k.lastIndexOf('_'));
            
            if (!indexGroupedScores[index]) {
                indexGroupedScores[index] = {};
            }
            indexGroupedScores[index][baseKey] = v;
        });

        let indexScores: Record<string, any> = {};
        
        if (aggregator === 'average') {
            Object.entries(indexGroupedScores).forEach(([index, group]) => {
                if (Object.keys(group).length) {
                    const total = Object.values(group).reduce((sum, v) => 
                        sum + (typeof v === 'object' ? Number(v.score) : Number(v)), 0);
                    indexScores[index] = total / Object.keys(group).length;
                }
            });
        } else if (aggregator === 'all') {
            Object.entries(indexGroupedScores).forEach(([index, group]) => {
                if (Object.keys(group).length) {
                    const hasNonOne = Object.values(group).some(v => 
                        (typeof v === 'object' ? Number(v.score) : Number(v)) !== 1);
                    indexScores[index] = hasNonOne ? 0 : 1;
                }
            });
        } else {
            indexScores = indexGroupedScores;
        }

        if (listAggregator === 'average') {
            if (Object.values(indexScores).every(v => typeof v === 'number')) {
                return Object.keys(indexScores).length ? 
                    Object.values(indexScores).reduce((a, b) => Number(a) + Number(b), 0) / Object.keys(indexScores).length : 
                    0;
            }

            const scoresAggregatedAcrossList: Record<string, any[]> = {};
            Object.values(indexScores).forEach(group => {
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
                    result[key] = values.reduce((sum, v) => 
                        sum + (typeof v === 'object' ? Number(v.score) : Number(v)), 0) / values.length;
                }
            });

            return result;
        } else if (listAggregator === 'all') {
            if (Object.values(indexScores).every(v => typeof v === 'number')) {
                return Object.values(indexScores).some(v => v !== 1) ? 0 : 1;
            }

            const scoresAggregatedAcrossList: Record<string, any[]> = {};
            Object.values(indexScores).forEach(group => {
                Object.entries(group).forEach(([key, value]) => {
                    if (!scoresAggregatedAcrossList[key]) {
                        scoresAggregatedAcrossList[key] = [];
                    }
                    scoresAggregatedAcrossList[key].push(value);
                });
            });

            const result: Record<string, number> = {};
            Object.entries(scoresAggregatedAcrossList).forEach(([key, values]) => {
                result[key] = values.some(v => v !== 1) ? 0 : 1;
            });

            return result;
        }
    }

    if (aggregator === 'average') {
        return Object.keys(scores).length ?
            Object.values(scores).reduce((sum, v) => 
                sum + (typeof v === 'object' ? Number(v.score) : Number(v)), 0) / Object.keys(scores).length :
            0;
    } else if (aggregator === 'all') {
        return Object.values(scores).some(v => 
            (typeof v === 'object' ? Number(v.score) : Number(v)) !== 1) ? 0 : 1;
    } else {
        const results: Record<string, any> = {};
        Object.entries(scores).forEach(([key, value]) => {
            if (typeof value === 'object') {
                results[key] = {score: Number(value.score), reasoning: value.reasoning};
            } else {
                results[key] = Number(value);
            }
        });
        
        if (Object.keys(results).length === 1) {
            const ans = Object.values(results)[0];
            if (typeof ans === 'object') {
                return [Number(ans.score), ans.reasoning];
            }
            return Number(ans);
        }
        return results;
    }
}

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
    listAggregator?: 'average' | 'all';
    rubric?: Record<string, string>;
    excludeKeys?: string[];
    judge?: ModelClient | BaseChatModel;
    model?: string;
    useReasoning?: boolean;
    listMatchMode?: ListMatchMode;
}) => {
    const wrappedEvaluator = async ({
        outputs,
        referenceOutputs,
    }: {
        outputs: any;
        referenceOutputs: any;
    }) => {
        async function scorer({
            outputs,
            referenceOutputs,
            rubric = {},
            excludeKeys = [],
            useReasoning = true,
        }: {
            outputs: any;
            referenceOutputs: any;
            rubric?: Record<string, string>;
            excludeKeys?: string[];
            useReasoning?: boolean;
        }): Promise<Record<string, any> | number | [any, any]> {
            const [
                processedOutputs,
                processedReferenceOutputs,
                jsonSchema,
                scores,
                formattedRubric,
                useListReducer
            ] = _prepareParameters({
                outputs,
                referenceOutputs,
                rubric,
                excludeKeys,
                useReasoning,
                listMatchMode,
            });

            let scorerFn: ((params: any) => any) | undefined;
            let outputKeys, expectedOutputKeys;
            if (Object.keys(formattedRubric ?? {}).length > 0) {
                outputKeys = Object.keys(jsonSchema.properties)
                    .map((key) => `${key}: ${processedOutputs[key]}`)
                    .join("\n");
                expectedOutputKeys = Object.keys(jsonSchema.properties)
                    .map((key) => `${key}: ${processedReferenceOutputs[key]}`)
                    .join("\n");

                scorerFn = _createLLMAsJudgeScorer({
                    prompt: USER_PROMPT,
                    system: SYSTEM_PROMPT,
                    schema: jsonSchema,
                    judge,
                    model,
                });
            }

            if (scorerFn) {
                const llmScores = await scorerFn({
                    outputs: outputKeys,
                    referenceOutputs: expectedOutputKeys,
                    rubric,
                });
                Object.assign(scores, llmScores);
            }

            return _aggregateResults({
                scores,
                useListReducer,
                aggregator,
                listAggregator,
            });
        }

        return _runEvaluator(
            "structured_match_evaluator",
            scorer,
            "structured_match_score",
            {
                outputs,
                referenceOutputs,
                rubric,
                excludeKeys,
                useReasoning,
            }
        );
    }

    return wrappedEvaluator;
}
