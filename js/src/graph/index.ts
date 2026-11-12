/** Evaluators built as a graph of nodes. Experimental. */

const notImplemented = (name: string): never => {
  throw new Error(`openevals/graph: ${name} is not implemented yet`);
};

export type ContextName = "run" | "thread" | "trajectory" | "example";

export interface Ref {
  root: ContextName | GraphNode;
  path: string[];
}

export interface GraphNode {
  readonly id: string;
}

export interface CodeNodeOptions {
  name?: string;
}

export function node<Args extends unknown[], Out>(
  _fn: (...args: Args) => Out | Promise<Out>,
  _options?: CodeNodeOptions
): GraphNode {
  return notImplemented("node");
}

export interface LLMOptions {
  model: string;
  prompt: string | Array<[string, string]>;
  inputs?: Record<string, Ref>;
  outputSchema?: Record<string, unknown>;
  feedbackKey?: string;
}

export function LLM(_options: LLMOptions): GraphNode {
  return notImplemented("LLM");
}

export type DecisionQuestion =
  | { type: "noul"; instructions: string }
  | { type: "score"; instructions: string; levels: string[] }
  | { type: "choice"; instructions: string; options: Record<string, string | null> };

export interface DecisionModelOptions {
  questions: Record<string, DecisionQuestion>;
  inputs?: Record<string, Ref>;
  model?: string;
}

export function DecisionModel(_options: DecisionModelOptions): GraphNode {
  return notImplemented("DecisionModel");
}

export class Graph {
  constructor(_nodes: GraphNode[]) {
    notImplemented("Graph");
  }

  async run(_ctx: unknown): Promise<Map<GraphNode, unknown>> {
    return notImplemented("Graph.run");
  }
}
