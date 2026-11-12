/**
 * Composable evaluators: an evaluator built as a graph of steps.
 *
 * Experimental: the API may change in any release.
 *
 * Each step is a node: a function wrapped with `node`, an `LLM` judge, or a
 * `DecisionModel`. A node reads the run being evaluated or the outputs of other
 * nodes, and `Graph` runs the nodes in dependency order. Names and options
 * mirror `openevals.graph` in Python.
 */

const notImplemented = (name: string): never => {
  throw new Error(`openevals/graph: ${name} is not implemented yet`);
};

export type ContextName = "run" | "thread" | "trajectory" | "example";

/** A path into the evaluated context or into another node's outputs. */
export interface Ref {
  root: ContextName | GraphNode;
  path: string[];
}

/** A step in a graph. */
export interface GraphNode {
  readonly id: string;
}

export interface CodeNodeOptions {
  name?: string;
}

/** Wraps a function as a code node. */
export function node<Args extends unknown[], Out>(
  _fn: (...args: Args) => Out | Promise<Out>,
  _options?: CodeNodeOptions
): GraphNode {
  return notImplemented("node");
}

export interface LLMOptions {
  model: string;
  /** A prompt or chat messages; `{{name}}` reads the input called `name`. */
  prompt: string | Array<[string, string]>;
  inputs?: Record<string, Ref>;
  outputSchema?: Record<string, unknown>;
  feedbackKey?: string;
}

/** An LLM-as-judge node. */
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

/** A node that asks a decision model the given questions about its inputs. */
export function DecisionModel(_options: DecisionModelOptions): GraphNode {
  return notImplemented("DecisionModel");
}

/** Runs nodes in dependency order, whatever order they are listed in. */
export class Graph {
  constructor(_nodes: GraphNode[]) {
    notImplemented("Graph");
  }

  async run(_ctx: unknown): Promise<Map<GraphNode, unknown>> {
    return notImplemented("Graph.run");
  }
}
