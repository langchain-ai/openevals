import type { Boxd } from "@boxd-sh/sdk";
import { describe, expect, test } from "vitest";

import { createBoxdExecutionEvaluator } from "../execution.js";
import { createBoxdTypeScriptEvaluator } from "../typescript.js";
import {
  EXTRACT_IMPORT_NAMES,
  TYPESCRIPT_EVALUATOR_FILE,
  TYPESCRIPT_EVALUATOR_SEPARATOR,
} from "../../e2b/sandbox/files.js";
import {
  PACKAGE_JSON_FILE,
  PACKAGE_JSON_FILE_WITH_TSX,
} from "../sandbox/files.js";

type Call =
  | { kind: "upload"; id: string; path: string; content: string }
  | {
      kind: "exec";
      id: string;
      params: {
        command: string;
        cwd?: string;
        env?: Record<string, string>;
        timeout?: number;
      };
    };

function fakeBoxd(result: {
  stdout?: string;
  stderr?: string;
  exitCode: number;
}) {
  const calls: Call[] = [];
  const client = {
    machines: {
      files: {
        upload: async (id: string, path: string, content: string) => {
          calls.push({ kind: "upload", id, path, content });
          return content.length;
        },
      },
      exec: async (
        id: string,
        params: Extract<Call, { kind: "exec" }>["params"]
      ) => {
        calls.push({ kind: "exec", id, params });
        return {
          stdout: result.stdout ?? "",
          stderr: result.stderr ?? "",
          exitCode: result.exitCode,
          success: result.exitCode === 0,
        };
      },
    },
  };
  const uploads = () =>
    Object.fromEntries(
      calls.flatMap((c) => (c.kind === "upload" ? [[c.path, c.content]] : []))
    );
  const execs = () => calls.flatMap((c) => (c.kind === "exec" ? [c] : []));
  return { boxdClient: client as unknown as Boxd, calls, uploads, execs };
}

describe("createBoxdExecutionEvaluator", () => {
  test("uploads the code and helpers, then runs it in the project directory", async () => {
    const fake = fakeBoxd({ exitCode: 0 });
    const evaluator = createBoxdExecutionEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
      environmentVariables: { K: "V" },
    });
    const result = await evaluator({ outputs: "console.log(1)" });
    expect(result.score).toBe(true);
    expect(result.key).toBe("execution_succeeded");
    expect(fake.calls[0]).toEqual({
      kind: "upload",
      id: "m-1",
      path: "openevals/outputs.ts",
      content: "console.log(1)",
    });
    expect(fake.uploads()).toEqual({
      "openevals/outputs.ts": "console.log(1)",
      "openevals/openevals_package.json": PACKAGE_JSON_FILE_WITH_TSX,
      "openevals/extract_import_names.mjs": EXTRACT_IMPORT_NAMES,
    });
    const [run] = fake.execs();
    expect(run.id).toBe("m-1");
    expect(run.params.cwd).toBe("openevals");
    expect(run.params.env).toEqual({ K: "V" });
    expect(run.params.timeout).toBe(300_000);
    expect(run.params.command).toContain("npx tsx outputs.ts");
  });

  test("keeps an existing package.json and installs from it", async () => {
    const fake = fakeBoxd({ exitCode: 0 });
    await createBoxdExecutionEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "1" });
    await createBoxdTypeScriptEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "1" });
    expect(Object.keys(fake.uploads())).not.toContain("openevals/package.json");
    for (const run of fake.execs()) {
      expect(
        run.params.command.startsWith(
          "[ -f package.json ] || cp openevals_package.json package.json && npm install "
        )
      ).toBe(true);
      expect(run.params.command).not.toContain("> package.json");
      expect(run.params.command).toContain(
        "ls node_modules/typescript >/dev/null 2>&1 || npm install typescript@^5.9.3 "
      );
      expect(run.params.command).toContain("node extract_import_names.mjs");
    }
    const [execution, typescript] = fake.execs();
    expect(execution.params.command).toContain(
      "ls node_modules/tsx >/dev/null 2>&1 || npm install tsx@"
    );
    expect(typescript.params.command).not.toContain("tsx@");
  });

  test("pins a TypeScript version that has the compiler API", () => {
    for (const file of [PACKAGE_JSON_FILE, PACKAGE_JSON_FILE_WITH_TSX]) {
      const { dependencies } = JSON.parse(file);
      expect(dependencies.typescript).toBe("^5.9.3");
    }
    expect(
      JSON.parse(PACKAGE_JSON_FILE_WITH_TSX).dependencies.tsx
    ).toBeDefined();
  });

  test("never echoes helper scripts into the shell", async () => {
    const fake = fakeBoxd({ exitCode: 0 });
    await createBoxdExecutionEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "1" });
    const [run] = fake.execs();
    expect(run.params.command).not.toContain("> extract_import_names.mjs");
    expect(run.params.command).not.toContain(EXTRACT_IMPORT_NAMES);
  });

  test("a failing run returns stderr", async () => {
    const fake = fakeBoxd({ exitCode: 1, stderr: "TypeError: boom" });
    const result = await createBoxdExecutionEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "throw new TypeError('boom')" });
    expect(result.score).toBe(false);
    expect(result.comment).toBe("TypeError: boom");
  });

  test("a failing run with empty stderr returns stdout", async () => {
    const fake = fakeBoxd({ exitCode: 2, stdout: "only stdout" });
    const result = await createBoxdExecutionEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "x" });
    expect(result.score).toBe(false);
    expect(result.comment).toBe("only stdout");
  });

  test("options reach the machine", async () => {
    const fake = fakeBoxd({ exitCode: 0 });
    await createBoxdExecutionEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-2",
      sandboxProjectDirectory: "/home/boxd/project",
      timeoutMs: 1234,
    })({ outputs: "1" });
    expect(Object.keys(fake.uploads())).toContain(
      "/home/boxd/project/outputs.ts"
    );
    const [run] = fake.execs();
    expect(run.id).toBe("m-2");
    expect(run.params.cwd).toBe("/home/boxd/project");
    expect(run.params.timeout).toBe(1234);
  });

  test("markdown extraction without a code block does not touch the machine", async () => {
    const fake = fakeBoxd({ exitCode: 0 });
    const result = await createBoxdExecutionEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
      codeExtractionStrategy: "markdown_code_blocks",
    })({ outputs: "no code here" });
    expect(result.score).toBe(false);
    expect(fake.calls).toEqual([]);
  });

  test("model requires llm extraction", () => {
    const fake = fakeBoxd({ exitCode: 0 });
    expect(() =>
      createBoxdExecutionEvaluator({
        boxdClient: fake.boxdClient,
        machineId: "m-1",
        model: "openai:gpt-4o",
      })
    ).toThrow();
    expect(() =>
      createBoxdTypeScriptEvaluator({
        boxdClient: fake.boxdClient,
        machineId: "m-1",
        model: "openai:gpt-4o",
      })
    ).toThrow();
  });
});

describe("createBoxdTypeScriptEvaluator", () => {
  test("parses the result after the separator", async () => {
    const fake = fakeBoxd({
      exitCode: 0,
      stdout: `${TYPESCRIPT_EVALUATOR_SEPARATOR}\n${JSON.stringify([false, "(1,7): bad"])}\n`,
    });
    const result = await createBoxdTypeScriptEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "const x: number = 'a';" });
    expect(result.score).toBe(false);
    expect(result.comment).toBe("(1,7): bad");
    expect(result.key).toBe("typescript_succeeded");
    expect(fake.uploads()).toEqual({
      "openevals/outputs.ts": "const x: number = 'a';",
      "openevals/openevals_package.json": PACKAGE_JSON_FILE,
      "openevals/check_typescript.mjs": TYPESCRIPT_EVALUATOR_FILE,
      "openevals/extract_import_names.mjs": EXTRACT_IMPORT_NAMES,
    });
    const [run] = fake.execs();
    expect(run.params.cwd).toBe("openevals");
    expect(run.params.timeout).toBe(300_000);
    expect(run.params.env).toBeUndefined();
    expect(run.params.command).toContain("node check_typescript.mjs");
  });

  test("passes clean code", async () => {
    const fake = fakeBoxd({
      exitCode: 0,
      stdout: `${TYPESCRIPT_EVALUATOR_SEPARATOR}\ntrue\n`,
    });
    const result = await createBoxdTypeScriptEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "const x = 1;" });
    expect(result.score).toBe(true);
  });

  test("a failed install returns stderr", async () => {
    const fake = fakeBoxd({ exitCode: 1, stderr: "npm error 404" });
    const result = await createBoxdTypeScriptEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "import x from 'nope';" });
    expect(result.score).toBe(false);
    expect(result.comment).toBe("npm error 404");
  });

  test("output without the separator fails with stdout", async () => {
    const fake = fakeBoxd({ exitCode: 0, stdout: "unexpected" });
    const result = await createBoxdTypeScriptEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "1" });
    expect(result.score).toBe(false);
    expect(result.comment).toBe("unexpected");
  });
});

describe("shared machine behaviour", () => {
  test("import names reach npm as one quoted argument", async () => {
    const fake = fakeBoxd({ exitCode: 0 });
    await createBoxdExecutionEvaluator({
      boxdClient: fake.boxdClient,
      machineId: "m-1",
    })({ outputs: "1" });
    const [run] = fake.execs();
    expect(run.params.command).toContain(
      `xargs -r -n1 sh -c 'ls "node_modules/$1" >/dev/null 2>&1 || npm install "$1"`
    );
    expect(run.params.command).not.toContain("{}");
  });

  test("evaluations on one machine and directory run one at a time", async () => {
    let active = 0;
    let maxActive = 0;
    let mismatches = 0;
    const current: Record<string, string> = {};
    const client = {
      machines: {
        files: {
          upload: async (_id: string, path: string, content: string) => {
            if (path.endsWith("/outputs.ts")) {
              active += 1;
              maxActive = Math.max(maxActive, active);
              current[path] = content;
            }
            return content.length;
          },
        },
        exec: async (_id: string, params: { cwd: string }) => {
          const seen = current[`${params.cwd}/outputs.ts`];
          await new Promise((resolve) => setTimeout(resolve, 20));
          if (current[`${params.cwd}/outputs.ts`] !== seen) mismatches += 1;
          active -= 1;
          return { stdout: "", stderr: "", exitCode: 0, success: true };
        },
      },
    } as unknown as Boxd;
    const execution = createBoxdExecutionEvaluator({
      boxdClient: client,
      machineId: "m-1",
    });
    const typescript = createBoxdTypeScriptEvaluator({
      boxdClient: client,
      machineId: "m-1",
    });
    await Promise.all(
      [execution, typescript, execution, typescript].map((ev, i) =>
        ev({ outputs: `console.log(${i})` })
      )
    );
    expect(maxActive).toBe(1);
    expect(mismatches).toBe(0);
  });

  test("a failed evaluation does not block the next one", async () => {
    let calls = 0;
    const client = {
      machines: {
        files: { upload: async () => 1 },
        exec: async () => {
          calls += 1;
          if (calls === 1) throw new Error("exec timed out");
          return { stdout: "", stderr: "", exitCode: 0, success: true };
        },
      },
    } as unknown as Boxd;
    const evaluator = createBoxdExecutionEvaluator({
      boxdClient: client,
      machineId: "m-9",
    });
    await expect(evaluator({ outputs: "1" })).rejects.toThrow("exec timed out");
    expect((await evaluator({ outputs: "1" })).score).toBe(true);
  });

  test("timeoutMs must be positive", () => {
    const fake = fakeBoxd({ exitCode: 0 });
    for (const timeoutMs of [0, -1]) {
      expect(() =>
        createBoxdExecutionEvaluator({
          boxdClient: fake.boxdClient,
          machineId: "m-1",
          timeoutMs,
        })
      ).toThrow("timeoutMs");
      expect(() =>
        createBoxdTypeScriptEvaluator({
          boxdClient: fake.boxdClient,
          machineId: "m-1",
          timeoutMs,
        })
      ).toThrow("timeoutMs");
    }
  });
});
