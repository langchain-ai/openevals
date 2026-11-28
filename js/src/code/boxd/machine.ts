import type { Boxd } from "@boxd-sh/sdk";

export const DEFAULT_TIMEOUT_MS = 300_000;

// One evaluation at a time per machine and project directory, because each one
// writes outputs.ts and runs it in that directory.
const queues = new Map<string, Promise<unknown>>();

export function _validateTimeoutMs(timeoutMs: number | undefined) {
  if (timeoutMs !== undefined && !(timeoutMs > 0)) {
    throw new Error("timeoutMs must be a positive number of milliseconds");
  }
}

/**
 * Writes the given files into `cwd` on the machine, then runs `command` there.
 */
export async function _runInMachine(params: {
  boxdClient: Boxd;
  machineId: string;
  cwd: string;
  files: Record<string, string>;
  command: string;
  env?: Record<string, string>;
  timeoutMs: number;
}) {
  const { boxdClient, machineId, cwd } = params;
  const key = `${machineId}\0${cwd}`;
  const run = async () => {
    for (const [name, content] of Object.entries(params.files)) {
      await boxdClient.machines.files.upload(
        machineId,
        `${cwd}/${name}`,
        content
      );
    }
    return boxdClient.machines.exec(machineId, {
      command: params.command,
      cwd,
      env: params.env,
      timeout: params.timeoutMs,
    });
  };
  const previous = queues.get(key) ?? Promise.resolve();
  const result = previous.then(run, run);
  const settled = result.catch(() => undefined);
  queues.set(key, settled);
  void settled.then(() => {
    if (queues.get(key) === settled) {
      queues.delete(key);
    }
  });
  return result;
}
