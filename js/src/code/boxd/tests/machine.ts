import { Boxd } from "@boxd-sh/sdk";

/**
 * The machine the boxd tests run in. Set BOXD_MACHINE_ID to use a machine you
 * already have; it is left as it is. Otherwise an isolated machine is created
 * and `release` deletes it. In case the run is killed before that, the machine
 * also deletes itself after BOXD_TEST_DELETE_AFTER seconds (default 3600, 0 for
 * no timer). BOXD_ORG and BOXD_REGION pick the org and region, and
 * BOXD_TEST_MACHINE_PREFIX the start of the name.
 */
export async function acquireTestMachine() {
  const boxdClient = new Boxd();
  const existing = process.env.BOXD_MACHINE_ID;
  if (existing) {
    // An exec wakes a machine that is in standby or hibernated; a status poll
    // alone does not.
    await boxdClient.machines.exec(existing, { command: "true" });
    await boxdClient.machines.waitUntilReady(existing);
    return { boxdClient, machineId: existing, release: async () => {} };
  }
  const prefix = process.env.BOXD_TEST_MACHINE_PREFIX ?? "openevals-test-";
  const machine = await boxdClient.machines.create({
    name: `${prefix}${Math.random().toString(16).slice(2, 10)}`,
    org: process.env.BOXD_ORG,
    isolated: true,
    labels: { purpose: "openevals-test" },
    deleteAfter:
      Number(process.env.BOXD_TEST_DELETE_AFTER ?? "3600") || undefined,
  });
  const release = async () => {
    await boxdClient.machines.delete(machine.id);
  };
  try {
    await boxdClient.machines.waitUntilReady(machine.id, { timeout: 180_000 });
  } catch (e) {
    await release();
    throw e;
  }
  return { boxdClient, machineId: machine.id, release };
}
