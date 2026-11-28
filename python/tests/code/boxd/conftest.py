import os
import uuid

import pytest


@pytest.fixture(scope="session")
def boxd_client():
    from boxd import Boxd

    return Boxd()


@pytest.fixture(scope="session")
def async_boxd_client():
    from boxd import AsyncBoxd

    return AsyncBoxd()


@pytest.fixture(scope="session")
def boxd_machine_id(boxd_client):
    """The machine every boxd test runs in.

    Set BOXD_MACHINE_ID to run the tests in a machine you already have; it is left
    as it is. Otherwise the fixture creates an isolated machine for the session and
    deletes it at the end. In case the run is killed before that, the machine also
    deletes itself after BOXD_TEST_DELETE_AFTER seconds (default 3600, 0 for no
    timer). BOXD_ORG and BOXD_REGION pick the org and region, and
    BOXD_TEST_MACHINE_PREFIX the start of the name.
    """
    existing = os.environ.get("BOXD_MACHINE_ID")
    if existing:
        # An exec wakes a machine that is in standby or hibernated; a status
        # poll alone does not.
        boxd_client.machines.exec(existing, "true")
        boxd_client.machines.wait_until_ready(existing)
        yield existing
        return
    prefix = os.environ.get("BOXD_TEST_MACHINE_PREFIX", "openevals-test-")
    machine = boxd_client.machines.create(
        f"{prefix}{uuid.uuid4().hex[:8]}",
        org=os.environ.get("BOXD_ORG"),
        isolated=True,
        labels={"purpose": "openevals-test"},
        delete_after=int(os.environ.get("BOXD_TEST_DELETE_AFTER", "3600")) or None,
    )
    try:
        boxd_client.machines.wait_until_ready(machine.id, timeout=180)
        yield machine.id
    finally:
        boxd_client.machines.delete(machine.id)
