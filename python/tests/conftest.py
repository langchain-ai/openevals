import asyncio
import gc
import os
import sys
import threading
import types

import pytest
from langsmith.evaluation._arunner import AsyncExperimentResults

# Workaround for a lost-wakeup bug in langsmith's AsyncExperimentResults.
#
# `client.aevaluate()` returns an AsyncExperimentResults whose background task
# (`_process_data`) fills a results list. Each `__anext__` call waits on an
# asyncio.Condition until the task adds a result or finishes. The task only
# notifies waiters on success. If it raises (for example, after a transient
# LangSmith API error), nothing wakes the waiter, so `async for r in res` waits
# forever on an idle event loop. The exception is never raised in the test.
#
# In CI this showed up as the Python job stalling at ~24% on
# test_embedding_similarity_aevaluate until pytest-timeout ended the run. The
# stack dump only showed the idle loop, so the real error was lost.
#
# Always notifying waiters when the task finishes lets `__anext__` see that the
# task is done and raise its exception, so the test fails right away with the
# actual error.
_original_process_data = AsyncExperimentResults._process_data


async def _process_data_always_notify(self, manager):
    try:
        await _original_process_data(self, manager)
    finally:
        async with self._condition:
            self._condition.notify_all()


AsyncExperimentResults._process_data = _process_data_always_notify


# --- Diagnostics for hung async tests -------------------------------------
#
# pytest-timeout's dump only shows threads. For an async test that hangs, the
# main thread is just an idle event loop in select(), which says nothing about
# *what* the test is waiting on. If a test is still running after
# _ASYNC_DUMP_AFTER_S seconds, print every pending asyncio task on the running
# loop with its await chain, plus every suspended async generator (await chains
# cannot be followed through `async for`, which langsmith uses heavily).
# Tests normally finish well under a minute; pytest-timeout ends the run at 300s.
_ASYNC_DUMP_AFTER_S = float(os.environ.get("OPENEVALS_ASYNC_DUMP_AFTER_S", "120"))


def _frame_line(frame) -> str:
    code = frame.f_code
    return f"{code.co_filename}:{frame.f_lineno} in {code.co_qualname}"


def _await_chain(obj) -> list[str]:
    lines = []
    for _ in range(100):
        if obj is None:
            break
        frame = (
            getattr(obj, "cr_frame", None)
            or getattr(obj, "ag_frame", None)
            or getattr(obj, "gi_frame", None)
        )
        if frame is not None:
            lines.append("    " + _frame_line(frame))
        else:
            lines.append(f"    <{type(obj).__name__}> {obj!r}"[:500])
        obj = (
            getattr(obj, "cr_await", None)
            or getattr(obj, "ag_await", None)
            or getattr(obj, "gi_yieldfrom", None)
        )
    return lines


def _dump_loop(loop, nodeid: str) -> None:
    out = sys.__stderr__
    tasks = asyncio.all_tasks(loop)
    print(
        f"\n===== {nodeid} still running after {_ASYNC_DUMP_AFTER_S}s: "
        f"{len(tasks)} pending asyncio task(s) =====",
        file=out,
    )
    for task in tasks:
        print(f"--- {task!r}"[:1000], file=out)
        print("\n".join(_await_chain(task.get_coro())), file=out)
    gens = [
        o
        for o in gc.get_objects()
        if isinstance(o, types.AsyncGeneratorType) and o.ag_frame is not None
    ]
    print(f"===== {len(gens)} suspended async generator(s) =====", file=out)
    for gen in gens:
        print(f"--- {gen!r}", file=out)
        print("\n".join(_await_chain(gen)), file=out)
    print("===== end asyncio dump =====", file=out, flush=True)


def _dump_running_loops(nodeid: str) -> None:
    for obj in gc.get_objects():
        if isinstance(obj, asyncio.AbstractEventLoop) and obj.is_running():
            # The loop thread is idle in select(); this wakes it to run the dump
            # on the loop itself, where all_tasks() and the coroutines are safe
            # to inspect.
            obj.call_soon_threadsafe(_dump_loop, obj, nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    timer = threading.Timer(_ASYNC_DUMP_AFTER_S, _dump_running_loops, (item.nodeid,))
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
