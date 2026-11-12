import asyncio
import signal
import threading
import traceback

import anthropic
import openai
import pytest
import pytest_timeout
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


# ── Request timeouts for provider clients ─────────────────────────────────────
#
# langchain builds OpenAI and Anthropic clients with `timeout=None`, which the
# SDKs treat as "no timeout at all" (not their 10-minute default). A stalled
# request, such as an embeddings call in test_embedding_similarity_aevaluate,
# then waits forever. Fill in a timeout for clients created without one so a
# stuck call raises APITimeoutError naming the request instead of hanging.
#
# 90s read covers the slowest judge calls (the slowest test is ~40s over several
# calls). With the SDKs' default 2 retries the worst case is ~3 x 90s, which
# stays under the 300s pytest-timeout limit. Clients created with an explicit
# timeout keep it.

_REQUEST_TIMEOUT_S = 90.0
_CONNECT_TIMEOUT_S = 10.0


def _default_timeout(cls, timeout) -> None:
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__


for _cls in (openai.OpenAI, openai.AsyncOpenAI):
    _default_timeout(
        _cls, openai.Timeout(_REQUEST_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S)
    )
for _cls in (anthropic.Anthropic, anthropic.AsyncAnthropic):
    _default_timeout(
        _cls, anthropic.Timeout(_REQUEST_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S)
    )

# ── pytest-timeout: fail the test, keep the run going ─────────────────────────
#
# pyproject.toml sets `timeout_method = "signal"`: SIGALRM raises in the main
# thread, so a hung test fails and later tests still run. (The "thread" method
# ends the whole run, losing every later result.) This hook adds two things:
#
# 1. Before failing, print each pending asyncio task with the chain of awaits
#    it is stuck in. For async tests the normal stack dump only shows the idle
#    event loop, which doesn't say which call hung.
# 2. A backstop: if the signal can't stop the test (a C call that ignores
#    EINTR, or the exception lands in another task), the thread method dumps
#    every stack and ends the run after a grace period, as before.


_BACKSTOP_GRACE_S = 60.0


def _await_chain(coro):
    """Yield the frames of a coroutine and everything it awaits, outermost first."""
    for _ in range(100):
        if coro is None:
            return
        frame = (
            getattr(coro, "cr_frame", None)
            or getattr(coro, "gi_frame", None)
            or getattr(coro, "ag_frame", None)
        )
        if frame is not None:
            yield frame
        coro = (
            getattr(coro, "cr_await", None)
            or getattr(coro, "gi_yieldfrom", None)
            or getattr(coro, "ag_await", None)
        )


def _dump_asyncio_tasks(terminal, nodeid: str) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # sync test: the thread stack dump below is enough
    terminal.sep("+", title=f"Timeout: pending asyncio tasks ({nodeid})")
    write = terminal.write
    for task in asyncio.all_tasks(loop):
        write(f"--- {task!r}\n")
        frames = [
            traceback.FrameSummary(f.f_code.co_filename, f.f_lineno, f.f_code.co_name)
            for f in _await_chain(task.get_coro())
        ]
        write("".join(traceback.format_list(frames)))


@pytest.hookimpl(tryfirst=True)
def pytest_timeout_set_timer(item, settings):
    if (
        settings.method != "signal"
        or threading.current_thread() is not threading.main_thread()
    ):
        return None  # use pytest-timeout's own implementation

    def handler(signum, frame):
        __tracebackhide__ = True
        _dump_asyncio_tasks(item.config.get_terminal_writer(), item.nodeid)
        # Dumps the other threads' stacks, then raises pytest's Failed.
        pytest_timeout.timeout_sigalrm(item, settings)

    backstop = threading.Timer(
        settings.timeout + _BACKSTOP_GRACE_S,
        pytest_timeout.timeout_timer,
        (item, settings),
    )
    backstop.daemon = True
    backstop.name = f"pytest-timeout backstop {item.nodeid}"

    def cancel():
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        backstop.cancel()

    item.cancel_timeout = cancel
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, settings.timeout)
    backstop.start()
    return True
