import anthropic
import openai
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
