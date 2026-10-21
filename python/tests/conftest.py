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
