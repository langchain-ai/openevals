import asyncio
import threading
from typing import Optional, Union

from boxd import AsyncBoxd, Boxd, ExecResult

DEFAULT_TIMEOUT_SECONDS = 300.0

# One evaluation at a time per machine and project directory, because each one
# writes outputs.py and runs it in that directory.
_locks: dict[tuple[str, str], threading.Lock] = {}
_async_locks: dict[tuple[int, str, str], asyncio.Lock] = {}
_locks_guard = threading.Lock()


def _validate_timeout(timeout: Optional[float]) -> None:
    if timeout is not None and timeout <= 0:
        raise ValueError(
            "timeout must be a positive number of seconds, or None for no limit"
        )


def _text(value: Union[str, bytes]) -> str:
    return value if isinstance(value, str) else value.decode("utf-8", "replace")


def _run_in_machine(
    boxd_client: Boxd,
    machine_id: str,
    *,
    cwd: str,
    files: dict[str, str],
    command: str,
    env: Optional[dict[str, str]],
    timeout: Optional[float],
) -> ExecResult:
    """Writes the files into cwd on the machine, then runs the command there."""
    with _locks_guard:
        lock = _locks.setdefault((machine_id, cwd), threading.Lock())
    with lock:
        for name, content in files.items():
            boxd_client.machines.files.upload(machine_id, f"{cwd}/{name}", content)
        return boxd_client.machines.exec(
            machine_id, command, cwd=cwd, env=env, timeout=timeout
        )


async def _arun_in_machine(
    boxd_client: AsyncBoxd,
    machine_id: str,
    *,
    cwd: str,
    files: dict[str, str],
    command: str,
    env: Optional[dict[str, str]],
    timeout: Optional[float],
) -> ExecResult:
    """Async version of `_run_in_machine`."""
    key = (id(asyncio.get_running_loop()), machine_id, cwd)
    with _locks_guard:
        lock = _async_locks.setdefault(key, asyncio.Lock())
    async with lock:
        for name, content in files.items():
            await boxd_client.machines.files.upload(
                machine_id, f"{cwd}/{name}", content
            )
        return await boxd_client.machines.exec(
            machine_id, command, cwd=cwd, env=env, timeout=timeout
        )
