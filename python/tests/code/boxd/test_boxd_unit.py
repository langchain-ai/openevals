import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from boxd import ExecResult

from openevals.code.boxd.execution import (
    create_async_boxd_execution_evaluator,
    create_boxd_execution_evaluator,
)
from openevals.code.boxd.pyright import (
    create_async_boxd_pyright_evaluator,
    create_boxd_pyright_evaluator,
)
from openevals.code.e2b.sandbox.files import (
    EXTRACT_IMPORT_NAMES,
    PYTHON_EVALUATOR_FILE,
    PYTHON_EVALUATOR_SEPARATOR,
)


class _FakeFiles:
    def __init__(self, calls):
        self._calls = calls

    def upload(self, machine_id, path, data):
        self._calls.append(("upload", machine_id, path, data))
        return len(data)


class _FakeMachines:
    def __init__(self, result):
        self.calls = []
        self.result = result
        self.files = _FakeFiles(self.calls)

    def exec(self, machine_id, command, *, env=None, cwd=None, timeout=None):
        self.calls.append(
            ("exec", machine_id, command, {"env": env, "cwd": cwd, "timeout": timeout})
        )
        return self.result


class FakeBoxd:
    def __init__(self, result):
        self.machines = _FakeMachines(result)


class _AsyncFakeFiles(_FakeFiles):
    async def upload(self, machine_id, path, data):
        return super().upload(machine_id, path, data)


class _AsyncFakeMachines(_FakeMachines):
    def __init__(self, result):
        super().__init__(result)
        self.files = _AsyncFakeFiles(self.calls)

    async def exec(self, machine_id, command, *, env=None, cwd=None, timeout=None):
        return super().exec(machine_id, command, env=env, cwd=cwd, timeout=timeout)


class AsyncFakeBoxd:
    def __init__(self, result):
        self.machines = _AsyncFakeMachines(result)


def _uploads(fake):
    return {c[2]: c[3] for c in fake.machines.calls if c[0] == "upload"}


def _run(fake):
    (run,) = [c for c in fake.machines.calls if c[0] == "exec"]
    return run


def _ok(stdout=""):
    return ExecResult(stdout=stdout, stderr="", exit_code=0)


def _pyright_stdout(score, errors):
    return f"{PYTHON_EVALUATOR_SEPARATOR}\n{json.dumps([score, errors])}\n"


def test_execution_uploads_code_and_runs_it_in_the_project_directory():
    fake = FakeBoxd(_ok())
    evaluator = create_boxd_execution_evaluator(
        boxd_client=fake, machine_id="m-1", environment_variables={"K": "V"}
    )
    result = evaluator(outputs="print('hi')")
    assert result["score"] is True
    assert result["key"] == "execution_succeeded"
    assert fake.machines.calls[0] == (
        "upload",
        "m-1",
        "openevals/outputs.py",
        "print('hi')",
    )
    run = _run(fake)
    assert run[1] == "m-1"
    assert run[3] == {"env": {"K": "V"}, "cwd": "openevals", "timeout": 300.0}
    assert "uv run python outputs.py" in run[2]


def test_execution_reuses_an_existing_virtualenv():
    fake = FakeBoxd(_ok())
    evaluator = create_boxd_execution_evaluator(boxd_client=fake, machine_id="m-1")
    evaluator(outputs="print(1)")
    evaluator(outputs="print(2)")
    commands = [c[2] for c in fake.machines.calls if c[0] == "exec"]
    assert len(commands) == 2
    assert all("uv venv --quiet --allow-existing" in c for c in commands)


def test_helper_scripts_are_uploaded_as_files_not_echoed():
    fake = FakeBoxd(_ok(_pyright_stdout(True, "[]")))
    create_boxd_pyright_evaluator(boxd_client=fake, machine_id="m-1")(outputs="x = 1")
    uploads = _uploads(fake)
    assert uploads["openevals/extract_import_names.py"] == EXTRACT_IMPORT_NAMES
    assert uploads["openevals/run_pyright.py"] == PYTHON_EVALUATOR_FILE
    assert "\\n" in uploads["openevals/extract_import_names.py"]
    command = _run(fake)[2]
    assert "> extract_import_names.py" not in command
    assert "> run_pyright.py" not in command
    assert EXTRACT_IMPORT_NAMES not in command

    fake = FakeBoxd(_ok())
    create_boxd_execution_evaluator(boxd_client=fake, machine_id="m-1")(outputs="x = 1")
    uploads = _uploads(fake)
    assert uploads["openevals/extract_import_names.py"] == EXTRACT_IMPORT_NAMES
    assert next(iter(uploads)) == "openevals/outputs.py"


def test_execution_failure_returns_exit_code_and_stderr():
    fake = FakeBoxd(
        ExecResult(stdout="", stderr="ModuleNotFoundError: nope", exit_code=1)
    )
    evaluator = create_boxd_execution_evaluator(boxd_client=fake, machine_id="m-1")
    result = evaluator(outputs="import nope")
    assert result["score"] is False
    assert "code 1" in result["comment"]
    assert "ModuleNotFoundError: nope" in result["comment"]


def test_execution_failure_falls_back_to_stdout_when_stderr_is_empty():
    fake = FakeBoxd(ExecResult(stdout="only stdout", stderr="", exit_code=2))
    evaluator = create_boxd_execution_evaluator(boxd_client=fake, machine_id="m-1")
    result = evaluator(outputs="x")
    assert result["score"] is False
    assert "only stdout" in result["comment"]


def test_execution_options_reach_the_machine():
    fake = FakeBoxd(_ok())
    evaluator = create_boxd_execution_evaluator(
        boxd_client=fake,
        machine_id="m-2",
        execution_command="python -X dev",
        sandbox_project_directory="/home/boxd/project",
        timeout=12.5,
    )
    evaluator(outputs="print(1)")
    assert "/home/boxd/project/outputs.py" in _uploads(fake)
    run = _run(fake)
    assert run[3] == {"env": None, "cwd": "/home/boxd/project", "timeout": 12.5}
    assert "uv run python -X dev outputs.py" in run[2]


def test_execution_extracts_markdown_code_blocks_before_upload():
    fake = FakeBoxd(_ok())
    evaluator = create_boxd_execution_evaluator(
        boxd_client=fake,
        machine_id="m-1",
        code_extraction_strategy="markdown_code_blocks",
    )
    evaluator(outputs="Here you go:\n```python\nprint('hi')\n```\n")
    assert fake.machines.calls[0][3].strip() == "print('hi')"


def test_execution_without_code_block_does_not_touch_the_machine():
    fake = FakeBoxd(_ok())
    evaluator = create_boxd_execution_evaluator(
        boxd_client=fake,
        machine_id="m-1",
        code_extraction_strategy="markdown_code_blocks",
    )
    result = evaluator(outputs="no code here")
    assert result["score"] is False
    assert fake.machines.calls == []


def test_model_requires_llm_extraction():
    with pytest.raises(ValueError):
        create_boxd_execution_evaluator(
            boxd_client=FakeBoxd(_ok()), machine_id="m-1", model="openai:gpt-4o"
        )
    with pytest.raises(ValueError):
        create_boxd_pyright_evaluator(
            boxd_client=FakeBoxd(_ok()), machine_id="m-1", model="openai:gpt-4o"
        )


def test_pyright_parses_the_result_after_the_separator():
    fake = FakeBoxd(_ok("installing...\n" + _pyright_stdout(False, "[{}]")))
    evaluator = create_boxd_pyright_evaluator(boxd_client=fake, machine_id="m-1")
    result = evaluator(outputs="x: int = 'a'")
    assert result["score"] is False
    assert result["comment"] == "[{}]"
    assert result["key"] == "pyright_succeeded"
    assert _uploads(fake)["openevals/outputs.py"] == "x: int = 'a'"
    run = _run(fake)
    assert run[3] == {"env": None, "cwd": "openevals", "timeout": 300.0}


def test_pyright_passes_clean_code():
    fake = FakeBoxd(_ok(_pyright_stdout(True, "[]")))
    evaluator = create_boxd_pyright_evaluator(boxd_client=fake, machine_id="m-1")
    assert evaluator(outputs="x: int = 1")["score"] is True


def test_pyright_finds_npm_global_binaries():
    fake = FakeBoxd(_ok(_pyright_stdout(True, "[]")))
    create_boxd_pyright_evaluator(boxd_client=fake, machine_id="m-1")(outputs="x = 1")
    command = _run(fake)[2]
    path_at = command.index('export PATH="$(npm config get prefix)/bin:$PATH"')
    assert path_at < command.index('command -v "pyright"')
    assert "uv venv --quiet --allow-existing" in command


def test_pyright_failed_command_returns_stderr():
    fake = FakeBoxd(ExecResult(stdout="", stderr="npm ERR! boom", exit_code=1))
    result = create_boxd_pyright_evaluator(boxd_client=fake, machine_id="m-1")(
        outputs="x = 1"
    )
    assert result["score"] is False
    assert "npm ERR! boom" in result["comment"]


def test_pyright_output_without_separator_fails_with_stdout():
    fake = FakeBoxd(_ok("unexpected"))
    result = create_boxd_pyright_evaluator(boxd_client=fake, machine_id="m-1")(
        outputs="x = 1"
    )
    assert result["score"] is False
    assert result["comment"] == "unexpected"


@pytest.mark.asyncio
async def test_async_execution_uploads_and_runs():
    fake = AsyncFakeBoxd(ExecResult(stdout="", stderr="Traceback", exit_code=1))
    evaluator = create_async_boxd_execution_evaluator(
        boxd_client=fake, machine_id="m-3", timeout=None
    )
    result = await evaluator(outputs="raise SystemExit(1)")
    assert result["score"] is False
    assert "Traceback" in result["comment"]
    assert _uploads(fake)["openevals/outputs.py"] == "raise SystemExit(1)"
    run = _run(fake)
    assert run[3] == {"env": None, "cwd": "openevals", "timeout": None}


@pytest.mark.asyncio
async def test_async_pyright_parses_the_result():
    fake = AsyncFakeBoxd(_ok(_pyright_stdout(True, "[]")))
    evaluator = create_async_boxd_pyright_evaluator(boxd_client=fake, machine_id="m-3")
    result = await evaluator(outputs="x: int = 1")
    assert result["score"] is True
    assert _run(fake)[3]["cwd"] == "openevals"
    assert set(_uploads(fake)) == {
        "openevals/outputs.py",
        "openevals/extract_import_names.py",
        "openevals/run_pyright.py",
    }


class _OverlapMachines:
    """Records how many evaluations touch one project directory at once."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.mismatches = 0
        self.current = {}
        self.files = self

    def upload(self, machine_id, path, data):
        if path.endswith("/outputs.py"):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.current[path] = data
        return len(data)

    def exec(self, machine_id, command, *, env=None, cwd=None, timeout=None):
        seen = self.current[f"{cwd}/outputs.py"]
        time.sleep(0.05)
        if self.current[f"{cwd}/outputs.py"] != seen:
            self.mismatches += 1
        self.active -= 1
        return ExecResult(stdout="", stderr="", exit_code=0)


def test_concurrent_evaluations_on_one_directory_do_not_overlap():
    machines = _OverlapMachines()
    fake = type("FakeBoxd", (), {"machines": machines})()
    execution = create_boxd_execution_evaluator(boxd_client=fake, machine_id="m-1")
    pyright = create_boxd_pyright_evaluator(boxd_client=fake, machine_id="m-1")
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(ev, outputs=f"print({i})")
            for i, ev in enumerate([execution, pyright] * 3)
        ]
        for f in futures:
            f.result()
    assert machines.max_active == 1
    assert machines.mismatches == 0


class _AsyncOverlapMachines(_OverlapMachines):
    async def upload(self, machine_id, path, data):
        return _OverlapMachines.upload(self, machine_id, path, data)

    async def exec(self, machine_id, command, *, env=None, cwd=None, timeout=None):
        seen = self.current[f"{cwd}/outputs.py"]
        await asyncio.sleep(0.05)
        if self.current[f"{cwd}/outputs.py"] != seen:
            self.mismatches += 1
        self.active -= 1
        return ExecResult(stdout="", stderr="", exit_code=0)


@pytest.mark.asyncio
async def test_concurrent_async_evaluations_on_one_directory_do_not_overlap():
    machines = _AsyncOverlapMachines()
    fake = type("FakeBoxd", (), {"machines": machines})()
    evaluator = create_async_boxd_execution_evaluator(
        boxd_client=fake, machine_id="m-1"
    )
    await asyncio.gather(*(evaluator(outputs=f"print({i})") for i in range(5)))
    assert machines.max_active == 1
    assert machines.mismatches == 0


@pytest.mark.parametrize("timeout", [0, -1])
def test_timeout_must_be_positive(timeout):
    with pytest.raises(ValueError):
        create_boxd_execution_evaluator(
            boxd_client=FakeBoxd(_ok()), machine_id="m-1", timeout=timeout
        )
    with pytest.raises(ValueError):
        create_async_boxd_pyright_evaluator(
            boxd_client=AsyncFakeBoxd(_ok()), machine_id="m-1", timeout=timeout
        )
