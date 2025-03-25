PYTHON_EVALUATOR_SEPARATOR = "---python-eval-start---"
PYTHON_EVALUATOR_FILE = """
import json
import subprocess
def _parse_pyright_output(stdout: bytes):
    try:
        # Parse the JSON output
        output = json.loads(stdout)

        errors = []
        for error in output.get("generalDiagnostics", []):
            if (error.get("severity", None) == "error"):
                del error["file"]
                errors.append(error)
        score = len(errors) == 0
        return (score, json.dumps(errors))
    except json.JSONDecodeError:
        return (False, f"Failed to parse Pyright output: " + stdout.decode())


def _analyze_with_pyright(
    *,
    pyright_cli_args = None,
):
    result = subprocess.run(
        [
            "pyright",
            "--outputjson",
            "--level",
            "error",  # Only report errors, not warnings
            *(pyright_cli_args or []),
            "outputs.py",
        ],
        capture_output=True,
    )

    print(json.dumps(_parse_pyright_output(result.stdout)))


_analyze_with_pyright()
"""

EXTRACT_IMPORT_NAMES = """
import ast
import sys

BUILTIN_MODULES = set(sys.stdlib_module_names)

def extract_import_names(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    python_imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                import_path = name.name
                if not import_path.startswith((".", "/")):
                    base_package = import_path.split(".")[0]
                    if base_package not in python_imports and base_package not in BUILTIN_MODULES:
                        python_imports.append(base_package)
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.module.startswith((".", "/")):
                base_package = node.module.split(".")[0]
                if base_package not in python_imports and base_package not in BUILTIN_MODULES:
                    python_imports.append(base_package)

    return python_imports

file_path = "./outputs.py"
imports = extract_import_names(file_path)
print("\\n".join(imports))
"""
