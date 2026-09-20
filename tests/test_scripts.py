import py_compile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_helper_scripts_are_valid_python():
    files = sorted(SCRIPTS.glob("*.py"))
    assert {f.name for f in files} >= {"bench.py", "chat_cli.py"}
    for f in files:
        py_compile.compile(str(f), doraise=True)
