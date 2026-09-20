import os
import subprocess
import sys
from pathlib import Path

from app.core.config import PROJECT_ROOT, Settings


def test_default_data_paths_live_in_the_project_root():
    s = Settings(_env_file=None)
    assert Path(s.SQLITE_DB_PATH) == PROJECT_ROOT / "memory.db"
    assert Path(s.QDRANT_STORAGE_PATH) == PROJECT_ROOT / "qdrant_storage"
    assert Path(s.FASTEMBED_CACHE_PATH) == PROJECT_ROOT / "fastembed_cache"


def test_paths_do_not_depend_on_the_launch_directory(tmp_path):
    code = "from app.core.config import Settings; print(Settings(_env_file=None).SQLITE_DB_PATH)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    ).stdout.strip()
    assert out == str(PROJECT_ROOT / "memory.db")


def test_absolute_paths_and_memory_db_are_left_alone(tmp_path):
    assert Settings(_env_file=None, SQLITE_DB_PATH=str(tmp_path / "x.db")).SQLITE_DB_PATH == str(tmp_path / "x.db")
    assert Settings(_env_file=None, SQLITE_DB_PATH=":memory:").SQLITE_DB_PATH == ":memory:"
    assert Settings(_env_file=None, QDRANT_STORAGE_PATH="my_qdrant").QDRANT_STORAGE_PATH == str(PROJECT_ROOT / "my_qdrant")
