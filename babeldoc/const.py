import shutil
import subprocess
from pathlib import Path
import os
import sys

__version__ = "0.1.13"

ROOT_DIR: str = os.path.abspath(os.getenv("ROOT_DIR", os.path.dirname(sys.argv[0])))

CACHE_FOLDER = Path(os.path.join(ROOT_DIR, "data/cache"))


def get_cache_file_path(filename):
    return CACHE_FOLDER / filename


try:
    git_path = shutil.which("git")
    if git_path is None:
        raise FileNotFoundError("git executable not found")

    WATERMARK_VERSION = (
        subprocess.check_output(  # noqa: S603
            [git_path, "describe", "--always"],
            cwd=Path(__file__).resolve().parent,
        )
        .strip()
        .decode()
    )
except (OSError, FileNotFoundError, subprocess.CalledProcessError):
    WATERMARK_VERSION = f"v{__version__}"
