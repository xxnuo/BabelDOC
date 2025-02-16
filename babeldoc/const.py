from pathlib import Path
import os
import sys

ROOT_DIR: str = os.path.abspath(os.getenv("ROOT_DIR", os.path.dirname(sys.argv[0])))

CACHE_FOLDER = os.path.join(ROOT_DIR, "data/cache")


def get_cache_file_path(filename):
    return CACHE_FOLDER / filename
