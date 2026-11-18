"""Shared pytest setup: make ``inference`` importable and find a model folder."""
import os
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))


def _model_dirs():
    from inference import find_model_dirs

    env = os.environ.get("HI_MODEL_DIR")
    if env:
        return [Path(env)]
    return find_model_dirs(APP_DIR / "models")


@pytest.fixture(scope="session")
def model_and_cfg():
    """The model under test: $HI_MODEL_DIR, else the first folder in models/. Skip if none."""
    dirs = _model_dirs()
    if not dirs:
        pytest.skip("no model found: set HI_MODEL_DIR or put a model folder in models/")
    from inference import load_model

    return load_model(dirs[0])
