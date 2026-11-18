"""Headless smoke tests of app.py with Streamlit's AppTest (no browser needed).

The app is copied to a temporary folder together with a model and a few generated samples, so the
tests do not depend on what you put in models/ and samples/. Skipped when no model is available.
"""
import shutil

import numpy as np
import nibabel as nib
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import APP_DIR, _model_dirs  # noqa: E402


def make_nifti(path, arr, spacing=(0.7, 0.7, 1.0)):
    nib.save(nib.Nifti1Image(arr, np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])), str(path))


@pytest.fixture()
def app_copy(tmp_path):
    dirs = _model_dirs()
    if not dirs:
        pytest.skip("no model found: set HI_MODEL_DIR or put a model folder in models/")
    for name in ("app.py", "inference.py"):
        shutil.copy(APP_DIR / name, tmp_path / name)
    shutil.copytree(dirs[0], tmp_path / "models" / "model-a")
    shutil.copytree(dirs[0], tmp_path / "models" / "model-b")
    (tmp_path / "samples").mkdir()
    yy, xx = np.mgrid[0:96, 0:96]
    hu = np.full((96, 96), 40, np.int16)
    hu[((xx - 30) ** 2 + (yy - 48) ** 2) < 400] = -850
    hu[((xx - 66) ** 2 + (yy - 48) ** 2) < 400] = -850
    make_nifti(tmp_path / "samples" / "case_a.nii.gz", hu)
    make_nifti(tmp_path / "samples" / "case_a_label.nii.gz", (hu < -500).astype(np.uint8))
    make_nifti(tmp_path / "samples" / "volume.nii.gz", np.stack([hu] * 5, axis=-1))
    return tmp_path / "app.py"


def run(app_file):
    return AppTest.from_file(str(app_file), default_timeout=120).run()


def test_app_starts_without_exception(app_copy):
    at = run(app_copy)
    assert not at.exception
    assert not at.error


def test_every_bundled_sample_runs(app_copy):
    at = run(app_copy)
    n = len(at.selectbox[0].options)
    assert n == 2  # case_a.nii.gz and volume.nii.gz (labels are not listed as samples)
    for i in range(n):
        at.selectbox[0].select_index(i).run()
        assert not at.exception, at.exception
        assert not at.error, [e.value for e in at.error]


def test_ground_truth_is_used_for_dice(app_copy):
    at = run(app_copy)
    labels = [m.label for m in at.metric]
    assert "Dice" in labels  # the first sample (case_a) has a *_label.nii.gz next to it
    dice = float(next(m.value for m in at.metric if m.label == "Dice"))
    assert 0.0 <= dice <= 1.0


def test_volume_shows_a_slice_slider(app_copy):
    at = run(app_copy)
    at.selectbox[0].select_index(1).run()
    assert any("3D volume" in s.label for s in at.slider)
    assert not at.exception


def test_compare_two_models(app_copy):
    at = run(app_copy)
    at.sidebar.checkbox[1].check().run()  # "Compare with a second model" (checkbox 0 = post-processing)
    assert not at.exception, at.exception
    assert any(m.label == "Dice between A and B" for m in at.metric)
