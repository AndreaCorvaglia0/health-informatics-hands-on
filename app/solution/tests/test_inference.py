"""Tests for inference.py. Run from the app folder:  python -m pytest -q

They check the *behaviour* that must never change: shapes, value ranges, determinism and, above all,
that preprocessing is identical to the one used in training (no training-serving skew).
"""
import numpy as np
import nibabel as nib
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import inference as inf

CFG = {"preprocessing": {"hu_min": -1000, "hu_max": 400, "img_size": [32, 32]}}


def fake_slice(shape=(96, 80), seed=0):
    """A blob of 'lung' (-800 HU) in 'soft tissue' (40 HU), with a padding value in the corners."""
    rng = np.random.default_rng(seed)
    hu = np.full(shape, 40.0, dtype=np.float32)
    hu[20:60, 15:65] = -800
    hu += rng.normal(0, 5, shape).astype(np.float32)
    hu[:5, :5] = -2000  # scanner padding value
    return hu


# --- preprocessing (no model needed) ------------------------------------------------------------
def test_preprocess_shape_and_range():
    x = inf.preprocess(fake_slice(), CFG)
    assert tuple(x.shape) == (1, 1, 32, 32)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_preprocess_clips_padding_values():
    a, b = fake_slice(), fake_slice()
    b[:5, :5] = -3024  # a different padding value must not change anything
    assert np.allclose(inf.preprocess(a, CFG).numpy(), inf.preprocess(b, CFG).numpy())


def test_preprocess_is_identical_to_training_transforms(tmp_path):
    """The key anti-skew test: same numbers as the training pipeline (get_transforms)."""
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Resized, ScaleIntensityRanged

    hu = fake_slice().astype(np.int16)
    path = tmp_path / "img.nii.gz"
    nib.save(nib.Nifti1Image(hu, np.diag([-0.7, -0.7, 1.0, 1.0])), str(path))
    train_like = Compose([  # the image part of get_transforms() in data_prep/train_baseline.py
        LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(keys="image", a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys="image", spatial_size=(32, 32), mode="bilinear"),
    ])
    expected = np.asarray(train_like({"image": str(path)})["image"])
    got = inf.preprocess(inf.load_any(path)[0], CFG)[0].numpy()
    assert np.allclose(expected, got, atol=1e-6)


# --- the full pipeline (needs a model) -------------------------------------------------------------
@pytest.mark.parametrize("shape", [(96, 80), (64, 64), (101, 57)])
def test_segment_slice_preserves_shape(model_and_cfg, shape):
    model, cfg = model_and_cfg
    res = inf.segment_slice(fake_slice(shape), (0.7, 0.7), model, cfg)
    assert res.mask.shape == shape and res.prob.shape == shape
    assert set(np.unique(res.mask)) <= {0, 1}
    assert 0.0 <= res.prob.min() and res.prob.max() <= 1.0


def test_segment_slice_uniform_soft_tissue_is_not_lung(model_and_cfg):
    # Not 'all air': a uniform air image looks like lung, so any answer is acceptable there.
    model, cfg = model_and_cfg
    res = inf.segment_slice(np.full((128, 128), 40.0, dtype=np.float32), (1.0, 1.0), model, cfg)
    assert res.mask.mean() < 0.5


def test_segment_slice_is_deterministic(model_and_cfg):
    model, cfg = model_and_cfg
    hu = fake_slice()
    a = inf.segment_slice(hu, (0.7, 0.7), model, cfg)
    b = inf.segment_slice(hu, (0.7, 0.7), model, cfg)
    assert np.array_equal(a.mask, b.mask) and np.allclose(a.prob, b.prob)


def test_segment_slice_rejects_3d(model_and_cfg):
    model, cfg = model_and_cfg
    with pytest.raises(ValueError):
        inf.segment_slice(np.zeros((8, 8, 8)), (1, 1), model, cfg)


def test_threshold_monotonic(model_and_cfg):
    model, cfg = model_and_cfg
    hu = fake_slice()
    lo = inf.segment_slice(hu, (1, 1), model, cfg, threshold=0.2, clean=False).mask.sum()
    hi = inf.segment_slice(hu, (1, 1), model, cfg, threshold=0.8, clean=False).mask.sum()
    assert lo >= hi


# --- post-processing, measurements, metrics -------------------------------------------------------
def test_measure_area_and_sides():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:30, 10:30] = 1   # low x  -> right lung (LPS), 400 px
    mask[10:40, 60:90] = 1   # high x -> left lung, 900 px
    m = inf.measure(mask, (0.5, 0.5))  # 0.25 mm2 per pixel = 0.0025 cm2
    assert m["area_cm2"] == pytest.approx(1300 * 0.0025)
    assert m["right_cm2"] == pytest.approx(400 * 0.0025) and m["left_cm2"] == pytest.approx(900 * 0.0025)


def test_keep_largest_components_removes_specks_and_fills_holes():
    mask = np.zeros((50, 50), np.uint8)
    mask[5:25, 5:25] = 1
    mask[10:12, 10:12] = 0      # a hole
    mask[30:45, 30:45] = 1      # second lung
    mask[48, 48] = 1            # speck
    out = inf.keep_largest_components(mask)
    assert out[48, 48] == 0 and out[10, 10] == 1 and out[35, 35] == 1


def test_dice_score():
    a = np.zeros((10, 10)); a[:5] = 1
    assert inf.dice_score(a, a) == 1.0
    assert inf.dice_score(a, 1 - a) == 0.0
    assert inf.dice_score(np.zeros((4, 4)), np.zeros((4, 4))) == 1.0


# --- readers ---------------------------------------------------------------------------------------
def test_nifti_roundtrip_and_volume(tmp_path):
    hu = fake_slice().astype(np.int16)
    p2 = tmp_path / "s.nii.gz"
    nib.save(nib.Nifti1Image(hu, np.diag([-0.8, -0.8, 1.0, 1.0])), str(p2))
    got, spacing = inf.load_any(p2)
    assert got.shape == hu.shape and spacing == pytest.approx((0.8, 0.8))
    p3 = tmp_path / "v.nii.gz"
    nib.save(nib.Nifti1Image(np.stack([hu, hu, hu], -1), np.diag([-0.8, -0.8, 2.0, 1.0])), str(p3))
    with pytest.raises(inf.NotASliceError):
        inf.load_any(p3)
    assert inf.load_any(p3, slice_index=1)[0].shape == hu.shape


def test_load_any_rejects_unknown_type(tmp_path):
    bad = tmp_path / "x.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        inf.load_any(bad)


def write_dicom(path, pixels_row_col, slope, intercept, spacing_row_col):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = pixels_row_col.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    ds.PixelSpacing = list(spacing_row_col)
    ds.PixelData = pixels_row_col.astype(np.int16).tobytes()
    ds.save_as(str(path))


def test_dicom_hu_orientation_and_spacing(tmp_path):
    stored = np.arange(12 * 20, dtype=np.int16).reshape(12, 20)  # [row, col] = [y, x]
    p = tmp_path / "s.dcm"
    write_dicom(p, stored, slope=1.0, intercept=-1024.0, spacing_row_col=(0.9, 0.7))
    hu, (sx, sy) = inf.load_any(p)
    assert hu.shape == (20, 12)                       # transposed to [x, y]
    assert hu[3, 5] == stored[5, 3] - 1024            # HU = stored * slope + intercept
    assert (sx, sy) == pytest.approx((0.7, 0.9))      # PixelSpacing is [row, col] = [sy, sx]


def test_mask_to_nifti_bytes_roundtrip(tmp_path):
    mask = (fake_slice() < 0).astype(np.uint8)
    data = inf.mask_to_nifti_bytes(mask, spacing=(0.7, 0.7))
    out = tmp_path / "m.nii.gz"
    out.write_bytes(data)
    back = np.asanyarray(nib.load(str(out)).dataobj)
    assert np.array_equal(back, mask)


# --- single source of truth ----------------------------------------------------------------------
def test_inference_py_is_identical_in_starter_and_notebook():
    """app/solution/inference.py == app/starter/inference.py == the file written by notebook 06."""
    from conftest import APP_DIR

    starter = APP_DIR.parent / "starter" / "inference.py"
    notebook = APP_DIR.parents[1] / "src" / "06_inference_and_packaging.py"
    if not (starter.exists() and notebook.exists()):
        pytest.skip("not running inside the course repository")
    ref = (APP_DIR / "inference.py").read_text()
    assert starter.read_text() == ref, "app/starter/inference.py differs from app/solution/inference.py"
    text = notebook.read_text()
    embedded = text.split("INFERENCE_PY = r'''", 1)[1].split("\n'''", 1)[0] + "\n"
    assert embedded == ref, "src/06_inference_and_packaging.py embeds a different inference.py: regenerate it"
