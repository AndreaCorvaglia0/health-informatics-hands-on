"""Lung segmentation inference: from a raw CT slice (HU) to a mask, on CPU.

This file is the single source of truth for "how a raw file becomes a mask".
It is built on day 4 (notebook 06) and used unchanged by the web app on day 5.

Conventions (same as the course dataset):
  * a slice is a 2D numpy array ``hu[x, y]`` in Hounsfield units, LPS index order
    (axis 0 = x towards the patient's left, axis 1 = y towards posterior).
    ``plt.imshow(hu.T)`` is the radiological view.
  * ``spacing = (sx, sy)`` in millimetres, same axis order as the array.

Dependencies: numpy, torch, monai, nibabel, pydicom, scipy (no streamlit, no matplotlib).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import pydicom
import torch
from monai.transforms import Resize, ScaleIntensityRange
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------------
# 1. Model folders (contract: model.pt = state_dict, config.json = how to rebuild)
# ---------------------------------------------------------------------------------
REQUIRED_CONFIG_KEYS = ("arch", "arch_kwargs", "preprocessing")


def find_model_dirs(root) -> list[Path]:
    """Sub-folders of ``root`` that contain both ``config.json`` and ``model.pt``."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.parent for p in root.rglob("config.json") if (p.parent / "model.pt").exists())


def load_model(model_dir, device="cpu"):
    """Rebuild the network from config.json and load its weights. Returns (model, cfg)."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{model_dir / 'config.json'} is missing the keys: {missing}")
    net = getattr(monai.networks.nets, cfg["arch"])(**cfg["arch_kwargs"])
    # weights_only=True: only tensors are unpickled, no arbitrary code can run
    state = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    net.load_state_dict(state)
    return net.to(device).eval(), cfg


# ---------------------------------------------------------------------------------
# 2. The three steps: preprocess -> predict -> postprocess (+ measure)
# ---------------------------------------------------------------------------------
def preprocess(hu: np.ndarray, cfg: dict) -> torch.Tensor:
    """HU slice [x, y] -> network input (1, 1, H, W).

    Uses the SAME MONAI transforms (and arguments) as ``get_transforms`` in training:
    clip to the HU window, scale to [0, 1], bilinear resize to the network size.
    """
    p = cfg["preprocessing"]
    x = torch.as_tensor(np.asarray(hu, dtype=np.float32))[None]  # channel first: (1, x, y)
    x = ScaleIntensityRange(a_min=p["hu_min"], a_max=p["hu_max"], b_min=0.0, b_max=1.0, clip=True)(x)
    x = Resize(spatial_size=tuple(p["img_size"]), mode="bilinear")(x)
    return torch.as_tensor(x, dtype=torch.float32)[None]  # add batch: (1, 1, H, W)


@torch.no_grad()
def predict(model, x: torch.Tensor) -> np.ndarray:
    """Network input (1, 1, H, W) -> probability of 'lung' at network resolution (H, W)."""
    device = next(model.parameters()).device
    logits = model(x.to(device))
    return torch.softmax(logits, dim=1)[0, 1].cpu().numpy()


def keep_largest_components(mask: np.ndarray, n: int = 2) -> np.ndarray:
    """Keep the ``n`` biggest connected components (two lungs) and fill holes in them."""
    labelled, n_found = ndi.label(mask)
    if n_found == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, labelled, index=range(1, n_found + 1))
    keep = np.argsort(sizes)[::-1][:n] + 1
    out = np.isin(labelled, keep)
    return ndi.binary_fill_holes(out).astype(np.uint8)


def upsample(prob: np.ndarray, original_shape) -> np.ndarray:
    """Resize a probability map (H, W) back to the original image shape (bilinear)."""
    p = Resize(spatial_size=tuple(original_shape), mode="bilinear")(torch.as_tensor(prob)[None])
    return np.asarray(p[0], dtype=np.float32)


def to_mask(prob: np.ndarray, threshold: float = 0.5, clean: bool = True) -> np.ndarray:
    """Probability map -> binary mask; optionally keep the two largest components and fill holes."""
    mask = (prob > threshold).astype(np.uint8)
    return keep_largest_components(mask) if clean else mask


def postprocess(prob: np.ndarray, original_shape, threshold: float = 0.5, clean: bool = True) -> np.ndarray:
    """Network-resolution probability map -> binary mask with the shape of the ORIGINAL image.

    The probability map is resized back (bilinear) BEFORE thresholding, which gives smoother
    borders than resizing a hard mask.
    """
    return to_mask(upsample(prob, original_shape), threshold, clean)


def measure(mask: np.ndarray, spacing) -> dict:
    """Measurements from a binary mask [x, y] and the pixel spacing (sx, sy) in mm."""
    pixel_area_cm2 = float(spacing[0]) * float(spacing[1]) / 100.0  # mm^2 -> cm^2
    out = {"area_cm2": float(mask.sum() * pixel_area_cm2), "area_pct_of_image": float(100 * mask.mean()),
           "n_components": 0, "right_cm2": None, "left_cm2": None}
    labelled, n = ndi.label(mask)
    out["n_components"] = int(n)
    if n >= 2:
        sizes = ndi.sum(mask, labelled, index=range(1, n + 1))
        two = np.argsort(sizes)[::-1][:2] + 1
        # LPS: x grows towards the patient's LEFT -> the component with the smaller x is the right lung
        cx = {int(k): ndi.center_of_mass(mask, labelled, int(k))[0] for k in two}
        right, left = sorted(two, key=lambda k: cx[int(k)])
        out["right_cm2"] = float((labelled == right).sum() * pixel_area_cm2)
        out["left_cm2"] = float((labelled == left).sum() * pixel_area_cm2)
    return out


@dataclass
class SegmentationResult:
    mask: np.ndarray                  # uint8 {0,1}, same shape as the input slice
    prob: np.ndarray                  # float32 in [0,1], same shape as the input slice
    prob_net: np.ndarray              # float32 in [0,1], network resolution (what the model really saw)
    measurements: dict = field(default_factory=dict)
    seconds: float = 0.0              # wall time of preprocess + predict + postprocess


def segment_slice(hu, spacing, model, cfg, threshold: float = 0.5, clean: bool = True) -> SegmentationResult:
    """THE function: raw HU slice [x, y] + spacing (mm) -> mask at the original resolution."""
    hu = np.asarray(hu)
    if hu.ndim != 2:
        raise ValueError(f"segment_slice expects a 2D slice [x, y], got an array of shape {hu.shape}")
    t0 = time.perf_counter()
    prob_net = predict(model, preprocess(hu, cfg))
    prob = upsample(prob_net, hu.shape)
    mask = to_mask(prob, threshold, clean)
    seconds = time.perf_counter() - t0
    return SegmentationResult(mask=mask, prob=prob, prob_net=prob_net.astype(np.float32),
                              measurements=measure(mask, spacing), seconds=seconds)


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    """Dice between two binary masks (1.0 if both are empty)."""
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * (a & b).sum() / denom)


# ---------------------------------------------------------------------------------
# 3. Readers: NIfTI and DICOM -> (hu[x, y], spacing)
# ---------------------------------------------------------------------------------
class NotASliceError(ValueError):
    """The file holds a 3D volume with more than one slice: the caller must choose a slice."""


def load_nifti(path):
    """Return (array, spacing_xy, affine). The array can be 2D or 3D [x, y, z]."""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    zooms = img.header.get_zooms()
    return data, (float(zooms[0]), float(zooms[1])), img.affine


def load_dicom(path):
    """Return (hu[x, y], spacing_xy) from one DICOM file (a single slice)."""
    ds = pydicom.dcmread(str(path))
    if "PixelData" not in ds:
        raise ValueError("this DICOM file has no pixel data")
    pixels = ds.pixel_array.astype(np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"expected a single-channel 2D DICOM slice, got pixel_array of shape {pixels.shape}")
    # stored values -> Hounsfield units
    hu = pixels * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    # DICOM: pixel_array is [row, col] = [y, x]; PixelSpacing = [row spacing, column spacing] = [sy, sx]
    sy, sx = (float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0]))
    return hu.T, (sx, sy)  # transpose to our [x, y]


def load_any(path, slice_index: int | None = None):
    """Read a NIfTI (.nii/.nii.gz) or DICOM (.dcm) slice. Returns (hu[x, y], spacing_xy).

    A 3D NIfTI with a single slice is squeezed; with several slices the caller must pass
    ``slice_index`` (otherwise ``NotASliceError`` is raised).
    """
    name = Path(path).name.lower()
    if name.endswith((".nii", ".nii.gz")):
        data, spacing, _ = load_nifti(path)
        return pick_slice(data, slice_index), spacing
    if name.endswith((".dcm", ".dicom")):
        return load_dicom(path)
    raise ValueError(f"unsupported file type: {Path(path).name} (use .nii, .nii.gz or .dcm)")


def pick_slice(data: np.ndarray, slice_index: int | None = None) -> np.ndarray:
    """2D array -> itself; 3D [x, y, z] -> the chosen slice (single-slice volumes are squeezed)."""
    data = np.asarray(data)
    if data.ndim == 2:
        return data.astype(np.float32)
    if data.ndim == 3 and data.shape[2] == 1:
        return data[..., 0].astype(np.float32)
    if data.ndim == 3:
        if slice_index is None:
            raise NotASliceError(f"volume with {data.shape[2]} slices: choose one (slice_index)")
        return data[..., int(slice_index)].astype(np.float32)
    raise ValueError(f"expected a 2D slice or a 3D volume, got shape {data.shape}")


def orientation_warning(affine) -> str | None:
    """Warn if a NIfTI is not stored in the course's LPS index order (arrays would look flipped)."""
    codes = nib.aff2axcodes(affine)
    if tuple(codes[:2]) != ("L", "P"):
        return (f"This file is stored as {''.join(codes)} (axis 0 -> {codes[0]}, axis 1 -> {codes[1]}); the "
                "model was trained on LPS arrays (axis 0 -> L, axis 1 -> P). The result may look flipped "
                "or be wrong.")
    return None


def mask_to_nifti_bytes(mask: np.ndarray, affine=None, spacing=(1.0, 1.0)) -> bytes:
    """Serialise a 2D mask as a gzipped NIfTI. Without an input affine, build the course's LPS one."""
    if affine is None:
        affine = np.diag([-float(spacing[0]), -float(spacing[1]), 1.0, 1.0])
    import tempfile

    img = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), affine)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mask.nii.gz"
        nib.save(img, str(out))
        return out.read_bytes()
