# %% [markdown]
# # Notebook 06 - From a notebook to something others can use
#
# **Day 4 - part 2** (about 2 hours)
#
# A trained model in a notebook is not a product. In this notebook we build the piece that sits between
# the model and the web app: **one code path that turns a raw file (NIfTI or DICOM, any size, in Hounsfield
# units) into a lung mask at the original resolution, on a CPU, exactly as in training**. We test it, time it,
# export it, document it in a model card, and pack everything into `app_starter_kit.zip`, which you will
# take to VS Code tomorrow.
#
# **Learning objectives.** After this notebook you can:
# * write `preprocess -> predict -> postprocess -> measure` and compose them into one `segment_slice` function;
# * explain **training-serving skew**, measure its effect, and prevent it (shared code, tests);
# * read NIfTI and DICOM slices correctly (HU conversion, axis order, pixel spacing);
# * measure latency and model size, and judge whether a model fits a free CPU host;
# * export a model (`state_dict` + config, TorchScript, ONNX) and know why we avoid pickled models;
# * write a model card and a first set of tests, and hand over a clean package to the app.
#
# **Icons.**
# | Marker | Meaning |
# |---|---|
# | ✏️ Exercise | **Mandatory**, with blanks (`____`) and a check cell. Hints are hidden under 💡. |
# | 🔍 Explore | Small experiments: change something and re-run. |
# | 🚀 Challenge | Additional, for those who finish early. |
# | 🌶️ Going further | Optional and open-ended. |
# | 🧠 Question | Think and answer in words. |
# | ⚠️ Pitfall | A classic mistake. |
# | 📚 Read more | Links. |

# %% [markdown]
# ## 0. Setup, configuration, model

# %%
# Setup: install what is missing (quietly). On Kaggle/Colab the internet must be switched on.
import importlib.util
import subprocess
import sys

for pip_name, module in [("monai", "monai"), ("nibabel", "nibabel"), ("pydicom", "pydicom"),
                         ("scipy", "scipy"), ("matplotlib", "matplotlib"), ("pandas", "pandas")]:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)

# %%
import os
from pathlib import Path

# --- Debug run: tiny & fast, to check that everything works before a real run ---
FAST = os.environ.get("HI_FAST", "0") == "1"   # set FAST = True by hand to debug quickly

# --- Where is the data? -----------------------------------------------------------
def find_data_dir() -> Path:
    """Locate the course dataset (folder containing images/, labels/, metadata.csv)."""
    candidates = [os.environ.get("HI_DATA_DIR")]
    candidates += sorted(Path("/kaggle/input").glob("*/")) if Path("/kaggle/input").exists() else []
    candidates += sorted(Path("/kaggle/input").glob("*/*/")) if Path("/kaggle/input").exists() else []
    candidates += ["data/luna16-lung-slices", "../data/luna16-lung-slices", "data/synthetic", "../data/synthetic"]
    for c in candidates:
        if c and (Path(c) / "metadata.csv").exists():
            return Path(c)
    raise FileNotFoundError(
        "Dataset not found. On Kaggle: 'Add input' and attach the course dataset. "
        "Elsewhere: download it (see README) and set HI_DATA_DIR."
    )

DATA_DIR = find_data_dir()
WORK_DIR = Path(os.environ.get("HI_WORK_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "outputs"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
print(f"data: {DATA_DIR}\noutputs: {WORK_DIR}\nFAST mode: {FAST}")

# %%
import json
import shutil
import time
import warnings
import zipfile
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import torch
from monai.data import Dataset
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped, Lambdad, LoadImaged, RandAffined,
                              RandGaussianNoised, Resize, Resized, ScaleIntensityRange, ScaleIntensityRanged)
from monai.utils import set_determinism
from scipy import ndimage as ndi

set_determinism(seed=42)
warnings.filterwarnings("ignore", category=UserWarning, module="monai")
plt.rcParams.update({"figure.dpi": 100, "image.cmap": "gray"})

# The web app will run on a CPU (free hosts have no GPU): we test on the CPU too, on purpose.
# A GPU would hide problems (speed, memory) that your users would meet.
device = torch.device("cpu")

N_EVAL = 6 if FAST else 40        # how many test cases we evaluate in this notebook
N_RUNS = 5 if FAST else 30        # repetitions for the timing experiments

# %% [markdown]
# ### Recap: the canonical training pipeline (given)
# These three functions are copied **verbatim** from `data_prep/train_baseline.py`, the reference of notebooks
# 03-05. Whatever the web app does, it must give the same numbers as `get_transforms`.

# %%
def lungs_only(x):
    """1 = right lung, 2 = left lung -> 1; everything else (incl. trachea) -> 0."""
    return ((x == 1) | (x == 2)).to(x.dtype)


def get_transforms(img_size, hu_min=-1000, hu_max=400, augment=False):
    keys = ["image", "label"]
    t = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Lambdad(keys="label", func=lungs_only),
        ScaleIntensityRanged(keys="image", a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=keys, spatial_size=(img_size, img_size), mode=("bilinear", "nearest")),
    ]
    if augment:
        t += [
            RandAffined(keys=keys, prob=0.5, rotate_range=0.15, scale_range=0.1, translate_range=img_size * 0.05,
                        mode=("bilinear", "nearest"), padding_mode="zeros"),
            RandGaussianNoised(keys="image", prob=0.2, std=0.02),
        ]
    t += [EnsureTyped(keys=keys)]
    return Compose(t)


def make_datalists(data_dir: Path):
    meta = pd.read_csv(data_dir / "metadata.csv")
    def rows(mask):
        return [{"image": str(data_dir / "images" / f"{c}.nii.gz"),
                 "label": str(data_dir / "labels" / f"{c}.nii.gz"), "case_id": c}
                for c in meta.loc[mask, "case_id"]]
    return rows(meta.subset <= 6), rows(meta.subset.isin([7, 8])), rows(meta.subset == 9)

# %% [markdown]
# ### Find and load a trained model (given)
# We use the model you trained (notebook 03/05) if it is in `WORK_DIR/models`, otherwise the teachers' model
# shipped with the dataset. The preprocessing settings always come from the model's own `config.json`.

# %%
def find_model_dirs():
    """All model folders (config.json + model.pt) we can see, most specific first."""
    roots = [os.environ.get("HI_MODEL_DIR"), WORK_DIR / "models", "/kaggle/input", DATA_DIR / "pretrained"]
    found = []
    for r in roots:
        if r and Path(r).exists():
            dirs = [p.parent for p in Path(r).rglob("config.json") if (p.parent / "model.pt").exists()]
            # within one place: the most recently saved model first
            found += sorted(dirs, key=lambda d: (d / "model.pt").stat().st_mtime, reverse=True)

    def is_binary_2d(d):   # the course contract: 2D, 1 input channel (HU window), 2 output channels
        k = json.loads((d / "config.json").read_text()).get("arch_kwargs", {})
        return (k.get("spatial_dims"), k.get("in_channels"), k.get("out_channels")) == (2, 1, 2)

    return [d for d in found if is_binary_2d(d)]


def load_model(model_dir, device="cpu"):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    net = getattr(monai.networks.nets, cfg["arch"])(**cfg["arch_kwargs"])
    net.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=device, weights_only=True))
    return net.to(device).eval(), cfg

# %%
model_dirs = find_model_dirs()
assert model_dirs, "No model found: finish notebook 03 or attach the dataset with its pretrained/ folder."
MODEL_DIR = model_dirs[0]
model, cfg = load_model(MODEL_DIR, device)
print("using:", MODEL_DIR)
print(json.dumps({k: cfg[k] for k in ("name", "preprocessing", "metrics")}, indent=1))

meta = pd.read_csv(DATA_DIR / "metadata.csv")
train_files, val_files, test_files = make_datalists(DATA_DIR)
eval_files = test_files[:N_EVAL]
print(f"{len(test_files)} test cases, we use {len(eval_files)} of them")

# %% [markdown]
# ## 1. The inference contract
#
# | | |
# |---|---|
# | **Input** | one 2D slice `hu[x, y]` in **Hounsfield units** (any size, int or float) + the pixel spacing `(sx, sy)` in mm |
# | **Output** | a binary `mask` with the **same shape** as the input, the lung probability map, and some measurements |
#
# Nothing else. The caller does not know the HU window, the network input size, or the architecture: all of that
# lives in the model's `config.json`. Let us look at the raw material we will have to deal with.

# %%
def read_raw(case):
    """Raw slice, spacing and full-resolution lung mask of one case (plain nibabel, readers come in section 3)."""
    img = nib.load(case["image"])
    hu = np.asanyarray(img.dataobj).astype(np.float32)
    lab = np.asanyarray(nib.load(case["label"]).dataobj)
    return hu, tuple(float(z) for z in img.header.get_zooms()[:2]), np.isin(lab, (1, 2)).astype(np.uint8)


def show_slice(ax, hu, mask=None, title="", color="red"):
    """Radiological view (.T), lung window; the optional mask is drawn as a contour."""
    ax.imshow(np.clip(hu, -1350, 150).T, cmap="gray", vmin=-1350, vmax=150)
    if mask is not None and mask.any():
        ax.contour(mask.T, levels=[0.5], colors=color, linewidths=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


case = test_files[0]
hu, spacing, gt_full = read_raw(case)
print(f"{case['case_id']}: shape {hu.shape}, dtype {hu.dtype}, spacing {spacing} mm, HU range [{hu.min():.0f}, {hu.max():.0f}]")
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
show_slice(ax[0], hu, title="raw HU slice")
show_slice(ax[1], hu, gt_full, "with the reference lung mask")
plt.show()

# %% [markdown]
# We will write four small functions and then compose them:
#
# ```
# hu [x, y] --preprocess--> tensor (1,1,H,W) --predict--> prob (H,W) --postprocess--> mask [x, y] --measure--> numbers
# ```
#
# ### ✏️ Exercise 1.1 - `preprocess`
# Training did: clip HU to the window `[hu_min, hu_max]`, scale to `[0, 1]`, resize **bilinearly** to
# `img_size`. We must do the same, with the **same MONAI transforms** (array versions, without the `d`
# of the dictionary ones), with the numbers read from `cfg["preprocessing"]`, never typed by hand.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `get_transforms` above has the two lines you need: `ScaleIntensityRanged(...)` and `Resized(...)`. The array
# versions have the same arguments, minus `keys`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `ScaleIntensityRange(a_min=p["hu_min"], a_max=p["hu_max"], b_min=0.0, b_max=1.0, clip=True)` and
# `Resize(spatial_size=tuple(p["img_size"]), mode="bilinear")`.
#
# </details>

# %%
def preprocess(hu, cfg):
    """HU slice [x, y] -> network input (1, 1, H, W)."""
    p = cfg["preprocessing"]
    x = torch.as_tensor(np.asarray(hu, dtype=np.float32))[None]  # channel first: (1, x, y)
#@solution
    x = ScaleIntensityRange(a_min=p["hu_min"], a_max=p["hu_max"], b_min=0.0, b_max=1.0, clip=True)(x)
    x = Resize(spatial_size=tuple(p["img_size"]), mode="bilinear")(x)
#@student
    #| x = ScaleIntensityRange(a_min=____, a_max=____, b_min=0.0, b_max=1.0, clip=____)(x)  # TODO: window from cfg, clip!
    #| x = Resize(spatial_size=____, mode=____)(x)  # TODO: network size from cfg, bilinear
#@end
    return torch.as_tensor(x, dtype=torch.float32)[None]  # add the batch dimension: (1, 1, H, W)

# %%
# Check 1.1
x = preprocess(hu, cfg)
size = tuple(cfg["preprocessing"]["img_size"])
assert tuple(x.shape) == (1, 1, *size), f"expected (1, 1, {size[0]}, {size[1]}), got {tuple(x.shape)}"
assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0, "values must be in [0, 1]: did you clip?"
padded = np.where(hu < cfg["preprocessing"]["hu_min"], -3024, hu)  # a different 'padding' value outside the window
assert torch.allclose(preprocess(padded, cfg), x), "values below hu_min must all give the same result (clip=True)"
print("✅ Exercise 1.1 passed")

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
show_slice(ax[0], hu, title=f"input: {hu.shape} px, HU")
ax[1].imshow(x[0, 0].T, vmin=0, vmax=1); ax[1].set_title(f"network input: {size} px, [0, 1]"); ax[1].axis("off")
ax[2].hist(hu.ravel(), bins=100, log=True); ax[2].axvspan(cfg["preprocessing"]["hu_min"], cfg["preprocessing"]["hu_max"], alpha=0.2)
ax[2].set_title("HU histogram (log) and the window"); plt.show()

# %% [markdown]
# > ⚠️ **Pitfall:** the window `[-1000, 400]` is a property of the **model**, not of the app. If you retrain with
# > another window and forget to change the app, the model silently receives wrong inputs and still returns a mask
# > (a wrong one). This is why the numbers live in `config.json`.
#
# ### ✏️ Exercise 1.2 - `predict`
# The network returns **logits** `(1, 2, H, W)`: channel 0 = background, channel 1 = lung. We want the
# probability of lung per pixel: a `softmax` over the *channel* dimension, then channel 1. Gradients are not
# needed at inference (`torch.no_grad`), saving memory and time.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Tensor dimensions are numbered from 0: batch = 0, channel = ?
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `torch.softmax(logits, dim=1)[0, 1]`: `[0]` is the first (only) item of the batch, `1` the lung channel.
#
# </details>

# %%
@torch.no_grad()
def predict(model, x):
    """Network input (1, 1, H, W) -> probability of 'lung' at network resolution (H, W)."""
    device = next(model.parameters()).device
    logits = model(x.to(device))
#@solution
    return torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
#@student
    #| return torch.softmax(logits, dim=____)[0, ____].cpu().numpy()  # TODO: which dim? which channel is the lung?
#@end

# %%
# Check 1.2
prob_net = predict(model, x)
assert prob_net.shape == size, f"expected a map of shape {size}, got {prob_net.shape}"
assert 0.0 <= prob_net.min() and prob_net.max() <= 1.0, "probabilities must be in [0, 1]"
assert prob_net.mean() < 0.5, "channel 1 must be the LUNG (a minority of the pixels), not the background"
print("✅ Exercise 1.2 passed")

# %% [markdown]
# ### ✏️ Exercise 1.3 - `postprocess`: back to the original size
# The probability map has the network size (say 256x256) but the user's image may be 512x512 or anything else.
# We resize the **probability map** back to the original shape (bilinear: smooth borders) and **then** threshold
# it. Optionally we clean the mask: keep the two largest connected components (the two lungs) and fill holes.
# The cleaning function is given.

# %%
def keep_largest_components(mask, n=2):
    """Keep the n biggest connected components (two lungs) and fill holes in them."""
    labelled, n_found = ndi.label(mask)
    if n_found == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, labelled, index=range(1, n_found + 1))
    keep = np.argsort(sizes)[::-1][:n] + 1
    out = np.isin(labelled, keep)
    return ndi.binary_fill_holes(out).astype(np.uint8)


def upsample(prob, original_shape):
    """Resize a probability map (H, W) back to the original image shape (bilinear)."""
#@solution
    p = Resize(spatial_size=tuple(original_shape), mode="bilinear")(torch.as_tensor(prob)[None])
#@student
    #| p = Resize(spatial_size=tuple(____), mode="bilinear")(torch.as_tensor(prob)[None])  # TODO: the target size
#@end
    return np.asarray(p[0], dtype=np.float32)


def to_mask(prob, threshold=0.5, clean=True):
    """Probability map -> binary mask; optionally clean it."""
#@solution
    mask = (prob > threshold).astype(np.uint8)
#@student
    #| mask = (prob > ____).astype(np.uint8)  # TODO: where is the decision boundary?
#@end
    return keep_largest_components(mask) if clean else mask


def postprocess(prob, original_shape, threshold=0.5, clean=True):
    """Network-resolution probability map -> binary mask with the shape of the ORIGINAL image."""
    return to_mask(upsample(prob, original_shape), threshold, clean)

# %% [markdown]
# <details>
# <summary>💡 Hint 1</summary>
#
# `upsample` needs the shape of the image the user gave us: that is the argument `original_shape`.
# The threshold is the argument `threshold` (0.5 by default: "more likely lung than not").
#
# </details>

# %%
# Check 1.3
mask = postprocess(prob_net, hu.shape)
assert mask.shape == hu.shape, f"the mask must have the shape of the input {hu.shape}, got {mask.shape}"
assert set(np.unique(mask)) <= {0, 1}
assert postprocess(prob_net, hu.shape, threshold=0.9, clean=False).sum() <= postprocess(prob_net, hu.shape, threshold=0.1, clean=False).sum(), \
    "a higher threshold can only remove pixels"
print("✅ Exercise 1.3 passed")

# %% [markdown]
# What does the cleaning do? Our model may be very good, so let us damage a mask on purpose: add specks outside the
# lungs and holes inside, then clean.

# %%
rng = np.random.default_rng(0)
damaged = mask.copy()
damaged[rng.random(mask.shape) < 0.0008] = 1                                  # specks anywhere
damaged[ndi.binary_erosion(mask, iterations=5) & (rng.random(mask.shape) < 0.003)] = 0   # holes inside
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, m, t in zip(ax, (mask, damaged, keep_largest_components(damaged)), ("model output", "damaged on purpose", "after cleaning")):
    a.imshow(m.T, vmin=0, vmax=1); a.set_title(f"{t}: {int(m.sum())} px, {ndi.label(m)[1]} components"); a.axis("off")
plt.show()

# %% [markdown]
# > ⚠️ **Pitfall:** "keep the 2 largest components" assumes **two lungs on the slice**. On slices above the
# > apices, below the diaphragm, or after a pneumonectomy there may be one lung or none, or the two lungs may touch
# > and form a single component. That is why the app lets the user switch post-processing off.
#
# ### 🔍 Explore
# * Resize the **mask** instead of the probability map: `Resize(..., mode="nearest")(mask_net)`. How do the borders look?
# * Plot the lung area for thresholds 0.1, 0.3, 0.5, 0.7, 0.9. How sensitive is the area to the threshold?
#
# ### ✏️ Exercise 1.4 - `measure`
# A radiologist does not want pixels, but **cm²**. The pixel area is `sx * sy` mm² and 1 cm² = 100 mm². (The
# right/left split is given: in our LPS arrays x grows towards the patient's **left**, so the component with the
# smaller x is the right lung.)
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Area of one pixel in mm²: `spacing[0] * spacing[1]`. Then convert to cm².
#
# </details>

# %%
def measure(mask, spacing):
    """Measurements from a binary mask [x, y] and the pixel spacing (sx, sy) in mm."""
#@solution
    pixel_area_cm2 = float(spacing[0]) * float(spacing[1]) / 100.0
#@student
    #| pixel_area_cm2 = ____  # TODO: area of ONE pixel in cm2
#@end
    out = {"area_cm2": float(mask.sum() * pixel_area_cm2), "area_pct_of_image": float(100 * mask.mean()),
           "n_components": 0, "right_cm2": None, "left_cm2": None}
    labelled, n = ndi.label(mask)
    out["n_components"] = int(n)
    if n >= 2:
        sizes = ndi.sum(mask, labelled, index=range(1, n + 1))
        two = np.argsort(sizes)[::-1][:2] + 1
        cx = {int(k): ndi.center_of_mass(mask, labelled, int(k))[0] for k in two}
        right, left = sorted(two, key=lambda k: cx[int(k)])
        out["right_cm2"] = float((labelled == right).sum() * pixel_area_cm2)
        out["left_cm2"] = float((labelled == left).sum() * pixel_area_cm2)
    return out

# %%
# Check 1.4: the reference mask must give the area stored in metadata.csv (pixels x spacing)
row = meta[meta.case_id == case["case_id"]].iloc[0]
area_mm2 = measure(gt_full, spacing)["area_cm2"] * 100
assert abs(area_mm2 - row.lung_area_mm2) / row.lung_area_mm2 < 0.02, \
    f"{area_mm2:.0f} mm2 vs {row.lung_area_mm2:.0f} mm2 in metadata.csv: check the unit conversion"
print(f"✅ Exercise 1.4 passed  ({area_mm2 / 100:.1f} cm2 of lung in the reference mask)")

# %% [markdown]
# ### ✏️ Exercise 1.5 - one function to rule them all: `segment_slice`
# Compose the pieces. This is the **only** function the web app will call. It returns a small dataclass.

# %%
@dataclass
class SegmentationResult:
    mask: np.ndarray                  # uint8 {0,1}, same shape as the input slice
    prob: np.ndarray                  # float32 in [0,1], same shape as the input slice
    prob_net: np.ndarray              # float32 in [0,1], network resolution (what the model really saw)
    measurements: dict = field(default_factory=dict)
    seconds: float = 0.0              # wall time of preprocess + predict + postprocess


def segment_slice(hu, spacing, model, cfg, threshold=0.5, clean=True):
    """THE function: raw HU slice [x, y] + spacing (mm) -> mask at the original resolution."""
    hu = np.asarray(hu)
    if hu.ndim != 2:
        raise ValueError(f"segment_slice expects a 2D slice [x, y], got an array of shape {hu.shape}")
    t0 = time.perf_counter()
#@solution
    x = preprocess(hu, cfg)
    prob_net = predict(model, x)
    prob = upsample(prob_net, hu.shape)
    mask = to_mask(prob, threshold, clean)
#@student
    #| x = ____(hu, cfg)            # TODO: HU slice -> network input
    #| prob_net = ____(model, x)    # TODO: network input -> probability map (network resolution)
    #| prob = upsample(prob_net, hu.shape)
    #| mask = ____(prob, threshold, clean)   # TODO: probability map -> mask
#@end
    seconds = time.perf_counter() - t0
    return SegmentationResult(mask=mask, prob=prob, prob_net=prob_net.astype(np.float32),
                              measurements=measure(mask, spacing), seconds=seconds)


def dice_score(a, b):
    """Dice between two binary masks (1.0 if both are empty)."""
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2 * (a & b).sum() / denom)

# %%
# Check 1.5
res = segment_slice(hu, spacing, model, cfg)
assert res.mask.shape == hu.shape and res.prob.shape == hu.shape, "outputs must have the shape of the input"
assert res.measurements["area_cm2"] > 0
d = dice_score(res.mask, gt_full)
if not FAST:
    assert d > 0.8, f"Dice {d:.3f} is suspiciously low: is there a bug in one of the steps?"
print(f"✅ Exercise 1.5 passed  (Dice vs the reference at full resolution: {d:.3f}, {res.seconds * 1000:.0f} ms)")

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
show_slice(ax[0], hu, title="input")
show_slice(ax[1], hu, res.mask, f"prediction: {res.measurements['area_cm2']:.1f} cm2")
ax[2].imshow(res.prob.T, vmin=0, vmax=1); ax[2].set_title("lung probability"); ax[2].axis("off")
plt.show()

# %% [markdown]
# ## 2. Does it behave like the training pipeline? (and what if it does not)
#
# Our `segment_slice` re-implements, for raw files, what `get_transforms` + the model did in notebooks 03-04. A
# re-implementation is a classic source of bugs. Last year's app resized the image with **nearest-neighbour**
# while the training used a different resampling, and fed the model values normalised another way: a real example
# of **training-serving skew**: the model that you validated is not the model that you serve. Nothing crashes,
# the masks just get worse and nobody notices.
#
# ### 2.1 Equivalence test
# For `N_EVAL` test cases we compute the Dice **at network resolution** in two ways:
# 1. "notebook 04 way": `get_transforms` (dictionary transforms, from file) -> model -> argmax;
# 2. "notebook 06 way": raw HU array -> `segment_slice` -> `prob_net > 0.5`.
#
# They must agree to numerical precision. We also report the Dice at the **original** resolution of the image (what
# the user of the app really gets).

# %%
IMG = cfg["preprocessing"]["img_size"][0]
eval_ds = Dataset(eval_files, get_transforms(IMG))
eval_rows, eval_data = [], []        # eval_data keeps (hu, spacing, gt_full, gt_net) for the next experiments

for f, item in zip(eval_files, eval_ds):
    gt_net = np.asarray(item["label"][0])                       # label at network resolution, from get_transforms
    with torch.no_grad():
        pred_04 = model(torch.as_tensor(item["image"])[None]).argmax(1)[0].numpy()   # notebook-04 way
    hu_i, sp_i, gt_full_i = read_raw(f)
    res_i = segment_slice(hu_i, sp_i, model, cfg, clean=False)  # notebook-06 way
    eval_rows.append({"case_id": f["case_id"],
                      "dice_04_net": dice_score(pred_04, gt_net),
                      "dice_06_net": dice_score(res_i.prob_net > 0.5, gt_net),
                      "dice_06_full": dice_score(res_i.mask, gt_full_i)})
    eval_data.append((hu_i, sp_i, gt_full_i, gt_net))

eval_df = pd.DataFrame(eval_rows)
print(eval_df.round(4).to_string(index=False))
max_diff = (eval_df.dice_04_net - eval_df.dice_06_net).abs().max()
print(f"max |Dice(04) - Dice(06)| = {max_diff:.2e}")
assert max_diff < 1e-4, "the two pipelines disagree: there is skew between training and serving!"
print("✅ same Dice as in training")

# %% [markdown]
# The two pipelines agree, because they share the same MONAI transforms and the same numbers. The Dice at the
# original resolution is a bit different: it is the same prediction, but judged on the full-resolution reference
# mask (in notebook 04 you evaluated at network resolution too).
#
# ### 2.2 ⚠️ Training-serving skew: how much does a small mistake cost?
# We now implement "slightly wrong" versions of `preprocess` and measure the damage with the same Dice.
# Each variant is a function `hu -> probability map at network resolution`.

# %%
P = cfg["preprocessing"]
NET_SIZE = tuple(P["img_size"])


def to_net_input(x01):
    """(1, x, y) in [0, 1] -> (1, 1, H, W) with the CORRECT bilinear resize (to isolate the other mistakes)."""
    x = Resize(spatial_size=NET_SIZE, mode="bilinear")(x01)
    return torch.as_tensor(x, dtype=torch.float32)[None]


def as_tensor(hu):
    return torch.as_tensor(np.asarray(hu, dtype=np.float32))[None]


def run_correct(hu):
    return predict(model, preprocess(hu, cfg))


def run_nearest(hu):          # mistake 1: nearest-neighbour resampling of the image (aliasing, blocky edges)
    x = ScaleIntensityRange(a_min=P["hu_min"], a_max=P["hu_max"], b_min=0.0, b_max=1.0, clip=True)(as_tensor(hu))
    x = torch.as_tensor(Resize(spatial_size=NET_SIZE, mode="nearest")(x), dtype=torch.float32)[None]
    return predict(model, x)


def run_minmax(hu):           # mistake 2: per-image min-max instead of the fixed HU window
    t = as_tensor(hu)
    return predict(model, to_net_input((t - t.min()) / (t.max() - t.min())))


def run_noclip(hu):           # mistake 3: scale, but forget clip=True (padding values, bone, metal...)
    x = ScaleIntensityRange(a_min=P["hu_min"], a_max=P["hu_max"], b_min=0.0, b_max=1.0, clip=False)(as_tensor(hu))
    return predict(model, to_net_input(x))


def run_transposed(hu):       # mistake 4: [row, col] vs [x, y] mix-up (what you get if you forget DICOM's transpose)
    return predict(model, preprocess(hu.T, cfg)).T


def run_png8(hu, lo=-1350, hi=150):   # mistake 5: "export as PNG with the lung window", then load it as [0, 1]
    png = np.round(255 * (np.clip(hu, lo, hi) - lo) / (hi - lo)).astype(np.uint8)
    return predict(model, to_net_input(torch.as_tensor(png.astype(np.float32) / 255.0)[None]))


variants = {"correct": run_correct, "nearest resize": run_nearest, "min-max per image": run_minmax,
            "no clip": run_noclip, "transposed": run_transposed, "8-bit PNG, lung window": run_png8}


def evaluate_variant(fn):
    """Mean Dice at network resolution over the evaluation cases."""
    return float(np.mean([dice_score(fn(hu_i) > 0.5, gt_net) for hu_i, _, _, gt_net in eval_data]))


def plot_skew(scores):
    names = sorted(scores, key=scores.get, reverse=True)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1))
    bars = ax.barh(names, [scores[n] for n in names], color=["tab:green" if n == "correct" else "tab:red" for n in names])
    ax.bar_label(bars, fmt="%.3f", padding=3); ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"mean Dice (n={len(eval_data)} test cases)"); ax.set_title("What a 'small' preprocessing mistake costs")
    plt.show()


skew_scores = {name: evaluate_variant(fn) for name, fn in variants.items()}
print({k: round(v, 3) for k, v in skew_scores.items()})
plot_skew(skew_scores)

# %%
# What did the model see? Same case, same model, different preprocessing.
hu_i, _, _, gt_net = eval_data[0]
fig, ax = plt.subplots(1, len(variants), figsize=(3 * len(variants), 3.2))
for a, (name, fn) in zip(ax, variants.items()):
    a.imshow(fn(hu_i).T, vmin=0, vmax=1); a.set_title(f"{name}\nDice {dice_score(fn(hu_i) > 0.5, gt_net):.2f}", fontsize=9); a.axis("off")
plt.show()

# %% [markdown] tags=["solution-only"]
# **Teacher note (dry run on synthetic data).** On the synthetic dataset (clean ellipses: no padding value, no bone,
# no texture) only `transposed` collapses; the other variants change the Dice by a few thousandths or even improve it.
# On real CT the drops are expected to be larger (`no clip` with -2000/-3024 padding, `min-max` with bone or metal,
# `nearest` with fine structures), but this was NOT verified on the real dataset: run this section once on the real
# data before the class and adapt the wording if needed.

# %% [markdown]
# The size of each drop depends on the data and on the model, but the lesson is general: **none of these variants
# crashes**, all produce a plausible-looking mask, and only a metric against the reference tells you that something
# is wrong. In production you have no reference.
#
# ### ✏️ Exercise 2.1 - implement a wrong variant yourself
# A colleague exports CT slices "with the mediastinal window" (centre 40 HU, width 400 HU) and the app scales them to
# `[0, 1]` before calling the model. Implement `run_mediastinal(hu)`: clip to that window, scale to `[0, 1]`, resize
# correctly, predict. Then add it to the chart.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# A window with centre `c` and width `w` spans `[c - w/2, c + w/2]`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `ScaleIntensityRange(a_min=-160, a_max=240, b_min=0.0, b_max=1.0, clip=True)(as_tensor(hu))`, then
# `predict(model, to_net_input(x))`.
#
# </details>

# %%
def run_mediastinal(hu):
    """Mediastinal window (centre 40, width 400) instead of the training window."""
#@solution
    x = ScaleIntensityRange(a_min=-160, a_max=240, b_min=0.0, b_max=1.0, clip=True)(as_tensor(hu))
#@student
    #| x = ScaleIntensityRange(a_min=____, a_max=____, b_min=0.0, b_max=1.0, clip=True)(as_tensor(hu))  # TODO: c -/+ w/2
#@end
    return predict(model, to_net_input(x))

# %%
# Check 2.1
p_med = run_mediastinal(eval_data[0][0])
assert p_med.shape == NET_SIZE and 0 <= p_med.min() and p_med.max() <= 1
skew_scores["mediastinal window"] = evaluate_variant(run_mediastinal)
plot_skew(skew_scores)
print(f"✅ Exercise 2.1 passed: the mediastinal window costs {skew_scores['correct'] - skew_scores['mediastinal window']:+.3f} Dice")

# %% [markdown]
# > 🧠 **Question:** how would you notice such a mistake in production, where there is no reference mask? Think of
# > at least three measures.

# %% [markdown] tags=["solution-only"]
# **Answer.**
# 1. **Share the code**: training and serving import the *same* `preprocess` (and read the numbers from the same
#    `config.json`). No re-implementation, no skew. This is why `inference.py` exists.
# 2. **Tests**: a test that feeds a fixed file through both pipelines and compares numbers (we did it in 2.1), and a
#    "golden file" test: a stored input with a stored expected mask, which fails if anything changes.
# 3. **Monitor inputs and outputs**: log simple statistics of what the model receives (min/max/mean of the
#    network input, image size, spacing) and what it returns (lung area, fraction of the image, number of components).
#    A shift in the distribution (e.g. all inputs suddenly in [0, 1] with a different mean) is an alarm.
# 4. **Sanity checks at runtime**: reject or warn on implausible inputs (8-bit range, unexpected orientation,
#    wrong modality), as our app does.
# 5. Version everything: model + config + code together (model card, git tag, container image).
# 6. Periodically evaluate on a small labelled sample from the real deployment setting.
#
# 📚 **Read more:** Sculley et al., "Hidden technical debt in machine learning systems" (NeurIPS 2015);
# Metrics Reloaded https://arxiv.org/abs/2206.01653 for choosing and reporting the metrics.

# %% [markdown]
# ### 🔍 Explore
# * Change the HU window of the *correct* pipeline by only 50 HU (`cfg["preprocessing"]["hu_max"] = 450` on a copy of
#   `cfg`). How much Dice do you lose? Is the model more sensitive to the lower or to the upper bound?
# * Add a `run_flipped` variant (`hu[::-1]`, a left-right flip) and compare with `run_transposed`.

# %% [markdown]
# ## 3. Readers: from a file to `(hu, spacing)`
#
# So far we read the files with plain `nibabel`. The app receives **uploads**: `.nii`, `.nii.gz` or `.dcm`.
# We need one function, `load_any(path) -> (hu[x, y], spacing)`.
#
# * **NIfTI**: `nib.load`; the array is already in our `[x, y]` convention and the pixel spacing is in the header.
#   A "2D" NIfTI is sometimes saved as `(x, y, 1)`: we squeeze it. A real volume needs a slice to be chosen.
# * **DICOM**: `pydicom.dcmread`. Three traps: (1) the stored values are **not** HU: `HU = stored * RescaleSlope +
#   RescaleIntercept`; (2) `pixel_array` is `[row, col] = [y, x]`: transpose it to our `[x, y]`; (3) `PixelSpacing`
#   is `[row spacing, column spacing] = [sy, sx]`, i.e. in the opposite order to our `(sx, sy)`.

# %%
class NotASliceError(ValueError):
    """The file holds a 3D volume with more than one slice: the caller must choose a slice."""


def pick_slice(data, slice_index=None):
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


def load_nifti(path):
    """Return (array, spacing_xy, affine). The array can be 2D or 3D [x, y, z]."""
    img = nib.load(str(path))
    zooms = img.header.get_zooms()
    return np.asanyarray(img.dataobj), (float(zooms[0]), float(zooms[1])), img.affine

# %% [markdown]
# ### ✏️ Exercise 3.1 - the DICOM reader
# Fill in the HU conversion and the axis order. First look at the header of a DICOM file.

# %%
dicom_files = sorted((DATA_DIR / "demo" / "dicom").glob("*/*.dcm"))
print(f"{len(dicom_files)} DICOM files in demo/dicom")
ds = pydicom.dcmread(str(dicom_files[len(dicom_files) // 2]))
for tag in ("Modality", "Rows", "Columns", "PixelSpacing", "SliceThickness", "RescaleSlope", "RescaleIntercept",
            "BitsStored", "PixelRepresentation", "WindowCenter", "WindowWidth"):
    print(f"{tag:22s} {getattr(ds, tag, '-')}")
print("\nThe header also contains the patient's name, ID, birth date...:", [t for t in ("PatientName", "PatientID", "PatientBirthDate") if t in ds])

# %% [markdown]
# > ⚠️ **Pitfall (privacy):** a DICOM header contains **personal data** (PHI). The course files are synthetic or
# > de-identified; real clinical DICOM must never be uploaded to a public web app or committed to a repository.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The slope and the intercept are already read into two variables. HU = stored * slope + intercept.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `hu = pixels * slope + intercept` and `return hu.T, (sx, sy)`.
#
# </details>

# %%
def load_dicom(path):
    """Return (hu[x, y], spacing_xy) from one DICOM file (a single slice)."""
    ds = pydicom.dcmread(str(path))
    if "PixelData" not in ds:
        raise ValueError("this DICOM file has no pixel data")
    pixels = ds.pixel_array.astype(np.float32)
    if pixels.ndim != 2:
        raise ValueError(f"expected a single-channel 2D DICOM slice, got pixel_array of shape {pixels.shape}")
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
#@solution
    hu = pixels * slope + intercept
    sy, sx = (float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0]))   # [row spacing, column spacing]
    return hu.T, (sx, sy)       # pixel_array is [row, col] = [y, x]; we want [x, y]
#@student
    #| hu = pixels * ____ + ____                                     # TODO: stored values -> HU
    #| sy, sx = (float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0]))   # [row spacing, column spacing]
    #| return ____, (sx, sy)       # TODO: pixel_array is [row, col] = [y, x]; we want [x, y]
#@end


def load_any(path, slice_index=None):
    """Read a NIfTI (.nii/.nii.gz) or DICOM (.dcm) slice. Returns (hu[x, y], spacing_xy)."""
    name = Path(path).name.lower()
    if name.endswith((".nii", ".nii.gz")):
        data, spacing, _ = load_nifti(path)
        return pick_slice(data, slice_index), spacing
    if name.endswith((".dcm", ".dicom")):
        return load_dicom(path)
    raise ValueError(f"unsupported file type: {Path(path).name} (use .nii, .nii.gz or .dcm)")

# %%
# Check 3.1
hu_d, sp_d = load_any(dicom_files[len(dicom_files) // 2])
assert hu_d.shape == (int(ds.Columns), int(ds.Rows)), f"expected [x, y] = ({ds.Columns}, {ds.Rows}), got {hu_d.shape}: transpose?"
assert hu_d.min() < -200, "a chest CT must contain air (< -200 HU): did you apply slope and intercept?"
assert sp_d == (float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0])), "spacing must be (sx, sy) = (column spacing, row spacing)"
print(f"✅ Exercise 3.1 passed: {hu_d.shape}, spacing {sp_d} mm, HU range [{hu_d.min():.0f}, {hu_d.max():.0f}]")

# %% [markdown]
# Does the orientation agree with the NIfTI volume of the same scan? The course ships the scan both as DICOM and
# as NIfTI in `demo/`: for every DICOM slice we look for the NIfTI slice that looks identical, with and without the
# transpose. A match with difference ~0 means that our reader gives the same array as the NIfTI one.

# %%
vol_files = sorted((DATA_DIR / "demo" / "volume").glob("*_image.nii.gz"))
if vol_files:
    vol, vol_sp, _ = load_nifti(vol_files[0])
    def best_match(a):
        if a.shape != vol.shape[:2]:
            return None
        diffs = np.abs(vol.astype(np.float32) - a[..., None]).mean(axis=(0, 1))
        return int(diffs.argmin()), float(diffs.min())
    print("our reader      -> best NIfTI slice, mean |difference| (HU):", best_match(hu_d))
    print("without the .T  -> ", best_match(hu_d.T) if hu_d.shape[0] == hu_d.shape[1] else "(non-square image: shapes would not even match)")

# %% [markdown]
# ### End to end on real files
# A NIfTI slice from `images/` and one DICOM slice, both through the **same** function. Different sizes and spacings:
# the output always lives in the grid of the input.

# %%
def segment_file(path, **kw):
    hu_f, sp_f = load_any(path)
    return hu_f, sp_f, segment_slice(hu_f, sp_f, model, cfg, **kw)


files_demo = [Path(test_files[1]["image"])] + dicom_files[len(dicom_files) // 2: len(dicom_files) // 2 + 1]
fig, ax = plt.subplots(1, len(files_demo), figsize=(5 * len(files_demo), 5))
for a, path in zip(np.atleast_1d(ax), files_demo):
    hu_f, sp_f, r = segment_file(path)
    show_slice(a, hu_f, r.mask, f"{path.name}\n{hu_f.shape[0]}x{hu_f.shape[1]} px, {sp_f[0]:.2f} mm: lung area {r.measurements['area_cm2']:.1f} cm2")
plt.show()

# %% [markdown]
# ### 🔍 Explore
# * Feed `segment_slice` a deliberately wrong spacing (`(1, 1)`): the mask is the same but the area is not. Which
#   one would a radiologist notice first?
# * Load the same DICOM *without* `.T` and look at the result.
#
# ## 4. Speed and size: will it run on a free CPU host?
# A free web host gives you a couple of CPU cores and limited memory, shared with others. We measure **latency**
# (one prediction), **size** (parameters, file) and how they depend on the number of threads and on `img_size`.
#
# ### ✏️ Exercise 4.1 - a tiny benchmark
# Run a function several times and report the **median** (robust to hiccups). The first call(s) are slower
# (memory allocation, lazy initialisation): do a warm-up and keep the cold start in mind.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `time.perf_counter()` before and after the call; their difference is in seconds.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `times.append(time.perf_counter() - t0)` and `float(np.median(times)) * 1000`.
#
# </details>

# %%
def benchmark(fn, n_runs=N_RUNS, n_warmup=2):
    """Median wall time of fn() in milliseconds, after a short warm-up."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
#@solution
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000
#@student
        #| times.append(____)   # TODO: elapsed seconds since t0
    #| return ____              # TODO: the median of the times, in milliseconds
#@end

# %%
# Check 4.1
ms = benchmark(lambda: time.sleep(0.01), n_runs=5, n_warmup=1)
assert 9 < ms < 50, f"sleeping 10 ms should be measured as ~10 ms, got {ms:.1f}"
print("✅ Exercise 4.1 passed")

# %%
hu_b, sp_b, _ = read_raw(test_files[0])
n_params = sum(p.numel() for p in model.parameters())
size_mb = (MODEL_DIR / "model.pt").stat().st_size / 1e6
latency_ms = benchmark(lambda: segment_slice(hu_b, sp_b, model, cfg))
print(f"parameters: {n_params:,}   model.pt: {size_mb:.1f} MB   input {hu_b.shape}, network {NET_SIZE}")
print(f"latency of segment_slice on CPU ({torch.get_num_threads()} threads): median {latency_ms:.0f} ms over {N_RUNS} runs")

# where does the time go?
x_b = preprocess(hu_b, cfg)
prob_b = predict(model, x_b)
parts = {"preprocess": benchmark(lambda: preprocess(hu_b, cfg)), "predict (network)": benchmark(lambda: predict(model, x_b)),
         "postprocess": benchmark(lambda: postprocess(prob_b, hu_b.shape))}
print({k: f"{v:.0f} ms" for k, v in parts.items()})

# %%
# Effect of the number of threads (a free host may give you 1-2 cores)
n_default = torch.get_num_threads()
thread_ms = {}
for n in sorted({1, 2, n_default}):
    torch.set_num_threads(n)
    thread_ms[n] = benchmark(lambda: predict(model, x_b))
torch.set_num_threads(n_default)

# Effect of the network input size (the model is fully convolutional: it accepts other sizes, but was trained for one)
sizes = (32, 64, 128) if FAST else (128, 192, 256, 384)
size_rows = []
for s in sizes:
    cfg_s = {**cfg, "preprocessing": {**P, "img_size": [s, s]}}
    lat = benchmark(lambda: segment_slice(hu_b, sp_b, model, cfg_s))
    dsc = np.mean([dice_score(segment_slice(h, sp, model, cfg_s, clean=False).mask, g) for h, sp, g, _ in eval_data[:5]])
    size_rows.append({"img_size": s, "latency_ms": round(lat), "dice_full_res": round(float(dsc), 3)})

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].bar([str(n) for n in thread_ms], list(thread_ms.values())); ax[0].set_xlabel("torch threads"); ax[0].set_ylabel("ms per prediction")
ax[1].plot([r["img_size"] for r in size_rows], [r["latency_ms"] for r in size_rows], "o-"); ax[1].set_xlabel("img_size"); ax[1].set_ylabel("ms")
ax[1].set_title(f"model trained at {IMG}: Dice " + ", ".join(f"{r['img_size']}: {r['dice_full_res']}" for r in size_rows), fontsize=8)
plt.tight_layout(); plt.show()

# %%
# Batch vs single: one image at a time is what a web app does; a batch amortises the overhead
batch_ms = {}
for b in ((1, 4) if FAST else (1, 4, 8, 16)):
    xb = x_b.repeat(b, 1, 1, 1)
    with torch.no_grad():
        batch_ms[b] = benchmark(lambda: model(xb)) / b
print("ms per image in a batch of b:", {b: round(v, 1) for b, v in batch_ms.items()})

# %% [markdown]
# > 🧠 **Question:** is this fast and small enough for a free CPU web host? What about 20 students uploading at the
# > same time? Which of the numbers above would you worry about first: latency, file size, or memory? What changes if
# > the model becomes 10x larger?

# %% [markdown] tags=["solution-only"]
# **Answer.** For a single 2D slice, a few hundred milliseconds on 1-2 cores and a model of a few MB are fine: the
# user waits less than a second. What usually bites on a free host is not the speed of one prediction but
# (1) **memory**: the Python process with PyTorch + MONAI already takes a few hundred MB before any image is
# loaded (hosts have a hard limit and kill the app if exceeded; check the host's documentation); (2) the **cold
# start**: the app may be put to sleep and the first request pays for imports + model loading, hence
# `st.cache_resource`; (3) **concurrency**: requests share the same cores, so 20 simultaneous users means a
# queue (latency x 20 in the worst case); (4) the **installation size**: GPU wheels of torch are huge, so we use
# CPU wheels. A 10x larger model would first hit the memory limit and the latency; options: smaller
# `img_size`, fewer channels, ONNX Runtime, quantisation, or a paid host.

# %% [markdown]
# ## 5. Export formats: how do you ship a model?
# The same trained network can be stored in several ways. We use the first one; the others are good to know.
#
# | Format | What is stored | Needs the Python class? | Use |
# |---|---|---|---|
# | `state_dict` + `config.json` | only the weights (tensors) + the recipe | yes (MONAI) | **our choice**: small, safe, transparent |
# | whole model pickled (`torch.save(model)`) | weights + a pickled Python object | yes, same code layout | avoid |
# | TorchScript | graph + weights | no (PyTorch only) | deploy in C++/mobile, no Python class needed |
# | ONNX | graph + weights, open standard | no | other runtimes (ONNX Runtime), other languages |
# | MONAI Bundle | weights + config + metadata + scripts | no | the standard "model package" of the MONAI ecosystem |
#
# ### 5.1 `state_dict` and why `weights_only=True`
# `torch.load` uses Python's `pickle`, and **unpickling can run arbitrary code**: loading a model file from a stranger
# is like running a program from a stranger. `weights_only=True` accepts only tensors and plain containers. A harmless
# demonstration: we craft a "model file" that prints a message when loaded.

# %%
class Prank:
    def __reduce__(self):                        # pickle calls this when LOADING: it can name any function...
        return (print, ("!!! code from inside the file ran while loading it !!!",))


prank_path = WORK_DIR / "prank.pt"
torch.save({"weights": Prank()}, prank_path)
print("weights_only=False ->"); _ = torch.load(prank_path, weights_only=False)
try:
    torch.load(prank_path, weights_only=True)
except Exception as err:
    print("weights_only=True  -> refused:", type(err).__name__)
prank_path.unlink()

# %% [markdown]
# > ⚠️ **Pitfall:** the `print` could have been `os.system(...)`. Never load pickled files from untrusted sources
# > (also not `.pkl`, joblib files, or old `torch.save(model)` files). In our web app users upload *images*, never
# > models, and the models come from our own repository.
#
# ### ✏️ Exercise 5.1 - TorchScript
# `torch.jit.trace` runs the model once on an example input and records the operations. The saved file can be
# loaded **without** the Python class (and without MONAI). Trace, save, reload and check that the outputs are the same.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `trace` needs the model and an example input with the right shape: `(1, 1, H, W)`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `torch.jit.trace(model, example)`
#
# </details>

# %%
example = torch.rand(1, 1, *NET_SIZE)
ts_path = WORK_DIR / "model_traced.pt"
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                  # tracing warns about data-dependent control flow
#@solution
    traced = torch.jit.trace(model, example)
#@student
    #| traced = torch.jit.trace(____, ____)  # TODO: the model and an example input
#@end
traced.save(str(ts_path))
reloaded = torch.jit.load(str(ts_path), map_location="cpu")   # no MONAI class involved

# %%
# Check 5.1
with torch.no_grad():
    diff = (model(example) - reloaded(example)).abs().max().item()
assert diff < 1e-4, f"the TorchScript model gives different outputs (max diff {diff})"
print(f"✅ Exercise 5.1 passed (max difference {diff:.1e}); TorchScript file {ts_path.stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# > ⚠️ **Pitfall:** a traced model has the *control flow of the example run* frozen in; and TorchScript is in
# > maintenance mode in recent PyTorch releases (check the PyTorch docs for the current recommendation, e.g.
# > `torch.export`). For a plain convolutional U-Net, tracing is fine. Always compare the outputs, as we did.
#
# ### 🚀 Challenge (additional): ONNX and ONNX Runtime
# ONNX is an open format; `onnxruntime` runs it on CPU, often faster than eager PyTorch, with a small installation.
# Export with a dynamic batch dimension, run it, compare with PyTorch and time it. The cell is guarded: if `onnx` or
# `onnxruntime` is not installed (`pip install onnx onnxruntime`), it just says so.

# %%
try:
    import onnxruntime as ort
    onnx_path = WORK_DIR / "model.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, example, str(onnx_path), input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_onnx = sess.run(None, {"image": example.numpy()})[0]
    with torch.no_grad():
        diff = np.abs(out_onnx - model(example).numpy()).max()
    t_onnx = benchmark(lambda: sess.run(None, {"image": x_b.numpy()}))
    t_torch = benchmark(lambda: predict(model, x_b))
    print(f"ONNX vs PyTorch max difference {diff:.1e}; network time: ONNX Runtime {t_onnx:.1f} ms, PyTorch {t_torch:.1f} ms")
except Exception as err:
    print("ONNX part skipped:", type(err).__name__, err)

# %% [markdown]
# ### 🌶️ Going further (optional): MONAI Bundle
# The MONAI **Bundle** is the standard package for models: weights, a JSON/YAML config describing the network and the
# pre/post-processing, metadata, and inference scripts that run from the command line. It is what the MONAI Model Zoo
# distributes. Our `model/` folder (`model.pt` + `config.json`) is a minimal, hand-made cousin.
# * Docs: https://docs.monai.io/en/stable/bundle.html
# * Model Zoo: https://github.com/Project-MONAI/model-zoo and tutorials: https://github.com/Project-MONAI/tutorials
#
# Idea: convert our model folder into a bundle and run it with `python -m monai.bundle run`.
#
# ## 6. The model card
# A model card is the "package leaflet" of a model: what it is for, what it is **not** for, what it was trained
# on, how well it works and where it fails. Hospitals, regulators and your future self will ask for it
# (Mitchell et al., "Model cards for model reporting", https://arxiv.org/abs/1810.03993).
#
# The facts we can compute (parameters, metrics, latency) are filled in automatically. **You** write the judgement.
#
# ### ✏️ Exercise 6.1 - write your model card
# Replace each `"____"` with a few sentences (a string, bullets allowed). Think of your real model and data:
# LUNA16 comes from the LIDC-IDRI collection (public, mainly North American/European adults, various scanners and
# slice thicknesses); the lung masks are produced by an algorithm (**silver standard**), we use **one slice per
# patient**; the metrics below were computed on **n** test cases.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Intended use: *research and teaching*, segmenting the lungs on a single axial chest-CT slice in HU. NOT intended:
# diagnosis, screening, treatment decisions, other body parts or modalities (X-ray, MRI), children, 3D volumes.
#
# </details>
#
# <details>
# <summary>💡 Hint 2</summary>
#
# Failure modes to consider: lungs with large consolidations, effusions, pneumothorax, atelectasis; very different
# reconstruction kernels or thick slices; slices at the apex or above the diaphragm; metal artefacts; implanted devices;
# images not in HU or flipped/rotated; paediatric patients.
#
# </details>

# %%
card = {
#@solution
    "intended_use": "Research and teaching: automatic segmentation of the lungs on a single axial chest-CT slice (Hounsfield units, "
                    "adults) to demonstrate a complete deployment pipeline from notebook to web app.",
    "not_intended_use": "NOT a medical device. Not for diagnosis, screening, treatment planning or any clinical decision; not for "
                        "patients' data on a public service; not for other modalities (X-ray, MRI), children, or 3D volumes.",
    "limitations": "- Trained on LUNA16 (LIDC-IDRI): public data from a limited set of centres, adults, mixture of scanners and "
                   "reconstruction kernels.\n- Reference masks are silver standard (made by an algorithm), so metrics measure "
                   "agreement with that algorithm, not anatomical truth.\n- One slice per patient: small and not representative of "
                   "all levels of the chest.",
    "failure_modes": "- Lungs with dense consolidation, effusion, pneumothorax or atelectasis (masks may miss them).\n- Slices at the "
                     "apex/diaphragm, where one lung or none is visible (post-processing assumes two).\n- Inputs not in HU, "
                     "flipped/rotated, or with metal artefacts.",
    "regulatory_ethics": "Educational prototype without clinical validation: it is not CE-marked or FDA-cleared. Any clinical use "
                         "would need a validation on local data, a risk analysis and regulatory approval (EU MDR / AI Act). "
                         "Do not upload identifiable patient data; DICOM headers contain PHI.",
#@student
    #| "intended_use": "____",        # what is it for?
    #| "not_intended_use": "____",    # what is it NOT for? (say that it is not a medical device)
    #| "limitations": "____",         # training data: source, licence/size, silver-standard labels, one slice per patient, population
    #| "failure_modes": "____",       # when do you expect it to fail?
    #| "regulatory_ethics": "____",   # regulation, privacy, responsible use
#@end
}

# %%
# Check 6.1
for key, text in card.items():
    assert "____" not in text and len(text) > 40, f"fill in '{key}' (a few meaningful sentences)"
assert "medical device" in (card["intended_use"] + card["not_intended_use"] + card["regulatory_ethics"]).lower(), \
    "say clearly that this is not a medical device"
print("✅ Exercise 6.1 passed")

# %% [markdown]
# Now we assemble the card. The test metrics come from `test_results.csv` (notebook 04) if you saved one next to your
# models or in `WORK_DIR`; otherwise from the evaluation of section 2. We write the card and an **updated
# `config.json`** to the model folder that goes into the starter kit.

# %%
results_csv = next((p for p in (WORK_DIR / "test_results.csv", MODEL_DIR / "test_results.csv", WORK_DIR / "outputs" / "test_results.csv") if p.exists()), None)
if results_csv is not None:
    tr = pd.read_csv(results_csv)
    dice_col = next(c for c in tr.columns if "dice" in c.lower())
    test_dice, n_test, source = float(tr[dice_col].mean()), len(tr), results_csv.name
else:
    test_dice, n_test, source = float(eval_df.dice_06_full.mean()), len(eval_df), "notebook 06, section 2 (full resolution)"
print(f"test Dice {test_dice:.4f} on n={n_test} cases (from {source})")

# the folder that goes to the app: a copy of the model folder, so we never touch the original
OUT_MODEL_DIR = WORK_DIR / "models" / cfg["name"]
if OUT_MODEL_DIR.resolve() != Path(MODEL_DIR).resolve():
    shutil.copytree(MODEL_DIR, OUT_MODEL_DIR, dirs_exist_ok=True)

cfg_out = {**cfg, "metrics": {**cfg.get("metrics", {}), "test_dice": round(test_dice, 4), "n_test": n_test},
           "deployment": {"cpu_latency_ms": round(latency_ms), "n_params": n_params, "model_mb": round(size_mb, 1)}}
(OUT_MODEL_DIR / "config.json").write_text(json.dumps(cfg_out, indent=2))

CARD_TEMPLATE = """# Model card: {name}

*Generated on {date} by notebook 06. {warning}*

## Model
* Architecture: `{arch}` (MONAI), {n_params:,} parameters, `model.pt` {model_mb:.1f} MB
* Input: one axial chest-CT slice in Hounsfield units, window [{hu_min}, {hu_max}] HU -> [0, 1], resized to {img_size} px
* Output: probability of "lung" per pixel; binary mask at the original resolution (threshold 0.5, optional cleaning)
* Trained by: {trained_by}

## Intended use
{intended_use}

## NOT intended use
{not_intended_use}

## Training data
LUNA16 lung slices (one slice per patient), derived from the public LIDC-IDRI collection. Check the licence of the
original data before redistributing it. Split by patient/subset: train subsets 0-6, validation 7-8, test 9.

## Performance
* Validation Dice (best epoch): {val_dice}
* Test Dice: {test_dice:.3f} on n = {n_test} slices (source: {source}). One slice per patient: wide confidence intervals; report the spread, not only the mean.
* CPU latency (one slice, median): {latency} ms

## Limitations
{limitations}

## Known failure modes
{failure_modes}

## Ethical and regulatory considerations
{regulatory_ethics}
"""
card_md = CARD_TEMPLATE.format(
    name=cfg["name"], date=time.strftime("%Y-%m-%d"), warning="Research/education only - not a medical device.",
    arch=cfg["arch"], n_params=n_params, model_mb=size_mb, hu_min=P["hu_min"], hu_max=P["hu_max"], img_size=NET_SIZE,
    trained_by=cfg.get("trained_by", "n/a"), val_dice=cfg.get("metrics", {}).get("val_dice", "n/a"),
    test_dice=test_dice, n_test=n_test, source=source, latency=round(latency_ms), **card)
(OUT_MODEL_DIR / "MODEL_CARD.md").write_text(card_md)
print(card_md[:1500], "\n[...]\nwritten to", OUT_MODEL_DIR / "MODEL_CARD.md")

# %% [markdown]
# ## 7. A tiny test suite
# Tests are the cheapest protection against skew and regressions. Three properties that must always hold for
# `segment_slice`:
# 1. the **shape** of the mask equals the shape of the input (for any input size);
# 2. an image with **no lungs** (uniform soft tissue) gives a small mask. (Why not an all-air image? Air *is* what
#    lungs look like: a uniform -1000 HU image is a legitimate "all lung" input, any answer is acceptable.)
# 3. **determinism**: the same input gives the same output.
#
# Tomorrow these become `pytest` tests in the app repository.
#
# ### ✏️ Exercise 7.1 - the determinism test
# <details>
# <summary>💡 Hint 1</summary>
#
# Call `segment_slice` twice on the same input and compare masks with `np.array_equal` and probabilities with `np.allclose`.
#
# </details>

# %%
def test_shape_is_preserved():
    for shape in [(96, 80), (64, 64), (101, 57)]:
        r = segment_slice(np.full(shape, 40.0, dtype=np.float32), (1.0, 1.0), model, cfg)
        assert r.mask.shape == shape and r.prob.shape == shape


def test_no_lungs_gives_a_small_mask():
    r = segment_slice(np.full((128, 128), 40.0, dtype=np.float32), (1.0, 1.0), model, cfg)   # soft tissue only
    assert r.mask.mean() < 0.5


def test_is_deterministic():
    hu_t = np.random.default_rng(0).normal(-300, 300, (80, 96)).astype(np.float32)
#@solution
    a = segment_slice(hu_t, (0.7, 0.7), model, cfg)
    b = segment_slice(hu_t, (0.7, 0.7), model, cfg)
    assert np.array_equal(a.mask, b.mask) and np.allclose(a.prob, b.prob)
#@student
    #| a = ____   # TODO: first run
    #| b = ____   # TODO: second run, same input
    #| assert ____ and ____   # TODO: same mask, same probabilities
#@end


for t in (test_shape_is_preserved, test_no_lungs_gives_a_small_mask, test_is_deterministic):
    t()
    print("✅", t.__name__)

# %% [markdown]
# ## 8. Hand-over: `inference.py` and the starter kit
# We now freeze everything we wrote into **one importable file**, `inference.py`. The cell below writes the
# reference version (the same one the app uses: single source of truth). It is long: you do not have to read it
# now, but compare it with your functions afterwards: it is the same code plus the readers. It depends only on
# `numpy`, `torch`, `monai`, `nibabel`, `pydicom`, `scipy`: no Streamlit, no matplotlib.

# %%
KIT_DIR = WORK_DIR / "app_starter_kit"
shutil.rmtree(KIT_DIR, ignore_errors=True)
KIT_DIR.mkdir(parents=True)
INFERENCE_PY = r'''"""Lung segmentation inference: from a raw CT slice (HU) to a mask, on CPU.

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
'''
(KIT_DIR / "inference.py").write_text(INFERENCE_PY)
print("written", KIT_DIR / "inference.py", f"({len(INFERENCE_PY.splitlines())} lines)")

# %%
# It must be importable and give the same result as the functions of this notebook
import importlib
sys.dont_write_bytecode = True        # do not leave __pycache__ in the kit
sys.path.insert(0, str(KIT_DIR))
importlib.invalidate_caches()
import inference as inf

hu_t, sp_t, gt_t = read_raw(test_files[2])
r_nb, r_mod = segment_slice(hu_t, sp_t, model, cfg), inf.segment_slice(hu_t, sp_t, *inf.load_model(OUT_MODEL_DIR))
assert np.array_equal(r_nb.mask, r_mod.mask) and np.allclose(r_nb.prob, r_mod.prob, atol=1e-6)
assert inf.load_any(test_files[2]["image"])[0].shape == hu_t.shape
print("✅ inference.py is importable and identical in behaviour to the notebook functions")

# %% [markdown]
# Now the samples, a small test file, and the zip. `samples/` gets a few **test-split** slices with their labels
# (`name.nii.gz` + `name_label.nii.gz`, the app uses the label automatically) and one DICOM slice.

# %%
TEST_FILE = r'''"""Starter tests for inference.py. Run:  python -m pytest -q   (from the app folder)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference as inf  # noqa: E402


@pytest.fixture(scope="module")
def model_and_cfg():
    env = os.environ.get("HI_MODEL_DIR")
    dirs = [Path(env)] if env else inf.find_model_dirs(Path(__file__).resolve().parents[1] / "models")
    if not dirs:
        pytest.skip("no model found: unzip app_starter_kit.zip or set HI_MODEL_DIR")
    return inf.load_model(dirs[0])


def test_shape_is_preserved(model_and_cfg):
    model, cfg = model_and_cfg
    for shape in [(96, 80), (64, 64), (101, 57)]:
        r = inf.segment_slice(np.full(shape, 40.0, dtype=np.float32), (1.0, 1.0), model, cfg)
        assert r.mask.shape == shape and r.prob.shape == shape


def test_no_lungs_gives_a_small_mask(model_and_cfg):
    model, cfg = model_and_cfg
    r = inf.segment_slice(np.full((128, 128), 40.0, dtype=np.float32), (1.0, 1.0), model, cfg)
    assert r.mask.mean() < 0.5


def test_is_deterministic(model_and_cfg):
    model, cfg = model_and_cfg
    hu = np.random.default_rng(0).normal(-300, 300, (80, 96)).astype(np.float32)
    a, b = inf.segment_slice(hu, (0.7, 0.7), model, cfg), inf.segment_slice(hu, (0.7, 0.7), model, cfg)
    assert np.array_equal(a.mask, b.mask)


# TODO (day 5): add a test of your own, e.g. preprocess() output in [0, 1], or measure() on a mask of known area.
'''
(KIT_DIR / "tests").mkdir()
(KIT_DIR / "tests" / "test_inference.py").write_text(TEST_FILE)

(KIT_DIR / "samples").mkdir()
n_samples = 3 if FAST else 5
for f in test_files[:n_samples]:
    shutil.copy(f["image"], KIT_DIR / "samples" / f"{f['case_id']}.nii.gz")
    shutil.copy(f["label"], KIT_DIR / "samples" / f"{f['case_id']}_label.nii.gz")
if dicom_files:
    shutil.copy(dicom_files[len(dicom_files) // 2], KIT_DIR / "samples" / "dicom_example.dcm")
shutil.copytree(OUT_MODEL_DIR, KIT_DIR / "models" / cfg["name"])

zip_path = Path(shutil.make_archive(str(WORK_DIR / "app_starter_kit"), "zip", root_dir=KIT_DIR))
with zipfile.ZipFile(zip_path) as z:
    for info in z.infolist():
        print(f"{info.file_size / 1e6:8.2f} MB  {info.filename}")
print(f"\n{zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB): download it and bring it to VS Code tomorrow")

# %% [markdown]
# > 📚 **Tomorrow:** unzip `app_starter_kit.zip` in the app folder. You get `inference.py`, `models/<name>/`
# > (weights, config, model card), `samples/` and `tests/`.
#
# ### Run the final check from a clean directory
# A last test from a *fresh* process, as the app will do: only files from the zip, nothing from this notebook.

# %%
check = subprocess.run([sys.executable, "-c", (
    "import sys; sys.path.insert(0, sys.argv[1]); import inference as inf, glob; "
    "m, c = inf.load_model(glob.glob(sys.argv[1] + '/models/*')[0]); "
    "hu, sp = inf.load_any(sorted(glob.glob(sys.argv[1] + '/samples/*.nii.gz'))[0]); r = inf.segment_slice(hu, sp, m, c); "
    "print('fresh process OK:', r.mask.shape, round(r.measurements['area_cm2'], 1), 'cm2')"), str(KIT_DIR)],
    capture_output=True, text=True)
print(check.stdout or check.stderr[-800:])
assert check.returncode == 0

# %% [markdown]
# ## 9. 🚀 / 🌶️ Going further
#
# ### 🚀 Challenge (additional): a 3D volume, slice by slice
# `segment_slice` works on one slice, so a volume is just a loop. Run it on `demo/volume`, plot the lung area per slice
# (a "volume curve") and show a montage. This is a bridge to the optional 3D track.

# %%
if vol_files:
    vol_hu, vol_sp, _ = load_nifti(vol_files[0])
    nz = vol_hu.shape[2]
    step = max(1, nz // (6 if FAST else 40))
    zs = list(range(0, nz, step))
    results = {z: segment_slice(pick_slice(vol_hu, z), vol_sp, model, cfg) for z in zs}
    areas = [results[z].measurements["area_cm2"] for z in zs]
    pick = zs[:: max(1, len(zs) // 8)][:8]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for a, z in zip(axes.ravel(), pick):
        show_slice(a, pick_slice(vol_hu, z), results[z].mask, f"z={z}: {results[z].measurements['area_cm2']:.0f} cm2")
    for a in axes.ravel()[len(pick):]:
        a.axis("off")
    plt.tight_layout(); plt.show()
    plt.figure(figsize=(6, 3)); plt.plot(zs, areas, "o-"); plt.xlabel("slice index z"); plt.ylabel("lung area (cm2)")
    plt.title("lung area per slice"); plt.show()

# %% [markdown]
# > ⚠️ **Pitfall:** the model was trained on slices that contain lung. On slices of the neck or the abdomen it
# > has never seen the right answer: look at the first and last slices of the montage. Real thinking about the **input
# > domain** (what slices will users upload?) belongs in the model card and in the app (a warning when the area is tiny).
#
# ### 🚀 Challenge (additional): dynamic quantisation
# `torch.ao.quantization.quantize_dynamic` stores weights as int8, which shrinks and speeds up models dominated by
# `nn.Linear` layers (transformers, LSTMs). Try it on our U-Net and compare size and latency.

# %%
from torch.ao.quantization import quantize_dynamic

q_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(q_model.state_dict(), WORK_DIR / "model_dynq.pt")
print("float32:", round(size_mb, 2), "MB   quantised:", round((WORK_DIR / "model_dynq.pt").stat().st_size / 1e6, 2), "MB")
print("latency:", round(benchmark(lambda: predict(model, x_b)), 1), "ms ->", round(benchmark(lambda: predict(q_model, x_b)), 1), "ms")

# %% [markdown]
# A U-Net is made of **convolutions**, not linear layers, so *dynamic* quantisation changes almost nothing. To
# quantise convolutions you need *static* quantisation (with calibration data) or quantisation-aware training:
# a bigger project, and a lesson in "an optimisation is not universal". Pointer:
# https://pytorch.org/docs/stable/quantization.html
#
# ### 🌶️ Going further (optional): the MONAI way with `Invertd`
# MONAI can **invert** the preprocessing for you: the dictionary transforms remember what they did (resize,
# padding...) and `Invertd` maps a prediction back to the grid of the original image. It is the idiomatic version of
# our `upsample`. Try it, and compare the mask with ours. (Our function also reads DICOM and works from an array: why
# do we prefer our version in the app?)

# %%
pre = Compose([LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"),
               ScaleIntensityRanged(keys="image", a_min=P["hu_min"], a_max=P["hu_max"], b_min=0.0, b_max=1.0, clip=True),
               Resized(keys="image", spatial_size=NET_SIZE, mode="bilinear")])
from monai.transforms import Activationsd, Invertd

post = Compose([Activationsd(keys="pred", softmax=True),
                Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=False, to_tensor=True)])
d = pre({"image": test_files[0]["image"]})
with torch.no_grad():
    d["pred"] = model(torch.as_tensor(d["image"])[None])[0]
d = post(d)
mask_invertd = (np.asarray(d["pred"][1]) > 0.5).astype(np.uint8)
print("Invertd mask", mask_invertd.shape, " vs ours", segment_slice(hu, spacing, model, cfg, clean=False).mask.shape,
      " Dice between them:", round(dice_score(mask_invertd, segment_slice(hu, spacing, model, cfg, clean=False).mask), 4))

# %% [markdown]
# ## 10. Recap
# * A deployable model = **one function** `segment_slice(hu, spacing, model, cfg)` + readers + config + tests.
# * Preprocessing is part of the model: share the code, read the numbers from `config.json`, **test equivalence**.
# * Skew is silent: only comparison with a reference (tests) reveals it.
# * Measure latency/size on the **target hardware** (CPU); we ship `state_dict` + config, loaded with `weights_only=True`.
# * Write the model card **before** the demo, with the limits and the "not a medical device" notice.
# * Tomorrow: `app_starter_kit.zip` -> VS Code -> Streamlit -> GitHub -> Streamlit Community Cloud -> Docker.
#
# **Files produced** (in `WORK_DIR`): `app_starter_kit.zip`, `models/<name>/{model.pt, config.json, MODEL_CARD.md}`.
#
# 📚 **References**
# * MONAI transforms and inverse transforms: https://docs.monai.io/en/stable/transforms.html
# * MONAI tutorials: https://github.com/Project-MONAI/tutorials
# * pydicom: https://pydicom.github.io/pydicom/stable/ - nibabel: https://nipy.org/nibabel/
# * PyTorch serialization notes (`weights_only`): https://pytorch.org/docs/stable/notes/serialization.html
# * LUNA16: https://luna16.grand-challenge.org/ - Hounsfield units: https://radiopaedia.org/articles/hounsfield-unit
# * Model cards: https://arxiv.org/abs/1810.03993 - Metrics Reloaded: https://arxiv.org/abs/2206.01653
