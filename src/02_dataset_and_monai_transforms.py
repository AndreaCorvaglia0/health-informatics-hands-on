# %% [markdown]
# # 02 - The dataset and MONAI transforms
# **Day 1 - part 2** | Advanced Topics in Machine Learning - hands-on | MSc Health Informatics
#
# In notebook 01 you looked at one scan in detail. Now we meet the **course dataset** (one axial chest-CT slice per
# patient, from LUNA16), explore it critically as data scientists should, and build the **pre-processing and
# augmentation pipeline** with [MONAI](https://monai.io) transforms. We finish with a correct batch coming out of a
# `DataLoader`, ready for the model of day 2. **No model is trained in this notebook.**
#
# **Learning objectives.** After this notebook you can:
# * describe the dataset (origin, how the slices were selected, label values, caveats, license) and explore its heterogeneity with pandas and matplotlib;
# * explain array vs dictionary transforms, `MetaTensor` and `Compose`;
# * build the pre-processing pipeline step by step (load, channel first, binary lung label, HU window, resize) and inspect every step;
# * choose augmentations that make anatomical sense for chest CT, and apply them identically to image and label;
# * get a correct batch `(B, 1, H, W)` out of a `Dataset`/`CacheDataset` + `DataLoader`.
#
# **Estimated time:** about 100 minutes (plus the optional challenges).
#
# **Legend**
#
# | Marker | Meaning |
# |---|---|
# | ✏️ **Exercise** | Mandatory. Fill the blanks `____`, then run the check cell. Hints are hidden in 💡 blocks. |
# | 🔍 **Explore** | Small experiments: change something and re-run. |
# | 🚀 **Challenge** | Additional task for those who finish early. |
# | 🌶️ **Going further** | Optional, open-ended. |
# | 🧠 **Question** | Think (or discuss with your neighbour) before moving on. |
# | ⚠️ **Pitfall** | A classic mistake. |
# | 📚 **Read more** | Links for later. |

# %% [markdown]
# ## 1. Setup

# %%
# Install what is missing (quiet). On Kaggle/Colab most packages are already there.
import importlib.util
import subprocess
import sys

for module, pip_name in {"monai": "monai", "nibabel": "nibabel"}.items():
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
import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import monai
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (AsDiscreted, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, Flipd,
                              Lambdad, LoadImage, LoadImaged, MapTransform, RandAffined, RandGaussianNoised,
                              Resized, ScaleIntensityRanged, Spacingd)
from monai.utils import first, set_determinism

set_determinism(seed=42)
plt.rcParams.update({"figure.dpi": 100, "image.cmap": "gray", "image.interpolation": "nearest"})

# Settings that depend on FAST (a tiny debug run on a laptop) or on a real run
IMG_SIZE = 64 if FAST else 256          # images are resized to IMG_SIZE x IMG_SIZE
BATCH_SIZE = 8 if FAST else 16
NUM_WORKERS = 0 if FAST else 2
N_GALLERY = 4 if FAST else 12
N_TIMING = 12 if FAST else 64           # how many cases we use to time the data loading

print("MONAI", monai.__version__, "| torch", torch.__version__)

# %% [markdown]
# ## 2. The dataset card: `luna16-lung-slices`
# | | |
# |---|---|
# | **Source** | [LUNA16](https://luna16.grand-challenge.org/): 888 chest CT scans selected from the public LIDC-IDRI collection (lung-cancer screening scans from several US centres). The challenge was designed for *nodule detection*, but it ships lung masks too. |
# | **License** | CC BY 4.0: you may share and adapt, **you must give credit**. Cite Setio et al., *Medical Image Analysis* 42 (2017) 1-13, [doi:10.1016/j.media.2017.06.015](https://doi.org/10.1016/j.media.2017.06.015). |
# | **What we did** | One **2D axial slice per patient**, chosen *at random among the slices that contain lung*, so there are 888 cases (one patient = one slice = one case). 2D keeps training fast on free GPUs. |
# | **Images** | `images/case_XXXX.nii.gz`: int16, **Hounsfield units**, usually 512x512, stored in LPS index order (use `img.T` to display). |
# | **Labels** | `labels/case_XXXX.nii.gz`: uint8, **0** background, **1** right lung, **2** left lung, **3** trachea / main airways. |
# | **Binary task** | lungs = label 1 or 2; the trachea counts as background. The 3-class version is an additional task. |
#
# **`metadata.csv`** (one row per case):
#
# | Column | Meaning |
# |---|---|
# | `case_id`, `seriesuid` | our id and the original LUNA16 series id |
# | `subset` | 0-9: the official LUNA16 10-fold split. We use 0-6 train, 7-8 validation, 9 test |
# | `slice_index`, `n_slices`, `z_rel` | which slice of the original scan we kept; `z_rel` = relative height (0 = most inferior, 1 = most superior) |
# | `spacing_x`, `spacing_y`, `slice_thickness` | pixel size and slice thickness in mm |
# | `size_x`, `size_y` | image size in pixels |
# | `right_lung_px`, `left_lung_px`, `trachea_px`, `lung_area_mm2` | size of the masks (pixels) and total lung area (mm²) |
# | `has_nodule` | 1 if the *patient* has at least one annotated nodule (not necessarily on this slice!) |
#
# > ⚠️ **Pitfall (silver standard):** the LUNA16 lung masks were **not drawn by radiologists**: they were produced by an
# > automatic algorithm. They are *silver standard* labels. A model trained on them learns to imitate that algorithm,
# > errors included, and a Dice of 0.99 against them is not the same as 0.99 against an expert. Keep this in mind when we evaluate on day 3.
# >
# > Also: all the slices are random, so some are close to the apex or the base of the lungs where the lungs are small
# > and the anatomy is complicated (diaphragm, liver, stomach gas).

# %% [markdown]
# ## 3. Exploring the data (EDA)
# Never train on a dataset you have not looked at. First the table:

# %%
meta = pd.read_csv(DATA_DIR / "metadata.csv")
print(f"{len(meta)} cases")
meta.head()

# %%
meta.describe().T[["count", "mean", "std", "min", "max"]].round(2)

# %% [markdown]
# Now the distributions that tell us how **heterogeneous** the images are:

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
meta["spacing_x"].plot.hist(bins=30, ax=axes[0, 0], title="pixel spacing x (mm)")
meta["slice_thickness"].plot.hist(bins=30, ax=axes[0, 1], title="slice thickness (mm) of the original scan")
meta["z_rel"].plot.hist(bins=30, ax=axes[0, 2], title="z_rel (0 = base, 1 = apex)")
meta["lung_area_mm2"].div(100).plot.hist(bins=30, ax=axes[1, 0], title="lung area (cm$^2$)")
meta[["size_x", "size_y"]].astype(str).agg("x".join, axis=1).value_counts().plot.bar(ax=axes[1, 1], title="image size (pixels)", rot=0)
meta["subset"].value_counts().sort_index().plot.bar(ax=axes[1, 2], title="cases per LUNA16 subset", rot=0)
for ax in axes.ravel():
    ax.set_ylabel("")
plt.tight_layout()

# %% [markdown]
# ### ✏️ Exercise 2.1 - two small pandas questions
# 1. `cases_per_subset`: how many cases are there in each subset? (a `Series` indexed by subset, sorted by subset number)
# 2. `smallest`: the row of `meta` of the case with the **smallest lung area**. Then we display it.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# For 1: count the occurrences of each value of the column `subset` (`value_counts`), then sort by the index (`sort_index`). For 2: there is a pandas method that returns the *index label* of the minimum of a column.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `meta["subset"].value_counts().sort_index()` and `meta.loc[meta["lung_area_mm2"].idxmin()]`
#
# </details>

# %%
#@solution
cases_per_subset = meta["subset"].value_counts().sort_index()
smallest = meta.loc[meta["lung_area_mm2"].idxmin()]
#@student
#| cases_per_subset = meta["subset"].____().____()   # TODO: count per subset, sorted by subset number
#| smallest = meta.loc[meta["lung_area_mm2"].____()]  # TODO: the row with the smallest lung area
#@end
print(cases_per_subset.to_dict())
print(smallest[["case_id", "z_rel", "spacing_x", "lung_area_mm2"]].to_dict())

# %%
assert cases_per_subset.sum() == len(meta) and list(cases_per_subset.index) == sorted(cases_per_subset.index)
assert smallest["lung_area_mm2"] == meta["lung_area_mm2"].min()
print("✅ Exercise 2.1 passed")

# %% [markdown]
# A small helper to load one case, and a gallery. Remember: **`img.T`** to display (notebook 01).

# %%
def case_paths(case_id):
    return DATA_DIR / "images" / f"{case_id}.nii.gz", DATA_DIR / "labels" / f"{case_id}.nii.gz"

def load_case(case_id):
    """Raw image (HU) and label of one case as numpy arrays indexed [x, y]."""
    img_path, lab_path = case_paths(case_id)
    return np.asanyarray(nib.load(img_path).dataobj), np.asanyarray(nib.load(lab_path).dataobj)

LABEL_COLORS = {1: "tab:red", 2: "tab:blue", 3: "gold"}   # right lung, left lung, trachea

def label_to_rgba(lab2d, alpha=0.4):
    """Colour a 2D label map (0 = transparent) to draw it on top of an image."""
    rgba = np.zeros(lab2d.shape + (4,))
    for value, color in LABEL_COLORS.items():
        rgba[lab2d == value] = (*mcolors.to_rgb(color), alpha)
    return rgba

def show_case(ax, img, lab, title=""):
    ax.imshow(img.T, cmap="gray", vmin=-1000, vmax=400)          # fixed lung-ish window for display only
    ax.imshow(label_to_rgba(lab).transpose(1, 0, 2))
    ax.set_title(title, fontsize=8)
    ax.axis("off")

rows = meta.sample(N_GALLERY, random_state=0)
n_cols = 4
fig, axes = plt.subplots(int(np.ceil(N_GALLERY / n_cols)), n_cols, figsize=(14, 3.6 * int(np.ceil(N_GALLERY / n_cols))), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
    img, lab = load_case(r["case_id"])
    show_case(ax, img, lab, f"{r['case_id']}  z_rel={r['z_rel']:.2f}  {r['spacing_x']:.2f} mm  {r['slice_thickness']:.1f} mm thick")
plt.tight_layout()

# %%
# The case with the smallest lung area (exercise 2.1)
img, lab = load_case(smallest["case_id"])
fig, ax = plt.subplots(figsize=(5, 5))
show_case(ax, img, lab, f"{smallest['case_id']}: smallest lung area ({smallest['lung_area_mm2'] / 100:.0f} cm2), z_rel={smallest['z_rel']:.2f}")

# %% [markdown]
# > 🧠 **Questions** (discuss, then open the answers):
# > 1. Pixel spacing and slice thickness vary between cases. Why does this matter for a model that sees only pixel matrices? And for the *area in pixels* of a lung?
# > 2. What does the distribution of `z_rel` tell you about which anatomy the model will see most? What would happen in an app if a user uploads a slice at the level of the liver dome or of the neck?
# > 3. Which biases may this dataset have (population, scanners, disease, how it was labeled)? Name two things you would check before claiming "the model works on patients".
# > 4. Look at the gallery: do you see any slice where the mask looks wrong or inconsistent? What does that tell you about using it as ground truth?

# %% [markdown] tags=["solution-only"]
# **Answers.**
# 1. A network sees pixels, not millimetres. The same lung is 1.3x larger in pixels at 0.6 mm than at 0.8 mm; pixel areas
#    cannot be compared across patients, and after a fixed-size `Resize` the physical scale of a pixel changes from case to case
#    (the network must be scale-tolerant, which augmentation with scaling helps). Clinically meaningful quantities (areas, volumes) need the spacing.
# 2. The slices come from the lung range only (random), so most are mid-lung where the lungs are large; apex and base slices are rarer
#    and harder. Slices outside the lung range (neck, abdomen) were never seen: the model would output garbage with high confidence. An app should say what it was trained for.
# 3. LIDC-IDRI is a screening/research collection from a few US institutions: limited scanners/protocols, adults with (mostly) nodule-related findings, few severe lung diseases
#    (consolidation, effusion, pneumothorax), no paediatric patients, thick slices excluded. Labels are automatic (silver standard). Check: performance per subset/scanner/spacing,
#    per z_rel, with/without nodules, and on an external dataset.
# 4. Typical findings on real data: the masks follow the algorithm's idea of the lung border (vessels and the hilum areas are inside or outside inconsistently), bowel gas or the trachea/bronchi
#    can be merged or cut, small apical/basal lungs are very rough. The labels are a (good) approximation, not the truth.

# %% [markdown]
# ## 4. MONAI in five minutes
# [MONAI](https://docs.monai.io/en/stable/) (Medical Open Network for AI) is a PyTorch-based library with the building blocks for medical imaging: transforms, datasets, networks, losses, metrics.
# It reads our files and returns tensors that **carry their metadata**.
#
# * **Array transforms** (`LoadImage`, `ScaleIntensityRange`, ...) work on one array: `out = transform(image)`.
# * **Dictionary transforms** (the same names with a `d`: `LoadImaged`, `ScaleIntensityRanged`, ...) work on a dict `{"image": ..., "label": ...}` and apply the operation to the `keys` you name. This is what we use for segmentation, because
#   **image and label must be transformed together** (same resize, same random rotation).
# * `Compose([t1, t2, ...])` chains transforms into a pipeline.
#
# `LoadImage` returns a **`MetaTensor`**: a `torch.Tensor` with the metadata attached (`.meta`, `.affine`, `.pixdim`).

# %%
img_path, lab_path = case_paths(meta["case_id"].iloc[0])
loaded = LoadImage(image_only=True)(img_path)
print(type(loaded).__name__, tuple(loaded.shape), loaded.dtype)
print("pixdim (mm):", loaded.pixdim.numpy() if hasattr(loaded.pixdim, "numpy") else loaded.pixdim)
print("affine:\n", np.round(loaded.affine.numpy(), 3))
print("some metadata keys:", list(loaded.meta.keys())[:8], "...")

# %% [markdown]
# > 📚 **Read more:** [MONAI transforms (API)](https://docs.monai.io/en/stable/transforms.html) |
# > [tutorial: transforms demo 2D](https://github.com/Project-MONAI/tutorials/blob/main/modules/transforms_demo_2d.ipynb) |
# > [tutorial: spleen segmentation 3D](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb) (the style we follow).

# %% [markdown]
# ## 5. Building the pre-processing, step by step
# We take one **typical case** (median lung area) and apply the transforms one at a time, looking at the result after each step.
# A dictionary with the *paths* is all we need to start: the first transform loads the files.

# %%
typical_id = meta.loc[(meta["lung_area_mm2"] - meta["lung_area_mm2"].median()).abs().idxmin(), "case_id"]
img_path, lab_path = case_paths(typical_id)
sample = {"image": str(img_path), "label": str(lab_path), "case_id": typical_id}
keys = ["image", "label"]

def describe(name, x):
    """One-line summary of a tensor: type, shape, dtype, value range."""
    print(f"{name:6s} {type(x).__name__:10s} shape={tuple(x.shape)!s:16s} dtype={str(x.dtype):14s} range=[{float(x.min()):.3g}, {float(x.max()):.3g}]")

def show_step(data, title, img_range=None):
    """Plot image, label and the overlay of the dictionary `data`."""
    img, lab = (np.asarray(data[k]).squeeze() for k in keys)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img.T, vmin=None if img_range is None else img_range[0], vmax=None if img_range is None else img_range[1])
    axes[0].set_title("image")
    axes[1].imshow(lab.T, cmap="viridis", vmin=0, vmax=3)
    axes[1].set_title(f"label, values {np.unique(lab).round(2)[:6]}")
    axes[2].imshow(img.T, vmin=None if img_range is None else img_range[0], vmax=None if img_range is None else img_range[1])
    axes[2].imshow(label_to_rgba(np.round(lab)).transpose(1, 0, 2))
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()

# %% [markdown]
# ### Step 1 - `LoadImaged` and `EnsureChannelFirstd`
# `LoadImaged` reads the files into `MetaTensor`s. PyTorch layers expect **channel first**: `(C, H, W)`. Our 2D files have no
# channel axis, so `EnsureChannelFirstd` adds one (it reads the metadata to decide what to do).

# %%
step1 = LoadImaged(keys=keys)(sample)
describe("image", step1["image"]); describe("label", step1["label"])
step2 = EnsureChannelFirstd(keys=keys)(step1)
describe("image", step2["image"]); describe("label", step2["label"])
show_step(step2, "after LoadImaged + EnsureChannelFirstd (raw HU)", img_range=(-1000, 400))

# %% [markdown]
# Note the values: the image is in **HU**, the label has the four values 0, 1, 2, 3. MONAI loads both as `float32` by default.
#
# ### Step 2 - from 3 classes to a binary "lungs" label
# For the mandatory task we want *lungs vs everything else*: labels 1 and 2 become 1; background and **trachea** become 0.
# (Why keep the trachea out? It is an airway, not lung parenchyma: if we segmented it as lung, a downstream measurement
# such as lung volume would be biased. And it is a nice example of a labelling *decision*: write it down, because every model and app depends on it.)
#
# ### ✏️ Exercise 2.2 - `lungs_only`
# Complete the function so that it maps the labels {1, 2} to 1 and everything else to 0, **keeping the dtype**. We then use it
# with `Lambdad`, which applies any function to the chosen keys.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# A comparison such as `x == 1` gives a boolean tensor; combine two of them with `|` (or).
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `((x == 1) | (x == 2)).to(x.dtype)`
#
# </details>

# %%
def lungs_only(x):
    """1 = right lung, 2 = left lung -> 1; everything else (incl. trachea) -> 0."""
    #@solution
    return ((x == 1) | (x == 2)).to(x.dtype)
    #@student
    #| return ((x == ____) | (x == ____)).to(x.dtype)  # TODO: which label values are lung?
    #@end


step3 = Lambdad(keys="label", func=lungs_only)(step2)
describe("label", step3["label"])
show_step(step3, "after lungs_only: binary label", img_range=(-1000, 400))

# %%
test = lungs_only(torch.tensor([0.0, 1.0, 2.0, 3.0]))
assert test.tolist() == [0.0, 1.0, 1.0, 0.0], "expected [0, 1, 1, 0]: lungs = labels 1 and 2, trachea (3) is background"
assert test.dtype == torch.float32, "keep the input dtype"
assert set(torch.unique(step3["label"]).tolist()) <= {0.0, 1.0}
print("✅ Exercise 2.2 passed")

# %% [markdown]
# ### Step 3 - HU window and scaling to [0, 1]
# A network trains best with inputs of order 1, not with values from -1000 to 3000. But we do not want a per-image normalisation
# (as for MRI): HU are physical, so we use a **fixed window**: clip HU to `[a_min, a_max]` and rescale linearly to `[0, 1]` with `ScaleIntensityRanged`.
#
# ### ✏️ Exercise 2.3 - choose the HU window
# Set `HU_MIN` and `HU_MAX`. The lung must be clearly visible *and* the soft tissue around it must not saturate.
# **Use -1000 and 400**: air is 0, lung parenchyma ends up around 0.1, soft tissue around 0.75, bone saturates at 1.
# These two numbers are used by all the following notebooks and by the web app.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Air is -1000 HU; soft tissue is 0-80 HU. A window ending at 400 HU keeps the soft tissue and the blood vessels but saturates the dense bone, which we do not need for lungs.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `HU_MIN, HU_MAX = -1000, 400`
#
# </details>

# %%
#@solution
HU_MIN, HU_MAX = -1000, 400
#@student
#| HU_MIN, HU_MAX = ____, ____   # TODO
#@end

scale = ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True)
step4 = scale(step3)
describe("image", step4["image"])
show_step(step4, "after ScaleIntensityRanged (window)", img_range=(0, 1))

# %%
def window_value(hu):
    return (hu - HU_MIN) / (HU_MAX - HU_MIN)

assert HU_MIN < -900 and HU_MAX > 200, "the window must contain air, lung (about -850 HU) and soft tissue (about 40 HU)"
assert 0 < window_value(-850) < 1 and 0 < window_value(40) < 1, "lung and soft tissue must not saturate"
assert 0.0 <= float(step4["image"].min()) and float(step4["image"].max()) <= 1.0
print("✅ Exercise 2.3 passed")

# %% [markdown]
# What did the window do? The histograms (log counts) before and after, and what happens **without clipping** or with a **bone window** (-500 to 1300 HU):

# %%
raw = step3["image"]
no_clip = ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=False)(step3)["image"]
bone = ScaleIntensityRanged(keys="image", a_min=-500, a_max=1300, b_min=0.0, b_max=1.0, clip=True)(step3)["image"]

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for col, (name, x) in enumerate([("raw HU", raw), ("window -1000..400, clip=True", step4["image"]),
                                 ("clip=False", no_clip), ("bone window -500..1300", bone)]):
    x = np.asarray(x).squeeze()
    axes[0, col].hist(x.ravel(), bins=100, log=True)
    axes[0, col].set_title(f"{name}\nrange [{x.min():.2f}, {x.max():.2f}]", fontsize=9)
    axes[1, col].imshow(x.T)
    axes[1, col].axis("off")
plt.tight_layout()

# %% [markdown]
# ### 🔍 Explore
# * With `clip=False` the values go far beyond [0, 1] (scanner padding at -2000 or -3024 HU in some scans!). What would this do to a network with a sigmoid/batch-norm? Look at the range printed above.
# * With the bone window the lungs are almost invisible (all dark): the window **is a modelling decision**. Try `a_min=-1000, a_max=0` too: what is lost?
# * Is the *scaling* the same for every patient with this method? What could go wrong with `ScaleIntensityd` (min-max **per image**) on CT instead?

# %% [markdown]
# ### Step 4 - `Resized`
# Images must have the same size to be stacked into a batch. `Resized(spatial_size=(S, S))` interpolates to `S x S`; with the mixed `mode`
# argument we choose the interpolation **per key**. Our `IMG_SIZE` is 256 (64 in FAST mode): a 512x512 slice at 256x256 already has
# 4x fewer pixels, which makes training 4x faster, at the price of finer details.
#
# > ⚠️ **Pitfall:** a label is a map of *categories*, not of intensities. Interpolating it linearly invents values that do not exist.
# > Let us see it on the raw 3-class label (0 = background, 1 = right lung, 2 = left lung, 3 = trachea):

# %%
lab3 = step2["label"]   # raw 3-class label, channel first
lab_bilinear = Resized(keys="label", spatial_size=(IMG_SIZE, IMG_SIZE), mode="bilinear")(step2)["label"]
lab_nearest = Resized(keys="label", spatial_size=(IMG_SIZE, IMG_SIZE), mode="nearest")(step2)["label"]

non_integer = (lab_bilinear - lab_bilinear.round()).abs() > 1e-3
print("bilinear: distinct values:", len(torch.unique(lab_bilinear)), "| pixels with a non-integer value:", int(non_integer.sum()))
print("nearest : distinct values:", torch.unique(lab_nearest).tolist())

# Round the bilinear result to "fix" it: do we get back the nearest-neighbour label?
differs = (lab_bilinear.round() != lab_nearest).sum()
print("after rounding, pixels that differ from the nearest-neighbour label:", int(differs))

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (title, x) in zip(axes, [("bilinear (wrong)", lab_bilinear), ("nearest (right)", lab_nearest)]):
    ax.imshow(np.asarray(x).squeeze().T, cmap="viridis", vmin=0, vmax=3); ax.set_title(title)
axes[2].imshow(np.asarray(non_integer).squeeze().T); axes[2].set_title("pixels with invented (non-integer) values")
for ax in axes:
    ax.axis("off")
plt.tight_layout()

# %% [markdown]
# With a multi-class label, bilinear interpolation between label 0 (background) and 3 (trachea) produces 1s and 2s around the trachea:
# **a ring of "right lung" and "left lung" around the airway**, classes that do not exist there. Even for a binary 0/1 label it produces values such as 0.37 that
# are neither classes nor probabilities. For labels always use `nearest` (or resample one-hot channels, then argmax). For the image, `bilinear` is fine.
#
# ### ✏️ Exercise 2.4 - `Resized` with the right modes
# Resize both keys to `IMG_SIZE x IMG_SIZE`: bilinear for the image, nearest for the label. The order of the modes follows the order of `keys`.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `mode` accepts a tuple with one entry per key. The first key is `"image"`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `mode=("bilinear", "nearest")`
#
# </details>

# %%
#@solution
resize = Resized(keys=keys, spatial_size=(IMG_SIZE, IMG_SIZE), mode=("bilinear", "nearest"))
#@student
#| resize = Resized(keys=keys, spatial_size=(IMG_SIZE, IMG_SIZE), mode=("____", "____"))  # TODO: image, label
#@end
step5 = resize(step4)
describe("image", step5["image"]); describe("label", step5["label"])
show_step(step5, f"after Resized to {IMG_SIZE}x{IMG_SIZE}", img_range=(0, 1))

# %%
assert tuple(step5["image"].shape) == (1, IMG_SIZE, IMG_SIZE) and tuple(step5["label"].shape) == (1, IMG_SIZE, IMG_SIZE)
assert set(torch.unique(step5["label"]).tolist()) <= {0.0, 1.0}, "the label must stay binary: use nearest for the label!"
assert float(step5["image"].min()) >= 0 and float(step5["image"].max()) <= 1.0 + 1e-4
print("✅ Exercise 2.4 passed")

# %% [markdown]
# ### Resize or resample to a fixed spacing? (`Spacingd`)
# `Resized` always gives the same *matrix size*, but a pixel then covers a different physical size in each case. The alternative is to
# **resample to a fixed pixel size in mm** with `Spacingd(pixdim=...)`: every pixel is e.g. 0.8 mm, but the matrix size now depends on the field of view,
# so we need to crop or pad to a common size afterwards.

# %%
small_sp, large_sp = meta.loc[meta["spacing_x"].idxmin(), "case_id"], meta.loc[meta["spacing_x"].idxmax(), "case_id"]
for case_id in (small_sp, large_sp):
    img_p, lab_p = case_paths(case_id)
    d = EnsureChannelFirstd(keys=keys)(LoadImaged(keys=keys)({"image": str(img_p), "label": str(lab_p)}))
    d_res = Spacingd(keys=keys, pixdim=(0.8, 0.8), mode=("bilinear", "nearest"))(d)
    print(f"{case_id}: pixel {d['image'].pixdim[0]:.2f} mm, matrix {tuple(d['image'].shape[1:])}"
          f"  ->  Spacingd(0.8 mm): matrix {tuple(d_res['image'].shape[1:])}")

# %% [markdown]
# | | `Resized` to a fixed matrix | `Spacingd` to a fixed pixel size |
# |---|---|---|
# | Keeps the full field of view | yes | yes, but the matrix size varies (needs pad/crop) |
# | Pixel size in mm | varies between patients | **same for all** |
# | Right when... | the task does not depend on absolute size, a 2D app accepts any image (a PNG has no spacing!) | measuring in mm (areas, volumes, distances), 3D data, multi-centre data with very different spacing |
#
# We use `Resized`: it is simple, robust, and the web app will receive images that may have no spacing information at all.
# For 3D, `Spacingd` (together with `Orientationd`) is the standard choice (see the 🌶️ ideas at the end).

# %% [markdown]
# ### Step 5 - `EnsureTyped`, and putting it all together
# `EnsureTyped` guarantees that the data are tensors of the dtype we want, at the end of the pipeline. We now put the steps in one function.
# The `augment` switch will be used in the next section (for now `get_augmentations` returns nothing).
#
# ### ✏️ Exercise 2.5 - assemble `get_transforms`
# Complete the function so that it does the five steps above, **in this order**: load, channel first, binary label, scale the HU window to [0, 1] (with the `hu_min`/`hu_max` arguments), resize.
# This function is the one used on all the following days (and by the app): it must behave exactly like the course reference.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Every step you built above is one transform of the list. The `keys` are `["image", "label"]`, except for the label function (only the label) and the intensity scaling (only the image).
#
# </details>
#
# <details>
# <summary>💡 Hint 2</summary>
#
# `Lambdad(keys="label", func=lungs_only)`, then `ScaleIntensityRanged(keys="image", a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True)`, then `Resized(...)` with the two modes.
#
# </details>
#
# <details>
# <summary>💡 Hint 3 (almost the solution)</summary>
#
# `Resized(keys=keys, spatial_size=(img_size, img_size), mode=("bilinear", "nearest"))`
#
# </details>

# %%
def get_augmentations(img_size):
    """Random transforms applied to the training data only. Filled in section 6."""
    return []


def get_transforms(img_size, hu_min=-1000, hu_max=400, augment=False):
    """Pre-processing pipeline (+ optional augmentation) for image/label dictionaries."""
    keys = ["image", "label"]
    t = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        #@solution
        Lambdad(keys="label", func=lungs_only),
        ScaleIntensityRanged(keys="image", a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=keys, spatial_size=(img_size, img_size), mode=("bilinear", "nearest")),
        #@student
        #| ____,  # TODO: label -> binary lungs
        #| ____,  # TODO: HU window -> [0, 1]
        #| ____,  # TODO: resize (image bilinear, label nearest)
        #@end
    ]
    if augment:
        t += get_augmentations(img_size)
    t += [EnsureTyped(keys=keys)]
    return Compose(t)


preprocess = get_transforms(IMG_SIZE)
out = preprocess(sample)
describe("image", out["image"]); describe("label", out["label"])
show_step(out, "output of get_transforms(IMG_SIZE)", img_range=(0, 1))

# %% [markdown]
# Check cell: instead of comparing with a reference solution, we verify the **behaviour** with an independent numpy computation
# (clip + scale the raw HU by hand, binary label by hand) on three random cases.

# %%
for case_id in meta["case_id"].sample(3, random_state=1):
    img_p, lab_p = case_paths(case_id)
    res = get_transforms(IMG_SIZE, hu_min=-1000, hu_max=400)({"image": str(img_p), "label": str(lab_p)})
    raw_img, raw_lab = load_case(case_id)
    expected_mean = np.clip((raw_img.astype(np.float64) + 1000) / 1400, 0, 1).mean()
    expected_frac = np.isin(raw_lab, (1, 2)).mean()
    assert tuple(res["image"].shape) == (1, IMG_SIZE, IMG_SIZE) and tuple(res["label"].shape) == (1, IMG_SIZE, IMG_SIZE)
    assert res["image"].dtype == torch.float32
    assert 0 <= float(res["image"].min()) and float(res["image"].max()) <= 1 + 1e-4, "the image must be in [0, 1] (clip!)"
    assert abs(float(res["image"].mean()) - expected_mean) < 0.03, "the window is not -1000..400 HU -> [0, 1] with clipping"
    assert set(torch.unique(res["label"]).tolist()) <= {0.0, 1.0}, "binary label with values 0 and 1 only"
    assert abs(float(res["label"].mean()) - expected_frac) < 0.03, "lung fraction changed: labels 1 and 2 are lung, 3 is not"
print("✅ Exercise 2.5 passed")

# %% [markdown]
# ## 6. Data augmentation
# We have a few hundred training images. A network can memorise them, so we create *new plausible examples* by randomly perturbing
# the training images **on the fly**, every time they are loaded. Augmentation encodes our knowledge of **invariances**: "a lung that is rotated by 5 degrees is still a lung".
#
# What makes sense for **chest CT**?
#
# | Transform | Verdict |
# |---|---|
# | Small rotation (about ±8 degrees), scaling (±10 %), translation (±5 %) | Yes: patients are never positioned identically, body sizes differ. |
# | Gaussian noise, slight blur | Yes: scanners/doses/reconstruction kernels differ. |
# | Small intensity shift/scale (a few %) | OK in moderation, but **HU have a physical meaning**: big changes (e.g. contrast x2) create tissues that do not exist. |
# | Elastic deformation (mild) | Often useful, imitates anatomical variability. |
# | **Left-right flip** | Careful: it swaps the patient's right and left lung (the heart goes to the right!). Harmless for the *binary* task, a **bug for the 3-class task** (label 1 would be on the left). |
# | 90-degree rotations, up-down flip | No: chest CT always has the same orientation (spine at the bottom). The model would waste capacity on impossible images, and our app will never see them. |
#
# **Order matters.** Geometric transforms fill the new border with a value: after our window the image is in [0, 1] and `0` is *air*, which is
# physically right for the border. If you rotated the *raw HU* image and padded with 0, you would pad with 0 HU = water. That is why the random transforms come **after** `ScaleIntensityRanged`.
#
# **Same transform for image and label.** A dictionary transform draws the random parameters **once** per sample and applies them to all the keys. Let us see what happens
# if we do *not* use it (two independent array transforms, each drawing its own rotation):

# %%
from monai.transforms import RandRotate

img_t, lab_t = out["image"], out["label"]
rot_img = RandRotate(range_x=0.3, prob=1.0, mode="bilinear", padding_mode="zeros")
rot_lab = RandRotate(range_x=0.3, prob=1.0, mode="nearest", padding_mode="zeros")
rot_img.set_random_state(seed=1); rot_lab.set_random_state(seed=2)    # two independent random draws: the bug
wrong_img, wrong_lab = rot_img(img_t), rot_lab(lab_t)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.imshow(np.asarray(wrong_img).squeeze().T, vmin=0, vmax=1)
ax.contour(np.asarray(wrong_lab).squeeze().T, levels=[0.5], colors="r", linewidths=1)
ax.set_title("image and label rotated independently\n= misaligned (never do this)", fontsize=9); ax.axis("off")

# %% [markdown]
# The left-right flip with a 3-class label. We flip the raw sample along the left-right axis (axis 0 of our `[x, y]` arrays) and see where the right lung (label 1) goes:

# %%
flipped = Flipd(keys=keys, spatial_axis=0)(step2)
def centroid_x(lab, value):
    xs = np.nonzero(np.asarray(lab).squeeze() == value)[0]
    return xs.mean() / lab.shape[-2]

print(f"label 1 = RIGHT lung, x centroid (0 = left of the array, 1 = right): "
      f"original {centroid_x(step2['label'], 1):.2f} -> flipped {centroid_x(flipped['label'], 1):.2f}")
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, (title, d) in zip(axes, [("original", step2), ("flipped left-right", flipped)]):
    show_case(ax, np.asarray(d["image"]).squeeze(), np.asarray(d["label"]).squeeze(), title)
plt.tight_layout()

# %% [markdown]
# After the flip the anatomy is mirrored (the heart is on the right, the larger right lung on the left of the screen) but label 1 still says "right lung".
# Labels and anatomy are now inconsistent: for a 3-class model this is a silent bug. (If you flip, you must also swap labels 1 and 2.)
#
# ### ✏️ Exercise 2.6 - build the augmentation
# Complete `get_augmentations`. Use **`RandAffined`** (random rotation + scaling + translation, applied to image and label, with `bilinear` for the image and `nearest` for the label)
# and **one intensity transform** on the image only: `RandGaussianNoised`. Suggested ranges: rotation about ±8 degrees (`rotate_range` is in **radians**), scaling ±10 %, translation ±5 % of the image (in pixels), `prob` 0.5 for the affine, noise `std` 0.02 with `prob` 0.2.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# 8 degrees is about 0.15 radians. `scale_range=0.1` means a random scale between 0.9 and 1.1. `translate_range` is in pixels: use a fraction of `img_size`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2</summary>
#
# Padding of the new border with zeros (`padding_mode="zeros"`) is correct here, because 0 means air after the window.
#
# </details>
#
# <details>
# <summary>💡 Hint 3 (almost the solution)</summary>
#
# `RandAffined(keys=keys, prob=0.5, rotate_range=0.15, scale_range=0.1, translate_range=img_size * 0.05, mode=("bilinear", "nearest"), padding_mode="zeros")` and `RandGaussianNoised(keys="image", prob=0.2, std=0.02)`
#
# </details>

# %%
def get_augmentations(img_size):
    """Random transforms applied to the training data only (after the window: 0 = air)."""
    keys = ["image", "label"]
    #@solution
    return [
        RandAffined(keys=keys, prob=0.5, rotate_range=0.15, scale_range=0.1, translate_range=img_size * 0.05,
                    mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandGaussianNoised(keys="image", prob=0.2, std=0.02),
    ]
    #@student
    #| return [
    #|     RandAffined(keys=keys, prob=____, rotate_range=____, scale_range=____, translate_range=____,
    #|                 mode=("____", "____"), padding_mode="zeros"),   # TODO: ranges as suggested above
    #|     RandGaussianNoised(keys="image", prob=____, std=____),           # TODO: prob and std
    #| ]
    #@end


train_tf = get_transforms(IMG_SIZE, augment=True)   # now includes the augmentation (get_transforms looks up get_augmentations at call time)

# %%
# Check: 12 random draws from the full augmented pipeline
draws = [train_tf(sample) for _ in range(12)]
base = preprocess(sample)
assert all(tuple(d["image"].shape) == (1, IMG_SIZE, IMG_SIZE) for d in draws)
assert all(set(torch.unique(d["label"]).tolist()) <= {0.0, 1.0} for d in draws), "the label must stay binary (nearest for the label!)"
assert all(float(d["image"].min()) >= -0.5 and float(d["image"].max()) <= 1.5 for d in draws), "the image must stay close to [0, 1]"
assert any(not torch.equal(d["image"], base["image"]) for d in draws), "nothing changed in 12 draws: is prob > 0?"
assert any(not torch.equal(d["label"], base["label"]) for d in draws), "the label never changed: is the geometric transform applied to the label too?"
print("✅ Exercise 2.6 passed")

# %% [markdown]
# The same sample, augmented 8 times (red line = label contour: it must follow the lung borders in every panel).

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
aug_only = Compose(get_augmentations(IMG_SIZE))
for i, ax in enumerate(axes.ravel()):
    a = aug_only(preprocess(sample)) if i else preprocess(sample)
    ax.imshow(np.asarray(a["image"]).squeeze().T, vmin=0, vmax=1)
    ax.contour(np.asarray(a["label"]).squeeze().T, levels=[0.5], colors="r", linewidths=1)
    ax.set_title("original" if i == 0 else f"augmented {i}", fontsize=9); ax.axis("off")
plt.tight_layout()

# %% [markdown]
# ### 🔍 Explore
# * Set `prob=1.0` and `rotate_range=0.6` (about 34 degrees): are the images still plausible? Where is the limit for a chest CT?
# * Change `translate_range` to `img_size * 0.3`: what happens with the lungs at the border? Would a real patient look like this?
# * **Determinism.** Run `aug_only.set_random_state(seed=3)` then apply `aug_only` twice to the same input, resetting the seed before each call: do you get the same image? `set_determinism(seed=42)` (top of the notebook) does the same for the global random generators, so that a *whole run* can be repeated.
# * Add a transform that is physically wrong, such as `Flipd(keys=keys, spatial_axis=1, prob=0.5)` (up-down flip) and look at the result.

# %% [markdown]
# ## 7. Dataset and DataLoader
# A **`Dataset`** applies the transforms to one item when you ask for `dataset[i]`; a **`DataLoader`** groups items into batches (and shuffles, and can use several worker processes).
# MONAI offers the same classes as PyTorch, plus useful variants:
#
# | Class | What it does |
# |---|---|
# | `monai.data.Dataset` | loads and transforms an item **every time** it is requested (nothing kept in memory) |
# | `monai.data.CacheDataset` | runs the **deterministic** transforms once and keeps the results in RAM; the **random** transforms (augmentation) are still applied each time |
# | `PersistentDataset`, `SmartCacheDataset` | cache on disk / cache only a moving window of items (for big data) |
#
# First the file list. For the real split (train/val/test by `subset`) we follow the course convention: subset 0-6 train, 7-8 validation, 9 test.
# One slice per patient: **a patient can never be in two splits** (this would be *leakage*; with several slices per patient you must split by patient, never by slice).

# %%
def make_datalist(frame):
    return [{"image": str(DATA_DIR / "images" / f"{c}.nii.gz"),
             "label": str(DATA_DIR / "labels" / f"{c}.nii.gz"), "case_id": c} for c in frame["case_id"]]

train_files = make_datalist(meta[meta["subset"] <= 6])
val_files = make_datalist(meta[meta["subset"].isin([7, 8])])
test_files = make_datalist(meta[meta["subset"] == 9])
print(f"train {len(train_files)} | val {len(val_files)} | test {len(test_files)} cases")
if FAST:
    train_files = train_files[:24]   # tiny debug run

# %% [markdown]
# ### ✏️ Exercise 2.7 - `Dataset`, `DataLoader`, one batch
# 1. Build `train_ds` with `Dataset(data=train_files, transform=...)` using the **augmented** transforms.
# 2. Build `train_loader`, a `DataLoader` with `batch_size=BATCH_SIZE`, `shuffle` on for training and `num_workers=NUM_WORKERS`.
# 3. Get one batch with `monai.utils.first`.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# We train on the *augmented* pipeline: `get_transforms(IMG_SIZE, augment=True)`. Shuffling the training data makes every epoch see the batches in a different order.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `first(loader)` returns the first batch of a loader (a dictionary, like the items).
#
# </details>

# %%
#@solution
train_ds = Dataset(data=train_files, transform=get_transforms(IMG_SIZE, augment=True))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
batch = first(train_loader)
#@student
#| train_ds = Dataset(data=train_files, transform=____)                 # TODO: the augmented pipeline
#| train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=____, num_workers=NUM_WORKERS)
#| batch = ____(train_loader)                                            # TODO: first batch
#@end
describe("image", batch["image"]); describe("label", batch["label"])
print("keys:", list(batch.keys()), "| case_ids:", batch["case_id"][:4], "...")

# %%
expected_b = min(BATCH_SIZE, len(train_files))
assert tuple(batch["image"].shape) == (expected_b, 1, IMG_SIZE, IMG_SIZE), f"expected (B, 1, H, W) = ({expected_b}, 1, {IMG_SIZE}, {IMG_SIZE})"
assert tuple(batch["label"].shape) == tuple(batch["image"].shape)
assert batch["image"].dtype == torch.float32
assert set(torch.unique(batch["label"]).tolist()) <= {0.0, 1.0}
assert float(batch["image"].min()) >= -0.5 and float(batch["image"].max()) <= 1.5
print("✅ Exercise 2.7 passed")

# %%
n_show = min(8, len(batch["image"]))
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax in axes.ravel():
    ax.axis("off")
for i, ax in zip(range(n_show), axes.ravel()):
    ax.imshow(batch["image"][i, 0].numpy().T, vmin=0, vmax=1)
    ax.contour(batch["label"][i, 0].numpy().T, levels=[0.5], colors="r", linewidths=1)
    ax.set_title(batch["case_id"][i], fontsize=9)
plt.tight_layout()

# %% [markdown]
# This is exactly what the model will see on day 2: a tensor `(B, 1, H, W)` with values in [0, 1] and a binary label of the same shape.
#
# ### ✏️ Exercise 2.8 - `CacheDataset` and timing an epoch
# Everything up to `Resized` is **deterministic**: the same file always gives the same result. Why re-read and re-resize it at every epoch?
# `CacheDataset` does it once and keeps the results in RAM; `cache_rate=1.0` caches all the items. Build `cache_ds` on the first `N_TIMING` training files with the augmented transforms
# and compare the time of one pass over the data with the plain `Dataset` and with the `CacheDataset`.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `cache_rate` is the fraction of items to cache: a number between 0 and 1.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `cache_rate=1.0`
#
# </details>

# %%
timing_files = train_files[:N_TIMING]

def time_epoch(dataset):
    """Seconds needed to iterate once over a dataset in batches."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    t0 = time.perf_counter()
    for _ in loader:
        pass
    return time.perf_counter() - t0

plain_ds = Dataset(data=timing_files, transform=get_transforms(IMG_SIZE, augment=True))
t0 = time.perf_counter()
#@solution
cache_ds = CacheDataset(data=timing_files, transform=get_transforms(IMG_SIZE, augment=True), cache_rate=1.0, num_workers=NUM_WORKERS)
#@student
#| cache_ds = CacheDataset(data=timing_files, transform=get_transforms(IMG_SIZE, augment=True), cache_rate=____, num_workers=NUM_WORKERS)  # TODO
#@end
t_build = time.perf_counter() - t0

print(f"{len(timing_files)} cases | Dataset: {time_epoch(plain_ds):.2f} s/epoch (every epoch)"
      f" | CacheDataset: {t_build:.2f} s to fill the cache, then {time_epoch(cache_ds):.2f} s/epoch")

# %%
assert len(cache_ds) == len(timing_files)
item = cache_ds[0]
assert tuple(item["image"].shape) == (1, IMG_SIZE, IMG_SIZE) and set(torch.unique(item["label"]).tolist()) <= {0.0, 1.0}
print("✅ Exercise 2.8 passed")

# %% [markdown]
# The cache pays off when an epoch is repeated many times (30 epochs on day 2). Notes:
# * The augmentation (random) part is **not** cached: every epoch the same cached item gets a *different* random perturbation. That is what we want.
# * `num_workers` > 0 loads batches in parallel processes: useful when the transforms are slow (3D, many random transforms). With `CacheDataset` the heavy part is already done and on a free GPU machine 2 workers are plenty. On Windows or in some notebooks workers can hang: if so, use `NUM_WORKERS = 0`.
# * The validation loader uses the transforms **without augmentation** (and no shuffling): we must evaluate on the data as they are.
#
# ### 🔍 Explore
# * Time the first and second epoch of the plain `Dataset` (the OS file cache can make the second faster).
# * Build `CacheDataset(..., cache_rate=0.5)`: what changes in memory and time?
# * Add `pin_memory=torch.cuda.is_available()` to the `DataLoader`. On a GPU machine, why could it help?

# %% [markdown]
# ### 🚀 Challenge (additional) - your own `MapTransform`
# `Lambdad` is great for one-liners, but a real transform is a **class**: subclass `MapTransform`, call `super().__init__(keys)`, implement `__call__(self, data)`
# that copies the dictionary, loops over `self.key_iterator(d)` and transforms each key. Write `LungsOnlyd` that does the job of `Lambdad(keys="label", func=lungs_only)`.
# The cell below runs as it is (the stub does nothing yet).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `d = dict(data)` makes a shallow copy so that the input is not modified in place.
#
# </details>
#
# <details>
# <summary>💡 Hint 2</summary>
#
# Inside the loop: `d[key] = lungs_only(d[key])`
#
# </details>

# %%
class LungsOnlyd(MapTransform):
    """Dictionary transform: lung labels {1, 2} -> 1, everything else -> 0."""

    #@solution
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = lungs_only(d[key])
        return d
    #@student
    #| def __call__(self, data):
    #|     d = dict(data)
    #|     # TODO: loop over self.key_iterator(d) and apply lungs_only
    #|     return d
    #@end


with_class = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys), LungsOnlyd(keys="label")])(sample)
print("✅ same result as Lambdad" if torch.equal(with_class["label"], step3["label"]) else "❌ not yet equal to the Lambdad result")

# %% [markdown]
# ### 🚀 Challenge (additional) - crop around the body
# The air around the patient and the scanner table carry no information but they take up pixels. `CropForegroundd` crops the image (and the label) to the bounding box of the "foreground".
# Insert `CropForegroundd(keys=keys, source_key="image", select_fn=lambda x: x > -500, margin=10)` **before** the intensity scaling (it works on raw HU: air is about -1000) and compare the result with the plain pipeline.
# What do you gain? What do you lose (hint: pixel size and the app)? The cell runs as it is, and returns the plain pipeline until you add the transform.

# %%
#@solution
crop_pipeline = Compose([
    LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys), Lambdad(keys="label", func=lungs_only),
    CropForegroundd(keys=keys, source_key="image", select_fn=lambda x: x > -500, margin=10),
    ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=keys, spatial_size=(IMG_SIZE, IMG_SIZE), mode=("bilinear", "nearest")), EnsureTyped(keys=keys),
])
#@student
#| crop_pipeline = get_transforms(IMG_SIZE)   # TODO: build a copy of the pipeline with CropForegroundd before the scaling
#@end
cropped = crop_pipeline(sample)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, (title, d) in zip(axes, [("plain pipeline", preprocess(sample)), ("with CropForegroundd", cropped)]):
    ax.imshow(np.asarray(d["image"]).squeeze().T, vmin=0, vmax=1)
    ax.contour(np.asarray(d["label"]).squeeze().T, levels=[0.5], colors="r", linewidths=1)
    ax.set_title(title); ax.axis("off")
plt.tight_layout()

# %% [markdown] tags=["solution-only"]
# **Discussion.** The cropped image uses its pixels better (the body fills the frame, a bit more resolution on the lungs) but the *scale* now depends on the body size of the patient, and, above all,
# **the app must apply exactly the same crop** at inference (and the crop rule can fail: a table or a plastic object counts as "foreground", a patient with a very low-density body may be cropped badly).
# More steps in the pipeline = more places where training-serving skew can appear (day 4).

# %% [markdown]
# ### 🚀 Challenge (additional) - keep the three classes, correctly
# Build `get_transforms_multiclass(img_size)`: the same pipeline **without** `lungs_only` (labels 0, 1, 2, 3 are kept), with a label that stays a valid class map after `Resized`, and with `AsDiscreted(keys="label", to_onehot=4)` at the end to get the one-hot label `(4, H, W)`.
# Check that: the label has exactly the values {0, 1, 2, 3} before the one-hot step, the one-hot channels sum to 1 in every pixel, and the augmentation (without any flip!) preserves this.
# The cell runs as it is, and gives the binary pipeline until you change it.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The modes of `Resized` are `("bilinear", "nearest")` as before: nearest keeps integer classes. `to_onehot` takes the number of classes.
#
# </details>

# %%
def get_transforms_multiclass(img_size, hu_min=-1000, hu_max=400):
    keys = ["image", "label"]
    #@solution
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityRanged(keys="image", a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=keys, spatial_size=(img_size, img_size), mode=("bilinear", "nearest")),
        *get_augmentations(img_size),
        EnsureTyped(keys=keys),
        AsDiscreted(keys="label", to_onehot=4),
    ])
    #@student
    #| return get_transforms(img_size)   # TODO: the 3-class version, one-hot label with 4 channels
    #@end


mc = get_transforms_multiclass(IMG_SIZE)(sample)
describe("label", mc["label"])
onehot_ok = mc["label"].shape[0] == 4 and torch.allclose(mc["label"].sum(0), torch.ones(IMG_SIZE, IMG_SIZE))
print("✅ one-hot label with 4 channels, one class per pixel" if onehot_ok else "❌ not yet: expected a (4, H, W) one-hot label")

# %% [markdown]
# ### 🌶️ Going further (optional)
# * **Inverse transforms.** After `Resized` the prediction has the model size, but a clinician wants it back at the original 512x512 (or in mm). Most MONAI spatial transforms are invertible: try `pipeline.inverse(pipeline(sample))` on a pipeline without `Lambdad`, and read about `Invertd` and test-time augmentation in the [inverse transforms tutorial](https://github.com/Project-MONAI/tutorials/blob/main/modules/inverse_transforms_and_test_time_augmentations.ipynb). This is exactly what day 4 (inference) needs.
# * **Elastic deformation.** Add `Rand2DElasticd(keys=keys, prob=0.3, spacing=(20, 20), magnitude_range=(1, 2), mode=("bilinear", "nearest"), padding_mode="zeros")` to the augmentations and look at 8 draws. When does it stop looking like a real lung?
# * **3D transforms on the demo volume.** Load `demo/volume/*_image.nii.gz` with `LoadImaged`, `EnsureChannelFirstd`, then try `Orientationd(keys="image", axcodes="RAS")` and `Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5))`: what are the shapes before/after? Which axis codes does the *same* file have now (compare with notebook 01)?
# * **A transform of your own** for the bias you identified in the 🧠 questions (e.g. simulate a different reconstruction kernel with a Gaussian blur or a sharpening transform), and test whether the augmented images look plausible.

# %% [markdown]
# ## 8. Recap: what to remember
# * The dataset: **one random lung slice per patient** from LUNA16 (CC BY 4.0, cite Setio et al. 2017), labels 0-3 from an **automatic algorithm** (silver standard). Look at heterogeneity (spacing, thickness, z level) before modelling.
# * MONAI **dictionary transforms** apply the same operations to image and label; `LoadImaged` returns `MetaTensor`s (data + metadata + affine).
# * The pipeline: load -> channel first -> binary lungs (trachea = background) -> **HU window [-1000, 400] -> [0, 1]** with clipping -> resize (**bilinear for the image, nearest for the label**).
# * Never interpolate labels linearly. Never apply a transformation to the image that you do not apply to the label. Mind the **order** (geometric augmentations after the window so that the border is air).
# * Augment only with transformations that produce *plausible* images: small affine, noise, mild intensity; **no left-right flip with left/right classes**, no 90-degree rotations, no big intensity changes.
# * A `DataLoader` gives `(B, 1, H, W)` tensors; `CacheDataset` stores the deterministic part; validation/test use the transforms **without** augmentation.
# * `get_transforms` and `lungs_only` are the reference pre-processing of the course: the app must do the same on day 5.
#
# **Next (day 2):** notebook 03 trains a U-Net on exactly these batches.
#
# > 📚 **Read more:**
# > * [MONAI transforms documentation](https://docs.monai.io/en/stable/transforms.html) and the [`modules/` tutorials](https://github.com/Project-MONAI/tutorials/tree/main/modules) (transforms demo 2D, inverse transforms, 2D slices from 3D)
# > * [MONAI data (Dataset, CacheDataset, DataLoader)](https://docs.monai.io/en/stable/data.html)
# > * Setio et al., *Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge*, Medical Image Analysis 42 (2017) 1-13, [doi:10.1016/j.media.2017.06.015](https://doi.org/10.1016/j.media.2017.06.015) and the [challenge site](https://luna16.grand-challenge.org/)
# > * Cardoso et al., *MONAI: An open-source framework for deep learning in healthcare*, [arXiv:2211.02701](https://arxiv.org/abs/2211.02701)
