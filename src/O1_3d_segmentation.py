# %% [markdown]
# # O1 - From 2D to 3D segmentation
# **Optional side track (days 3-4)** - for students, or a whole class, that are ahead of schedule.
#
# In the main path we segmented the lungs on **one axial slice per patient**. A CT scan is
# really a 3D volume of about 300 slices. In this notebook we move to **3D**: we train a 3D U-Net
# on small 3D **patches**, run it on whole volumes with a **sliding window**, measure the
# **lung volume in litres**, and finally ask the key question: *is a 3D model really better
# than our 2D model applied slice by slice?*
#
# This notebook is more self-driven than the core ones: there are fewer blanks, and you
# are expected to read the code, run it and extend it. The idioms follow the MONAI tutorial
# [`spleen_segmentation_3d`](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb).
#
# **Learning objectives**
# * explain what changes from 2D to 3D: memory, context, anisotropy, patches
# * build 3D dictionary transforms (orientation, resampling, patch sampling) and read their effect
# * create a 3D U-Net and run it on large volumes with `sliding_window_inference`
# * compute a 3D Dice score and a clinically meaningful number: **lung volume in litres**
# * compare a 2D model applied slice by slice against a true 3D model, and spot the typical 2D artefacts
#
# **Estimated time:** 2-3 hours (spine ✏️ about 1.5 h, 🚀 and 🌶️ optional).
#
# | Marker | Meaning |
# |---|---|
# | ✏️ Exercise | **Mandatory** (for this notebook): fill the blanks `____`, then run the check cell |
# | 🔍 Explore | Small "try this / change that" experiments |
# | 🚀 Challenge | Additional task (solution in the teachers' notebook) |
# | 🌶️ Going further | Open-ended, only pointers and starter code |
# | 🧠 Question / ⚠️ Pitfall / 📚 Read more | Concept question / classic mistake / links |
#
# > **Prerequisite:** the course dataset must contain the folder `volumes_lowres/` (3D volumes at
# > about 2.5 mm isotropic). Ask your teachers if it is missing. You also need the 2D model from day 2
# > (or the teachers' model) for the comparison in section 8.

# %% [markdown]
# ## 0. Setup and configuration

# %%
# Install what is missing (quiet). On Kaggle/Colab MONAI is usually not preinstalled.
import importlib, subprocess, sys

for module, pip_name in [("monai", "monai[nibabel]"), ("nibabel", "nibabel"), ("sklearn", "scikit-learn")]:
    try:
        importlib.import_module(module)
    except ImportError:
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
import copy
import json

import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (AsDiscrete, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, Lambdad,
                              LoadImaged, Orientationd, RandCropByPosNegLabeld, RandFlipd, ScaleIntensityRanged,
                              SpatialPadd, Spacingd)
from monai.utils import set_determinism

set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if FAST else 2

# 3D settings. The FAST values are chosen for the tiny synthetic volumes (about 48 x 48 x 32 voxels).
SPACING = (8.0, 8.0, 8.0) if FAST else (3.0, 3.0, 3.0)   # target voxel size in mm (isotropic)
ROI = (32, 32, 16) if FAST else (96, 96, 64)              # patch size for training (divisible by 16)
PATCHES_PER_VOLUME = 4                                    # patches cut from each volume at each step
MAX_EPOCHS = 20 if FAST else 60
print(f"device: {device} | voxel size {SPACING} mm | patch {ROI} | epochs {MAX_EPOCHS}")

# %%
# The 3D data live in a sub-folder of the course dataset. Stop early, with a clear message, if they are missing.
VOL_DIR = DATA_DIR / "volumes_lowres"
if not (VOL_DIR / "metadata.csv").exists():
    raise FileNotFoundError(
        f"3D volumes not found: expected {VOL_DIR}/{{images,labels,metadata.csv}}.\n"
        "This optional notebook needs the 'volumes_lowres' folder of the course dataset "
        "(low-resolution 3D CT volumes). Attach the full dataset, or ask your teachers for it."
    )

vol_meta = pd.read_csv(VOL_DIR / "metadata.csv")

def volume_rows(mask):
    return [{"image": str(VOL_DIR / "images" / f"{c}.nii.gz"),
             "label": str(VOL_DIR / "labels" / f"{c}.nii.gz"), "case_id": c}
            for c in vol_meta.loc[mask, "case_id"]]

# Same split rule as the course: subsets 0-6 train, 7-8 validation, 9 = test (untouched here).
train_files, val_files = volume_rows(vol_meta.subset <= 6), volume_rows(vol_meta.subset.isin([7, 8]))
if len(val_files) < 2:                        # tiny teaching dataset: carve a validation set out of the training one
    train_files, val_files = train_files[:-2], train_files[-2:]
if FAST:
    train_files = train_files[:4]
print(f"{len(train_files)} training volumes, {len(val_files)} validation volumes")
vol_meta.head()

# %% [markdown]
# ## 1. What changes when we go from 2D to 3D?
#
# | | 2D (main path) | 3D (this notebook) |
# |---|---|---|
# | Input | one axial slice, 256 x 256 | a volume, e.g. 512 x 512 x 300 (native) or about 140 x 140 x 120 (our low-res set) |
# | Context | only the slice | also the slices above and below: the model can follow a vessel or a bronchus, and knows if a dark region continues into the trachea |
# | Memory | small | **large**: activations grow with `x * y * z`, a whole CT does not fit on a GPU |
# | Training samples | 1 slice = 1 sample | **patches** cut from a volume (e.g. 96 x 96 x 64), many per patient |
# | Inference | one forward pass | **sliding window** over the volume and stitching of the overlapping outputs |
# | Parameters | 3 x 3 kernels | 3 x 3 x 3 kernels: about 3x more weights per convolution |
# | Voxels | pixels (mm in-plane) | voxels with **three** sizes (mm), often not equal: **anisotropy** |
#
# **Anisotropy.** A LUNA16 scan has an in-plane pixel size of 0.5-0.9 mm but a slice thickness of
# 0.6 up to 2.5 mm. A `3 x 3 x 3` kernel would then "see" 2 mm in x/y but 7.5 mm in z: not the same
# anatomy in each direction. The usual remedies are (i) **resample to isotropic voxels** (what we do:
# `Spacingd`), or (ii) use anisotropic kernels and strides (what nnU-Net does for very anisotropic data).
#
# **Patches.** We train on random 3D crops. To avoid a model that only sees empty background, we
# sample patches centred on lung voxels and on background voxels in a fixed ratio
# (`RandCropByPosNegLabeld`). The price: a patch sees only a part of the chest, so the model has
# a *limited context* (important for distinguishing right from left lung, see the challenge).

# %%
# A quick feeling for the numbers (float32 = 4 bytes per voxel). One single input volume, before any network:
for name, shape in {"our low-res volume": (140, 140, 120), "a native LUNA16 CT": (512, 512, 300),
                    "a 3D training patch": ROI if not FAST else (96, 96, 64)}.items():
    n = int(np.prod(shape))
    print(f"{name:22s} {str(shape):16s} {n / 1e6:7.2f} M voxels  = {n * 4 / 1e6:8.1f} MB (input only)")
print("A U-Net keeps 16-256 feature maps of this size alive for the backward pass: x 100-1000.")

# %% [markdown]
# ### 🔍 Explore
# * A 2D slice of 256 x 256 has 65 536 pixels. How many voxels has a 96 x 96 x 64 patch? And a native CT? Do the arithmetic before looking at the next section.
# * Why is "downsampling the whole CT to fit in memory" not the same as "cropping patches"? Think of the size of a small nodule (5 mm) at 2.5 mm voxels.

# %% [markdown]
# ## 2. Looking at a volume
#
# Before any model: **look at the data**. We load one raw volume with `nibabel` and check what the
# files say about orientation and voxel size.

# %%
raw_img = nib.load(train_files[0]["image"])
raw_lbl = nib.load(train_files[0]["label"])
raw = np.asarray(raw_img.dataobj)          # int16, HU
raw_l = np.asarray(raw_lbl.dataobj)
print("shape:", raw.shape, "| dtype:", raw.dtype, "| HU range:", raw.min(), raw.max())
print("voxel size (mm):", raw_img.header.get_zooms())
print("axis codes (direction in which each array axis grows):", nib.aff2axcodes(raw_img.affine))
print("label values and voxel counts:", dict(zip(*np.unique(raw_l, return_counts=True))))

# %% [markdown]
# `('L', 'P', 'S')` means: array axis 0 grows towards the patient's **L**eft, axis 1 towards **P**osterior,
# axis 2 towards **S**uperior. This is the **LPS+** convention of DICOM/ITK (and of LUNA16). MONAI tutorials
# (and nnU-Net) use **RAS+** instead. We will have to convert, and to be careful with left/right.
#
# We write two small helpers that always show the **radiological view** (anterior on top, patient's right on the
# left of the screen) for an array in **RAS** order: three planes, and a montage of axial slices.

# %%
def ras_planes(a, centre):
    """Axial, coronal and sagittal 2D views of a (X, Y, Z) array in RAS order, in radiological convention.
    Use with origin="lower" in imshow."""
    x, y, z = centre
    return {"axial": a[::-1, :, z].T,          # anterior on top, patient's right on the left of the screen
            "coronal": a[::-1, y, :].T,        # superior on top
            "sagittal": a[x, ::-1, :].T}       # anterior on the left

def show_planes(image, label=None, window=(-1000, 400), title=""):
    """Three orthogonal planes through the centre of the label (or of the volume). `image`: (X, Y, Z) in RAS."""
    image = np.asarray(image)
    if label is not None and np.asarray(label).any():
        centre = np.argwhere(np.asarray(label) > 0).mean(0).round().astype(int)
    else:
        centre = np.array(image.shape) // 2
    imgs = ras_planes(image, centre)
    lbls = ras_planes(np.asarray(label), centre) if label is not None else None
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, name in zip(axes, imgs):
        ax.imshow(imgs[name], cmap="gray", vmin=window[0], vmax=window[1], origin="lower")
        if lbls is not None:
            ax.imshow(np.ma.masked_where(lbls[name] == 0, lbls[name]), cmap="autumn", alpha=0.45,
                      origin="lower", vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"{name}  (x,y,z)={tuple(centre)}" if name == "axial" else name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_montage(image, label=None, n=12, window=(-1000, 400), title=""):
    """Montage of `n` evenly spaced axial slices of a (X, Y, Z) RAS array, with the label as an overlay."""
    image = np.asarray(image)
    zs = np.linspace(0, image.shape[2] - 1, n).round().astype(int)
    cols = 6 if n >= 6 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, z in zip(np.atleast_1d(axes).ravel(), zs):
        ax.imshow(image[::-1, :, z].T, cmap="gray", vmin=window[0], vmax=window[1], origin="lower")
        if label is not None:
            m = np.asarray(label)[::-1, :, z].T
            ax.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.45, origin="lower",
                      vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"z={z}", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# The raw array is LPS: reversing axes 0 and 1 gives RAS (the array is flipped, not rotated).
raw_ras, raw_l_ras = raw[::-1, ::-1, :], raw_l[::-1, ::-1, :]
show_planes(raw_ras, raw_l_ras, title=f"{train_files[0]['case_id']}: three planes (raw HU)")
show_montage(raw_ras, raw_l_ras, title="axial montage (labels: 1 right lung, 2 left lung, 3 trachea)")

# %% [markdown]
# ### 🔍 Explore
# * In the montage, where does the trachea (label 3) end and the two lungs start? In a *single* slice the trachea can look like just another dark circle: in 3D you can follow it.
# * Change `n` to 24: how many slices contain lung? Does that depend on `z`? (This is exactly the information a 2D model never sees.)
# * The `show_planes` helper uses `a[::-1, :, z]` for axial views of a RAS array. What would the image look like without the `::-1`? (Try it: which side of the patient is on the left of the screen then?)

# %% [markdown]
# ## 3. 3D dictionary transforms
#
# The idea is the same as in day 1, with three differences: orientation, voxel size and **patch sampling**.
#
# 1. `Orientationd(axcodes="RAS")` reorders/flips the array axes so that axis 0 grows to the patient's **R**ight,
#    axis 1 to **A**nterior, axis 2 to **S**uperior. Our files are LPS, so this *flips axes 0 and 1* (we saw it
#    above). Why do it? To have **one** convention for every dataset and every model (also for pretrained ones),
#    whatever the scanner or the converter wrote. The affine is updated, so world coordinates stay correct.
# 2. `Spacingd(pixdim=SPACING, mode=("bilinear", "nearest"))` resamples to a common voxel size. Use
#    **`nearest` for labels** (never interpolate class IDs!).
# 3. `ScaleIntensityRanged`: HU window `[-1000, 400]` -> `[0, 1]`, as in the 2D pipeline.
#
# The blanks below are the three arguments that matter. `lungs_only` is the same function as in the 2D course
# (labels 1 and 2 -> lungs, trachea -> background).

# %%
def lungs_only(x):
    """1 = right lung, 2 = left lung -> 1; everything else (incl. trachea) -> 0."""
    return ((x == 1) | (x == 2)).to(x.dtype)

keys = ["image", "label"]
load_tf = [LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)]
binary_label_tf = Lambdad(keys="label", func=lungs_only)

# %% [markdown]
# ### ✏️ Exercise 3.1 - orientation, resampling, intensity
#
# Complete the list `spatial_tf`. Use `SPACING` for the voxel size, and think about the interpolation mode
# of the image versus the label.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `axcodes` is a string with three letters, one per axis. MONAI tutorials use the letters of **RAS**.
#
# </details>
#
# <details>
# <summary>💡 Hint 2</summary>
#
# `mode` is a tuple with one entry per key: the image can be interpolated smoothly (`"bilinear"`, which is
# trilinear in 3D), the label must keep its integer values (`"nearest"`).
#
# </details>
#
# <details>
# <summary>💡 Hint 3 (almost the solution)</summary>
#
# `Orientationd(keys=keys, axcodes="RAS")` and `Spacingd(keys=keys, pixdim=SPACING, mode=("bilinear", "nearest"))`.
# For the HU window use `a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True`.
#
# </details>

# %%
spatial_tf = [
    #@solution
    Orientationd(keys=keys, axcodes="RAS"),
    Spacingd(keys=keys, pixdim=SPACING, mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys="image", a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    #@student
    #| Orientationd(keys=keys, axcodes="____"),                                           # TODO: RAS
    #| Spacingd(keys=keys, pixdim=____, mode=("bilinear", "____")),                       # TODO: SPACING; label mode
    #| ScaleIntensityRanged(keys="image", a_min=____, a_max=____, b_min=0.0, b_max=1.0, clip=True),  # TODO: HU window
    #@end
]
val_transforms = Compose(load_tf + [binary_label_tf] + spatial_tf + [EnsureTyped(keys=keys)])
sample = val_transforms(train_files[0])
image, label = sample["image"], sample["label"]
print("image", tuple(image.shape), image.dtype, "| label", tuple(label.shape), "| values:", label.unique().tolist())
print("axcodes now:", nib.aff2axcodes(image.affine.numpy()), "| voxel size:", [round(float(s), 2) for s in image.pixdim])

# %%
# Check 3.1
assert nib.aff2axcodes(image.affine.numpy()) == ("R", "A", "S"), "Orientationd should give RAS"
assert np.allclose([float(s) for s in image.pixdim], SPACING, atol=1e-3), "voxel size should be SPACING"
assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0, "HU window should map to [0, 1] with clipping"
assert set(label.unique().tolist()) <= {0, 1}, "label must stay binary (nearest interpolation!)"
print("✅ Exercise 3.1 passed")

# %%
# What did Orientationd do to the array? Compare with the raw LPS array: it is a flip of axes 0 and 1.
only_orient = Compose(load_tf + [Orientationd(keys=keys, axcodes="RAS")])(train_files[0])
print("identical to raw[::-1, ::-1, :]:", np.array_equal(np.asarray(only_orient["image"][0]), raw[::-1, ::-1, :]))
show_planes(image[0], label[0], window=(0, 1), title="after the pre-processing (RAS, resampled, scaled): label = both lungs")

# %% [markdown]
# > ⚠️ **Pitfall (orientation):** a model trained on RAS arrays and fed an LPS array (or the reverse) does not
# > crash. It just gives bad results, and on real chests it can confuse **left and right**. Always
# > orient explicitly, and read the orientation from the file, not from your expectations.
#
# ### Patch sampling
#
# Training on whole volumes is impossible on a free GPU. We cut random patches:
#
# 1. `CropForegroundd` removes the empty border around the body (less data, fewer useless patches).
# 2. `SpatialPadd` pads a volume that is smaller than the patch (otherwise the crop would fail).
# 3. `RandCropByPosNegLabeld` takes `num_samples` random patches per volume: centred on a **foreground** voxel
#    with probability `pos / (pos + neg)`, otherwise on a **background** voxel.
#
# ### ✏️ Exercise 3.2 - patch sampler
#
# Complete `RandCropByPosNegLabeld`: patch size `ROI`, a 1:1 ratio of lung-centred and background-centred
# patches, and `PATCHES_PER_VOLUME` patches per volume.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `spatial_size` is the patch size, `pos` and `neg` are *relative weights* (1 and 1 = 50% / 50%), and `num_samples`
# says how many patches come out of one volume.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `spatial_size=ROI, pos=1, neg=1, num_samples=PATCHES_PER_VOLUME`
#
# </details>

# %%
def body_foreground(x):
    """Everything brighter than air after the HU window (0 = air or padding)."""
    return x > 0.02

patch_tf = [
    CropForegroundd(keys=keys, source_key="image", select_fn=body_foreground, margin=2, allow_smaller=True),
    SpatialPadd(keys=keys, spatial_size=ROI),
    RandCropByPosNegLabeld(
        keys=keys, label_key="label", image_key="image",
        #@solution
        spatial_size=ROI, pos=1, neg=1, num_samples=PATCHES_PER_VOLUME,
        #@student
        #| spatial_size=____, pos=____, neg=____, num_samples=____,   # TODO: patch size, 1:1 ratio, patches per volume
        #@end
    ),
]
# Random flips: only for the *binary* task (see the question below). Left/right = axis 0 in RAS.
flip_tf = RandFlipd(keys=keys, prob=0.5, spatial_axis=0)

sampler = Compose(load_tf + [binary_label_tf] + spatial_tf + patch_tf)
patches = sampler(train_files[0])
print(f"{len(patches)} patches of shape {tuple(patches[0]['image'].shape)}")

# %%
# Check 3.2
assert isinstance(patches, list) and len(patches) == PATCHES_PER_VOLUME, "one dict per patch expected"
assert all(tuple(p["image"].shape) == (1, *ROI) for p in patches), "every patch must have the shape of ROI"
fg_fraction = np.mean([float(p["label"].sum() > 0) for p in patches])
print(f"fraction of patches that contain lung: {fg_fraction:.2f}")
print("✅ Exercise 3.2 passed")

# %%
# Look at the patches: middle axial slice of each one, with its label.
fig, axes = plt.subplots(1, len(patches), figsize=(3 * len(patches), 3))
for ax, p in zip(np.atleast_1d(axes), patches):
    zmid = p["image"].shape[-1] // 2
    ax.imshow(p["image"][0].numpy()[::-1, :, zmid].T, cmap="gray", vmin=0, vmax=1, origin="lower")
    m = p["label"][0].numpy()[::-1, :, zmid].T
    ax.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.45, origin="lower", vmin=0, vmax=1)
    ax.set_title(f"lung voxels: {int(p['label'].sum())}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# > 🧠 **Question:** `RandFlipd(spatial_axis=0)` flips the patch left <-> right. It is a cheap and
# > popular augmentation. Is it safe for (a) the binary lungs task, (b) a task with *separate right and
# > left lung* labels? And is a flipped chest realistic?

# %% [markdown] tags=["solution-only"]
# **Answer.** (a) For the binary task the label does not distinguish the sides, so flipping is
# harmless for the mask. But a flipped chest is anatomically unusual (heart on the right, liver on the left:
# *situs inversus*), so it may waste model capacity or hurt if the test set is never flipped. It is mostly
# a regulariser, to be checked with an experiment. (b) With `right lung = 1` and `left lung = 2` the flipped
# image would show a **left** lung carrying the label "right": the labels are wrong. Either do not flip along
# the left-right axis, or flip and swap the two label values. Flipping along z (superior <-> inferior) is
# never realistic for a chest.

# %% [markdown]
# ### 🔍 Explore
# * Add `flip_tf` to the transforms in the next section by setting `USE_FLIP = True` and compare the validation Dice at the end (it is only a few lines).
# * Change `pos=1, neg=1` to `pos=3, neg=1`: what fraction of patches contains lung now? When can a *higher* background proportion help? (Think of false positives outside the body.)
# * What happens if you remove `SpatialPadd` and use a patch larger than the volume? (Run the sampler, read the error message.)

# %% [markdown]
# ## 4. A 3D U-Net
#
# The 2D U-Net from day 2:
#
# ```python
# UNet(spatial_dims=2, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
# ```
#
# MONAI networks are written for any number of spatial dimensions. **Going 3D is a one-argument change.**
#
# ### ✏️ Exercise 4.1 - the 3D model
#
# Create `model` (the 3D U-Net) and `model_2d_ref` (the 2D reference, only to compare sizes).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The only thing that changes is `spatial_dims`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)`
#
# </details>

# %%
unet_kwargs = dict(in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
#@solution
model = UNet(spatial_dims=3, **unet_kwargs).to(device)
#@student
#| model = UNet(spatial_dims=____, **unet_kwargs).to(device)   # TODO: 3D U-Net
#@end
model_2d_ref = UNet(spatial_dims=2, **unet_kwargs)

def n_params(net):
    return sum(p.numel() for p in net.parameters())

print(f"2D U-Net: {n_params(model_2d_ref):,} parameters | 3D U-Net: {n_params(model):,} parameters")

# %%
# Check 4.1
x = torch.zeros(1, 1, *ROI, device=device)
assert model(x).shape == (1, 2, *ROI), "a 3D U-Net maps (B, 1, X, Y, Z) to (B, 2, X, Y, Z)"
assert any(p.ndim == 5 for p in model.parameters()), "3D convolution kernels have 5 dimensions: (out, in, kx, ky, kz)"
print("✅ Exercise 4.1 passed")

# %% [markdown]
# The number of parameters grows less than you may expect. **What really grows is memory.** We can measure it
# without a GPU: PyTorch's `meta` device runs a forward pass on tensors that have a shape but no data, and we add up
# the size of every convolution output (the values that must be kept for backpropagation).

# %%
def conv_output_gb(net, input_shape):
    """Size in GB (float32) of all convolution outputs of one forward pass; measured on the 'meta' device."""
    net = copy.deepcopy(net).to("meta").eval()
    total = [0]
    for m in net.modules():
        if isinstance(m, torch.nn.modules.conv._ConvNd):
            m.register_forward_hook(lambda _m, _i, out: total.__setitem__(0, total[0] + out.numel()))
    with torch.no_grad():
        net(torch.zeros(*input_shape, device="meta"))
    return total[0] * 4 / 1e9

cases = {
    "2D  training batch   16 x 256 x 256":      (model_2d_ref, (16, 1, 256, 256)),
    "3D  patch batch       4 x 96 x 96 x 64":   (model, (4, 1, 96, 96, 64)),
    "3D  low-res volume    1 x 144 x 144 x 128": (model, (1, 1, 144, 144, 128)),
    "3D  native CT         1 x 512 x 512 x 304": (model, (1, 1, 512, 512, 304)),
}
for name, (net, shape) in cases.items():
    print(f"{name:40s} conv outputs: {conv_output_gb(net, shape):7.2f} GB")
print("(sizes rounded to multiples of 16, as the U-Net with 4 strides of 2 requires)")
print("\nReal training needs roughly 3-5x more (normalisation, activations, gradients, optimizer state).")

# %% [markdown]
# A free T4 GPU has 16 GB: a patch batch fits, a native CT does not even fit for inference. That is
# why 3D medicine runs on patches, and uses a **sliding window** at inference time (next section).
#
# > 📚 **Read more:** the 3D U-Net paper [Cicek et al., 2016](https://arxiv.org/abs/1606.06650) and V-Net
# > [arXiv 1606.04797](https://arxiv.org/abs/1606.04797).

# %% [markdown]
# ## 5. Sliding-window inference
#
# At inference we cannot feed a whole volume (memory!) and we do not want to crop it (we would lose
# anatomy). `monai.inferers.sliding_window_inference` slides a window of `roi_size` over the volume, runs the model
# on `sw_batch_size` windows at a time, and **blends the overlapping predictions** (a Gaussian-weighted
# average by default would be `mode="gaussian"`; the default is a constant weighting).
#
# ### ✏️ Exercise 5.1 - choose the sliding-window parameters
#
# * `SW_ROI`: window size. What is the natural choice, given how the network was trained?
# * `SW_BATCH_SIZE`: how many windows per forward pass (memory vs speed). Try 4.
# * `OVERLAP`: fraction of overlap between neighbouring windows, in `[0, 1)`. 0.5 is a common value (0.25 is faster,
#   0.5-0.75 smoother at the borders of the windows).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# A network is happiest with the patch size it saw during training.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `SW_ROI = ROI`, `SW_BATCH_SIZE = 4`, `OVERLAP = 0.5`
#
# </details>

# %%
#@solution
SW_ROI = ROI
SW_BATCH_SIZE = 4
OVERLAP = 0.5
#@student
#| SW_ROI = ____          # TODO: window size (tuple of 3)
#| SW_BATCH_SIZE = ____   # TODO: windows per forward pass
#| OVERLAP = ____         # TODO: overlap between windows, in [0, 1)
#@end

# An untrained model on a whole validation volume: we only check shapes and that the pipeline runs.
val_sample = val_transforms(val_files[0])
x = val_sample["image"].as_tensor()[None].to(device)           # (1, 1, X, Y, Z)
model.eval()
with torch.no_grad():
    out = sliding_window_inference(inputs=x, roi_size=SW_ROI, sw_batch_size=SW_BATCH_SIZE, predictor=model, overlap=OVERLAP)
print("input", tuple(x.shape), "-> output", tuple(out.shape))

# %%
# Check 5.1
assert len(SW_ROI) == 3 and SW_BATCH_SIZE >= 1 and 0 <= OVERLAP < 1, "roi: 3 numbers, batch >= 1, overlap in [0, 1)"
assert out.shape == (1, 2, *x.shape[2:]), "the output has the size of the whole input volume, with one channel per class"
if tuple(SW_ROI) != tuple(ROI):
    print("note: SW_ROI differs from the training patch size; that works, but is usually not what you want.")
print("✅ Exercise 5.1 passed")

# %% [markdown]
# ### 🔍 Explore
# * Time the call above with `OVERLAP = 0` and `OVERLAP = 0.75`: how does the cost scale?
# * With `OVERLAP = 0` look at the output logits (`out[0, 1, :, :, z]`) near the borders of the windows (every 32 voxels in x): do you see the seams? (They are clearer after training.)
# * Try `mode="gaussian"` in `sliding_window_inference`: windows are weighted more in the centre.

# %% [markdown]
# ## 6. Training on patches
#
# The training loop is the same as on day 2, except that the data loader returns **patches**. `DiceCELoss` combines
# Dice and cross-entropy (more stable on tiny structures). Validation is run on **whole volumes** with the sliding window
# of section 5. Everything is wrapped in functions, because we will train more than one model in this notebook.

# %%
USE_FLIP = False   # see the explore box in section 3

def make_loaders(label_tf, flip=False):
    """Training loader (random patches) and validation loader (whole volumes). One volume per step."""
    tf = load_tf + [label_tf] + spatial_tf
    train_tf = Compose(tf + patch_tf + ([flip_tf] if flip else []) + [EnsureTyped(keys=keys)])
    val_tf = Compose(tf + [EnsureTyped(keys=keys)])
    # CacheDataset keeps the result of the deterministic transforms in RAM; the random patches are drawn on the fly.
    train_ds = CacheDataset(train_files, train_tf, cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds = CacheDataset(val_files, val_tf, cache_rate=1.0, num_workers=NUM_WORKERS)
    return (DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=NUM_WORKERS),
            DataLoader(val_ds, batch_size=1, num_workers=NUM_WORKERS))

train_loader, val_loader = make_loaders(binary_label_tf, flip=USE_FLIP)
batch = next(iter(train_loader))
print("one training step sees", tuple(batch["image"].shape), "(patches, channel, x, y, z)")

# %%
def validate(net, loader, n_classes):
    """Mean Dice per foreground class over the whole validation volumes (sliding window)."""
    net.eval()
    metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred, post_label = AsDiscrete(argmax=True, to_onehot=n_classes), AsDiscrete(to_onehot=n_classes)
    with torch.no_grad():
        for b in loader:
            x, y = b["image"].to(device), b["label"].to(device)
            logits = sliding_window_inference(x, SW_ROI, SW_BATCH_SIZE, net, overlap=OVERLAP)
            metric(y_pred=[post_pred(o) for o in decollate_batch(logits)], y=[post_label(t) for t in decollate_batch(y)])
    per_class = metric.aggregate().cpu()
    metric.reset()
    return per_class

def save_model(net, name, arch, arch_kwargs, dice):
    """Course contract (model.pt + config.json). Stored in models_3d/, NOT in models/: the 2D notebooks and the app
    use models/ and must never pick up a 3D model by mistake."""
    out = WORK_DIR / "models_3d" / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "model.pt")
    cfg = {"name": name, "arch": arch, "arch_kwargs": arch_kwargs,
           "preprocessing": {"hu_min": -1000, "hu_max": 400, "spacing": list(SPACING), "orientation": "RAS", "roi": list(ROI)},
           "metrics": {"val_dice": round(float(dice), 4)}}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return out

def run_training(net, loader_pair, n_classes, name, arch, arch_kwargs, epochs=MAX_EPOCHS, lr=1e-3):
    train_dl, val_dl = loader_pair
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        net.train()
        running = 0.0
        for b in train_dl:
            x, y = b["image"].to(device), b["label"].to(device)
            optimizer.zero_grad()
            loss = loss_function(net(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        dice = validate(net, val_dl, n_classes)
        mean_dice = float(torch.nanmean(dice))
        history.append((running / len(train_dl), mean_dice))
        if mean_dice > best:
            best = mean_dice
            save_model(net, name, arch, arch_kwargs, best)
        print(f"epoch {epoch:3d}/{epochs}  loss {history[-1][0]:.4f}  val dice {mean_dice:.4f}")
    # Reload the best weights, as on day 2
    net.load_state_dict(torch.load(WORK_DIR / "models_3d" / name / "model.pt", map_location=device, weights_only=True))
    return history, best

history, best_dice = run_training(model, (train_loader, val_loader), 2, "unet3d-lungs", "UNet",
                                  {**unet_kwargs, "spatial_dims": 3, "channels": list(unet_kwargs["channels"]),
                                   "strides": list(unet_kwargs["strides"])})
print(f"best validation Dice (3D, binary lungs): {best_dice:.4f}")

# %%
hist = np.array(history)
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].plot(hist[:, 0]); axes[0].set_title("training loss (DiceCE)"); axes[0].set_xlabel("epoch")
axes[1].plot(hist[:, 1]); axes[1].set_title("validation Dice (whole volumes)"); axes[1].set_xlabel("epoch")
plt.tight_layout()
plt.show()

# %% [markdown]
# > ⚠️ **Pitfall:** the validation metric is computed on **whole volumes** with sliding window; the training loss on
# > **patches** with a 50% background/foreground sampling. They are *not* comparable numbers (the patch loss is
# > made of "harder" patches). Never judge a 3D model by the training loss alone.
#
# ### 🔍 Explore
# * Set `lr=1e-4` or `lr=1e-2` in `run_training` and re-train (on a free GPU, a few minutes with 3D): what happens to the curve?
# * Look at `batch["image"].shape[0]`: why do we get `PATCHES_PER_VOLUME` patches with `batch_size=1`?

# %% [markdown]
# ## 7. 3D evaluation: Dice and lung volume in litres
#
# **Dice** in 3D is the same formula as in 2D, summed over *all voxels* of the volume:
# `Dice = 2 |P ∩ G| / (|P| + |G|)`.
#
# The 3D mask opens a clinically useful measurement that a single slice cannot give: the **lung volume**.
# A voxel has a volume of `sx * sy * sz` mm³, and 1 litre = 1 000 000 mm³. Volumetry (and lung density) is
# used for emphysema, lung resection planning or fibrosis follow-up. **The voxel size must come from the image,
# never from a guess**: a wrong spacing is a wrong diagnosis.
#
# ### ✏️ Exercise 7.1 - Dice and litres
#
# Write the two functions below with plain numpy/torch operations (no MONAI).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The intersection of two binary masks is `(pred * gt).sum()`. The voxel volume is the product of the three spacings.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `litres = mask.sum() * np.prod(spacing_mm) / 1e6`
#
# </details>

# %%
def dice_3d(pred, gt, eps=1e-8):
    """Dice between two binary 3D masks (arrays or tensors of 0/1)."""
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    #@solution
    intersection = (pred & gt).sum()
    return float(2 * intersection / (pred.sum() + gt.sum() + eps))
    #@student
    #| intersection = ____                     # TODO: number of voxels that are 1 in both masks
    #| return float(____)                      # TODO: 2 * intersection / (|P| + |G|), with eps in the denominator
    #@end

def lung_volume_litres(mask, spacing_mm):
    """Volume of a binary 3D mask in litres, given the voxel size (sx, sy, sz) in mm."""
    #@solution
    return float(np.asarray(mask).astype(bool).sum() * np.prod(spacing_mm) / 1e6)
    #@student
    #| return float(____)                      # TODO: number of voxels * volume of one voxel (mm^3) -> litres
    #@end

# %%
# Check 7.1
a = np.zeros((10, 10, 10)); a[:5] = 1
b = np.zeros((10, 10, 10)); b[:5] = 1
c = np.zeros((10, 10, 10)); c[5:] = 1
assert abs(dice_3d(a, b) - 1.0) < 1e-6 and dice_3d(a, c) < 1e-6, "Dice: 1 for identical masks, 0 for disjoint ones"
assert abs(lung_volume_litres(np.ones((10, 10, 10)), (10, 10, 10)) - 1.0) < 1e-9, "1000 voxels of 1 cm^3 are 1 litre"
print("✅ Exercise 7.1 passed")

# %%
@torch.no_grad()
def predict_volume(net, image):
    """Whole-volume prediction (X, Y, Z) of class indices, by sliding window. `image`: (1, X, Y, Z) tensor."""
    net.eval()
    logits = sliding_window_inference(image.as_tensor()[None].to(device), SW_ROI, SW_BATCH_SIZE, net, overlap=OVERLAP)
    return logits.argmax(1)[0].cpu().numpy()

val_tf_binary = Compose(load_tf + [binary_label_tf] + spatial_tf + [EnsureTyped(keys=keys)])
rows, preds_3d = [], {}
for f in val_files:
    item = val_tf_binary(f)
    gt = item["label"][0].numpy()
    pred = predict_volume(model, item["image"])
    preds_3d[f["case_id"]] = pred
    spacing = [float(s) for s in item["image"].pixdim]
    rows.append({"case": f["case_id"], "dice_3d": dice_3d(pred, gt),
                 "litres_gt": lung_volume_litres(gt, spacing), "litres_pred": lung_volume_litres(pred, spacing)})
results = pd.DataFrame(rows)
results["abs_error_%"] = 100 * (results.litres_pred - results.litres_gt).abs() / results.litres_gt
results.round(3)

# %% [markdown]
# On real adult CTs a total lung volume of about **3-6 litres** is plausible (it depends on size, sex and how deeply the
# patient inhaled). A value far outside this range is a red flag for the *spacing* or for the mask (e.g. the whole
# body segmented). On our synthetic data the numbers are meaningless, only the method counts.
#
# > 🧠 **Question:** Dice is 0.97 but the lung volume is 4% too small. Is that a contradiction? Which of the two numbers is
# > more useful for a clinician following a patient over time, and what would you check?

# %% [markdown] tags=["solution-only"]
# **Answer.** Not a contradiction: Dice is dominated by the large inner part of the lungs; a thin border
# error (partial-volume voxels at the pleura, vessels, the choice of the HU threshold of the silver-standard labels)
# changes a few percent of the volume but barely moves Dice. For follow-up the **volume (and its change)** is the
# clinical number, so we should check the *systematic bias* (Bland-Altman plot of predicted vs reference volume) and
# the repeatability, not only the overlap. The same voxel size must be used at every time point.

# %% [markdown]
# ## 8. The experiment: is 3D better than 2D slice by slice?
#
# Our 2D model from day 2 can also segment a volume: just run it on every axial slice and stack the results.
# Why bother with 3D, then? The 2D model has **no idea of the neighbouring slices**, so it is free to be inconsistent from
# one slice to the next. We will measure it.
#
# **Loading the 2D model.** We use the canonical functions of the course (the student's model if there is one,
# otherwise the teachers' one) and keep only models with `spatial_dims == 2`.

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

dirs_2d = [d for d in find_model_dirs() if json.loads((d / "config.json").read_text())["arch_kwargs"].get("spatial_dims") == 2]
if not dirs_2d:
    raise FileNotFoundError("No 2D model found. Finish notebook 03 (day 2) or use the teachers' model in DATA_DIR/pretrained.")
model_2d, cfg_2d = load_model(dirs_2d[0], device)
IMG_SIZE_2D = tuple(cfg_2d["preprocessing"]["img_size"])
print(f"2D model: {dirs_2d[0]}\n  trained on slices resized to {IMG_SIZE_2D}, HU window "
      f"[{cfg_2d['preprocessing']['hu_min']}, {cfg_2d['preprocessing']['hu_max']}]")

# %% [markdown]
# **Three things must match between the 2D model's training and our volume** (training-serving skew again):
#
# 1. **Intensity window**: same `[-1000, 400]` -> `[0, 1]` (our volumes are already scaled like that).
# 2. **Orientation**: the 2D slices were LPS (`img.T` is radiological). Our volume is RAS, so each slice must be
#    flipped in both x and y before the 2D model, and flipped back afterwards.
# 3. **Size**: each axial slice is resized to the model's `img_size` (read from `config.json`!), and the prediction back to the original size.
#
# ### ✏️ Exercise 8.1 - apply the 2D model to every axial slice
#
# Fill the three blanks: the size for the first resize, the size for the way back, and the dimension of the `argmax`.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The model wants `IMG_SIZE_2D`; the output must have the size of the original slice, `sl.shape`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# The class dimension of a tensor `(B, C, H, W)` is dimension 1: `prob.argmax(dim=1)`.
#
# </details>

# %%
@torch.no_grad()
def predict_2d_stack(net2d, volume, img_size=IMG_SIZE_2D, chunk=16):
    """Segment a (X, Y, Z) RAS volume in [0, 1] slice by slice with a 2D model. Returns (X, Y, Z) class indices."""
    net2d.eval()
    out = np.zeros(volume.shape, dtype=np.uint8)
    for z0 in range(0, volume.shape[2], chunk):
        sl = volume[:, :, z0:z0 + chunk].permute(2, 0, 1)[:, None]          # (n, 1, X, Y), slices as a batch
        sl = torch.flip(sl, dims=(2, 3))                                    # RAS -> LPS, as in the 2D training data
        #@solution
        x = F.interpolate(sl, size=img_size, mode="bilinear", align_corners=False)
        prob = net2d(x.to(device)).softmax(dim=1)
        prob = F.interpolate(prob, size=sl.shape[2:], mode="bilinear", align_corners=False)
        pred = prob.argmax(dim=1).cpu()
        #@student
        #| x = F.interpolate(sl, size=____, mode="bilinear", align_corners=False)               # TODO: size the model expects
        #| prob = net2d(x.to(device)).softmax(dim=1)
        #| prob = F.interpolate(prob, size=____, mode="bilinear", align_corners=False)          # TODO: back to the slice size
        #| pred = prob.argmax(dim=____).cpu()                                                   # TODO: the class dimension
        #@end
        pred = torch.flip(pred, dims=(1, 2))                                # LPS -> RAS
        out[:, :, z0:z0 + chunk] = pred.permute(1, 2, 0).numpy()
    return out

item = val_tf_binary(val_files[0])
vol, gt = item["image"][0].as_tensor(), item["label"][0].numpy()
pred_2d = predict_2d_stack(model_2d, vol)
print("2D stack prediction:", pred_2d.shape, "| Dice vs reference:", round(dice_3d(pred_2d, gt), 4))

# %%
# Check 8.1
assert pred_2d.shape == tuple(vol.shape), "the stacked prediction must have the size of the volume"
assert set(np.unique(pred_2d)) <= {0, 1}, "binary lungs: values 0/1"
print("✅ Exercise 8.1 passed")

# %% [markdown]
# > 🧠 **Question:** why do we flip the slices before and after the 2D model? What would happen on a real chest if we skipped it?

# %% [markdown] tags=["solution-only"]
# **Answer.** The 2D network was trained on LPS slices (anterior up, patient's left on the right of the array). A RAS slice
# is the same picture mirrored in both axes: rotated by 180 degrees. A CNN is *not* rotation invariant: it expects the
# spine at the bottom and the heart on one side. On a real chest the model would still produce something
# lung-shaped, but with more errors around the mediastinum, and nothing would crash: a silent skew. Always verify with a
# plot that the input of the model looks like the training data.

# %% [markdown]
# ### Compare: 3D model vs 2D stack
# Dice, the typical artefact (inconsistency between adjacent slices) and a **coronal view**, which is the plane
# the 2D model never sees. In each row of a coronal image the neighbouring pixels come from *different* axial
# slices, so any inconsistency between slices becomes a jagged edge.

# %%
def adjacent_slice_dice(mask):
    """Mean Dice between slice z and z+1 over the axial slices that contain a mask. 1.0 = perfectly smooth in z."""
    mask = np.asarray(mask).astype(bool)
    vals = [dice_3d(mask[:, :, z], mask[:, :, z + 1]) for z in range(mask.shape[2] - 1)
            if mask[:, :, z].any() or mask[:, :, z + 1].any()]
    return float(np.mean(vals)) if vals else float("nan")

rows = []
for f in val_files:
    item = val_tf_binary(f)
    gt = item["label"][0].numpy()
    p3d = preds_3d[f["case_id"]]
    p2d = predict_2d_stack(model_2d, item["image"][0].as_tensor())
    spacing = [float(s) for s in item["image"].pixdim]
    rows.append({"case": f["case_id"],
                 "dice 2D stack": dice_3d(p2d, gt), "dice 3D": dice_3d(p3d, gt),
                 "z-smoothness GT": adjacent_slice_dice(gt), "z-smoothness 2D": adjacent_slice_dice(p2d),
                 "z-smoothness 3D": adjacent_slice_dice(p3d),
                 "litres GT": lung_volume_litres(gt, spacing), "litres 2D": lung_volume_litres(p2d, spacing),
                 "litres 3D": lung_volume_litres(p3d, spacing)})
comparison = pd.DataFrame(rows).set_index("case")
comparison.round(3)

# %%
# One case in detail: coronal view (reference | 2D stack | 3D) and lung area per axial slice.
f = val_files[0]
item = val_tf_binary(f)
img_np, gt = item["image"][0].numpy(), item["label"][0].numpy()
p2d, p3d = predict_2d_stack(model_2d, item["image"][0].as_tensor()), preds_3d[f["case_id"]]
y_mid = int(np.argwhere(gt > 0)[:, 1].mean())

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (name, m) in zip(axes[:3], {"reference": gt, "2D slice by slice": p2d, "3D model": p3d}.items()):
    ax.imshow(img_np[::-1, y_mid, :].T, cmap="gray", vmin=0, vmax=1, origin="lower")
    mm = m[::-1, y_mid, :].T
    ax.imshow(np.ma.masked_where(mm == 0, mm), cmap="autumn", alpha=0.5, origin="lower", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"coronal: {name}"); ax.axis("off")
area = lambda m: m.sum(axis=(0, 1)) * SPACING[0] * SPACING[1] / 100      # cm^2 per axial slice
axes[3].plot(area(gt), "k-", label="reference"); axes[3].plot(area(p2d), "r.-", label="2D stack"); axes[3].plot(area(p3d), "b.-", label="3D")
axes[3].set_xlabel("axial slice z"); axes[3].set_ylabel("lung area (cm$^2$)"); axes[3].legend(); axes[3].set_title("area per slice")
plt.tight_layout()
plt.show()

# %% [markdown]
# *(In FAST mode, with 4 tiny synthetic volumes at 8 mm, the 3D model is under-trained and the numbers say nothing about
# 2D versus 3D. On the real dataset, run the comparison on a model trained for the full number of epochs.)*
#
# **How to read it.** On real CTs, the 2D model is usually *good* on average (the lung is easy), and the Dice
# difference to 3D is small. The 3D advantages are in the details:
#
# * a **smoother** result along z (the `z-smoothness` column: closer to the reference value is better) and a
#   less jagged coronal edge;
# * fewer **isolated false positives** in single slices (e.g. the stomach bubble or bowel gas below the lungs), because the 3D model
#   knows that nothing connects them to the lungs;
# * better **volumes** (small slice-wise errors add up, and the slices at the lung apex and base are the hardest).
#
# The 3D model also has costs: more memory, more training time, fewer independent training samples (60 patients
# are 60 volumes, not 60 slices), and patches that lose global context.
#
# > ⚠️ **Pitfall (fair comparison):** the 2D model of day 2 has seen *one slice* of the patients of subsets 0-6. If a volume
# > you compare on belongs to those subsets (as in our tiny synthetic validation set, which is carved out of the training subsets),
# > the 2D model has an advantage. With the real dataset use the validation (7-8) or test (9) subsets, which are unseen by both.
#
# > 🧠 **Question:** the 2D model was trained on 512 x 512 slices resized to `img_size`, at 0.5-0.9 mm per pixel. We feed it slices from a volume that
# > was resampled to 3 mm voxels and resized. Name two reasons why the 2D result may be *worse* than it should be, which have nothing to do with 2D versus 3D.

# %% [markdown] tags=["solution-only"]
# **Answer.** (1) *Scale mismatch*: the apparent size of the anatomy in pixels differs from the training data, if the field of
# view or the resizing differ (our volumes cover the same field of view, but after cropping they would not). (2) *Resolution /
# texture mismatch*: the low-res volume is blurrier than the original slice; the 2D model has never seen interpolated, 3-mm images
# (domain shift). (3) Orientation flip errors, or a different HU window, are other silent causes. This is exactly the training-serving
# skew of notebook 06.

# %% [markdown]
# ### 🔍 Explore
# * Remove the two `torch.flip` lines in `predict_2d_stack` and recompute the Dice. By how much does it change on the synthetic data? What would you expect on a real chest?
# * Compute the difference between the 2D and the 3D volume in millilitres for each case. Is the error biased (always the same sign)?
# * Show the *sagittal* view instead of the coronal one.

# %% [markdown]
# ## 9. 🚀 Challenges (additional)
#
# ### 🚀 Challenge 9.1 - multi-class: right and left lung in 3D
#
# In 2D, telling the **right lung from the left lung** is hard for a model that sees a patch: they look alike. The
# distinguishing information is *global* (the heart is on the left, the liver on the right), and a
# **3D context** with a large field of view helps. Train a 3-class model (0 = background, 1 = right lung, 2 = left lung; the
# trachea is merged with the background) and look at Dice **per class**.
#
# Steps: (1) a new `label_tf` that keeps labels 1 and 2 and sets 3 to 0, (2) loaders **without the left/right flip**
# (why? see section 3), (3) a `UNet` with `out_channels=3`, (4) `run_training(..., n_classes=3)`, (5) per-class Dice
# on the validation set.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `lambda x: x * (x < 3)` keeps 0, 1, 2 and removes 3 (use a named function, so that it can be pickled by data loader workers).
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `make_loaders(Lambdad(keys="label", func=right_left_only), flip=False)`, `UNet(spatial_dims=3, **{**unet_kwargs, "out_channels": 3})`, and `validate(net, val_loader, 3)` returns one Dice per foreground class.
#
# </details>

# %%
#@solution
def right_left_only(x):
    """Keep 1 = right lung, 2 = left lung; trachea (3) becomes background."""
    return x * (x < 3)

loaders_mc = make_loaders(Lambdad(keys="label", func=right_left_only), flip=False)
kwargs_mc = {**unet_kwargs, "out_channels": 3, "spatial_dims": 3}
model_mc = UNet(**kwargs_mc).to(device)
_, best_mc = run_training(model_mc, loaders_mc, 3, "unet3d-right-left", "UNet",
                          {**kwargs_mc, "channels": list(unet_kwargs["channels"]), "strides": list(unet_kwargs["strides"])},
                          epochs=MAX_EPOCHS)
dice_mc = validate(model_mc, loaders_mc[1], 3)
print(f"Dice right lung: {dice_mc[0]:.3f} | left lung: {dice_mc[1]:.3f} | mean {best_mc:.3f}")
#@student
#| # TODO: your code. Re-use make_loaders, run_training and validate from section 6.
#@end

# %% [markdown]
# ### 🚀 Challenge 9.2 - another 3D architecture: `SegResNet`
#
# `monai.networks.nets.SegResNet` is a residual encoder-decoder (winner of BraTS 2018) with GroupNorm: a strong, memory-friendly
# 3D baseline, and it is the model of the MONAI Auto3DSeg pipeline. Replace the U-Net by `SegResNet(spatial_dims=3, init_filters=16, in_channels=1, out_channels=2)`,
# train with the same loaders and compare the parameters, the validation Dice and the time per epoch with the 3D U-Net.
#
# <details>
# <summary>💡 Hint</summary>
#
# Reuse `train_loader`/`val_loader` of section 6 and `run_training(...)` with `arch="SegResNet"` and the matching `arch_kwargs`.
# The window size must be divisible by 8 for this network.
#
# </details>

# %%
#@solution
import time

segres_kwargs = dict(spatial_dims=3, init_filters=8 if FAST else 16, in_channels=1, out_channels=2)
model_sr = monai.networks.nets.SegResNet(**segres_kwargs).to(device)
t0 = time.time()
_, best_sr = run_training(model_sr, (train_loader, val_loader), 2, "segresnet-lungs", "SegResNet", segres_kwargs,
                          epochs=MAX_EPOCHS)
print(f"SegResNet: {n_params(model_sr):,} parameters, best val Dice {best_sr:.4f}, {(time.time() - t0) / MAX_EPOCHS:.1f} s/epoch "
      f"| 3D U-Net: {n_params(model):,} parameters, best val Dice {best_dice:.4f}")
#@student
#| # TODO: your code
#@end

# %% [markdown]
# ### 🚀 Pointer - deep supervision and `DynUNet`
# nnU-Net-like networks add **deep supervision**: the decoder emits a prediction at several resolutions and the loss is
# computed on all of them (the early layers receive a stronger gradient signal). In MONAI this is
# `monai.networks.nets.DynUNet(..., deep_supervision=True)` together with `monai.losses.DeepSupervisionLoss(DiceCELoss(...))`.
# `DynUNet` also takes **anisotropic kernel sizes and strides**, the right tool when you do not want to resample CTs with a thick
# slice spacing to an isotropic grid. See the [networks documentation](https://docs.monai.io/en/stable/networks.html).

# %% [markdown]
# ## 10. 🌶️ Going further (optional)
#
# ### 🌶️ Export a prediction and look at it in 3D Slicer or ITK-SNAP
# Always look at your predictions in a real viewer: you can scroll, change the window and overlay the mask.
# The prediction lives in the resampled RAS space of our pipeline; the **affine** that MONAI keeps in the
# `MetaTensor` maps it correctly into the patient's coordinates, so it lines up with the original CT in the viewer.
# Download both files and open them in [3D Slicer](https://www.slicer.org/) (drag and drop both files) or ITK-SNAP.

# %%
export_dir = WORK_DIR / "export"
export_dir.mkdir(parents=True, exist_ok=True)
case = val_files[0]["case_id"]
item = val_tf_binary(val_files[0])
affine = item["image"].affine.numpy()
hu_image = (item["image"][0].numpy() * 1400 - 1000).astype(np.int16)      # back to HU (window [-1000, 400])
nib.save(nib.Nifti1Image(hu_image, affine), export_dir / f"{case}_image.nii.gz")
nib.save(nib.Nifti1Image(preds_3d[case].astype(np.uint8), affine), export_dir / f"{case}_pred3d.nii.gz")
print("written:", sorted(p.name for p in export_dir.iterdir()))

# %% [markdown]
# ### 🌶️ Strong pretrained baselines
# Before training your own 3D model on a new problem, **compare with what exists**. If a pretrained model already gives
# 0.98 Dice on your data, you may not need to train anything:
#
# * [**nnU-Net**](https://github.com/MIC-DKFZ/nnUNet) ([arXiv 1809.10486](https://arxiv.org/abs/1809.10486)): a self-configuring pipeline that chooses patch size, spacing and architecture from your dataset's fingerprint. It is the baseline to beat in medical segmentation challenges.
# * **MONAI Auto3DSeg** (`monai.apps.auto3dseg.AutoRunner`, see the `auto3dseg` folder of the [MONAI tutorials](https://github.com/Project-MONAI/tutorials)): similar idea inside MONAI, with SegResNet/SwinUNETR/DiNTS.
# * [**TotalSegmentator**](https://github.com/wasserth/TotalSegmentator): pretrained 3D nnU-Net models for more than 100 structures of a CT, including the lung lobes.
# * [**lungmask**](https://github.com/JoHof/lungmask): U-Net models trained specifically for lung (and lobe) segmentation.
#
# The cell below is **starter code**. It runs only if you set `RUN_PRETRAINED = True` and the package is installed
# (`pip install lungmask` needs internet and downloads weights). Check the README of each project for the current API.
# On the course data you can compare with our silver-standard labels: remember that the LUNA16 labels are
# produced by an algorithm, so a disagreement does not always mean that the pretrained model is wrong.

# %%
RUN_PRETRAINED = False        # set to True if you have internet and pip access

if RUN_PRETRAINED:
    try:
        import SimpleITK as sitk
        from lungmask import LMInferer          # pip install lungmask

        sitk_img = sitk.ReadImage(val_files[0]["image"])
        seg = LMInferer(modelname="R231").apply(sitk_img)          # (z, y, x) array: 1 = right, 2 = left lung
        seg = np.transpose(seg, (2, 1, 0))                         # -> (x, y, z), the index order of the file
        gt_raw = np.asarray(nib.load(val_files[0]["label"]).dataobj)
        print("lungmask vs reference, Dice:", round(dice_3d(seg > 0, (gt_raw == 1) | (gt_raw == 2)), 4))
    except ImportError as err:
        print("lungmask / SimpleITK not installed:", err)
else:
    print("skipped (RUN_PRETRAINED = False)")

# %% [markdown]
# **Starter ideas**
# * Run TotalSegmentator (`TotalSegmentator -i <ct.nii.gz> -o <out_dir> --fast` on the command line) on the exported image of the previous cell and view the result in 3D Slicer.
# * Fine-tune a pretrained 3D model on our volumes instead of training from scratch: how many epochs does it need to match your Dice?
# * Plot the Bland-Altman diagram (difference vs mean) of lung volume between a model and the reference.

# %% [markdown]
# ## Wrap-up
# * 3D = one argument in MONAI, but **memory, patches, orientation and voxel size** change everything around the model.
# * Train on patches, **evaluate on whole volumes** with a sliding window.
# * A 3D mask gives clinical measurements that a slice cannot: **volume in litres** (voxel size from the image!).
# * The 2D-stack baseline is strong; 3D wins on consistency and context. Always **measure** the difference, and compare with a pretrained baseline before building your own.
#
# **Back to the main path:** your 2D model is what the web app on day 5 serves. A 3D model needs
# the full volume as input, so a 3D app would take a NIfTI or DICOM series upload instead of a single slice.
