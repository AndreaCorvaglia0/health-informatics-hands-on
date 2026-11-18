"""Regenerate the figures of the handbook from the course dataset.

    python make_figures.py /path/to/luna16-lung-slices
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from monai.transforms import (Compose, EnsureChannelFirstd, Lambdad, LoadImaged, RandAffined, Resized,
                              ScaleIntensityRanged)
from monai.utils import set_determinism

D = sys.argv[1]
plt.rcParams.update({"font.size": 9})
save = dict(dpi=200, bbox_inches="tight", pad_inches=0.05)

image = nib.load(f"{D}/images/case_0001.nii.gz").get_fdata()
mask = nib.load(f"{D}/labels/case_0001.nii.gz").get_fdata()
lung = (mask == 1) | (mask == 2)

# windows
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
for ax, (title, lo, hi) in zip(axes, [("Lung window (-1000 to 400)", -1000, 400),
                                      ("Soft tissue window (-160 to 240)", -160, 240),
                                      ("Bone window (-500 to 1500)", -500, 1500)]):
    ax.imshow(image.T, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
fig.savefig("figures/windows.png", **save)

# mask
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
axes[0].imshow(image.T, cmap="gray", vmin=-1000, vmax=400)
axes[0].set_title("Image", fontsize=8)
axes[1].imshow(lung.T, cmap="gray")
axes[1].set_title("Mask (1 = lung, 0 = background)", fontsize=8)
axes[2].imshow(image.T, cmap="gray", vmin=-1000, vmax=400)
axes[2].imshow(np.ma.masked_equal(lung, 0).T, cmap="autumn", alpha=0.4)
axes[2].set_title("Mask over the image", fontsize=8)
for ax in axes:
    ax.axis("off")
fig.savefig("figures/mask.png", **save)

# histogram
fig, ax = plt.subplots(figsize=(5, 2.2))
ax.hist(np.clip(image, -1100, 1000).ravel(), bins=150, color="0.35")
for x, text in [(-1000, "air"), (-800, "lung"), (40, "soft tissue"), (600, "bone")]:
    ax.annotate(text, (x, ax.get_ylim()[1] * 0.82), ha="center", fontsize=8)
ax.set_xlabel("Hounsfield units")
ax.set_ylabel("pixels")
ax.set_yticks([])
fig.savefig("figures/histogram.png", **save)

# augmentation
set_determinism(3)
keys = ["image", "label"]
pre = Compose([
    LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys),
    Lambdad(keys="label", func=lambda x: ((x == 1) | (x == 2)).float()),
    ScaleIntensityRanged(keys="image", a_min=-1000, a_max=400, b_min=0, b_max=1, clip=True),
    Resized(keys=keys, spatial_size=(256, 256), mode=("bilinear", "nearest")),
])
sample = pre({"image": f"{D}/images/case_0001.nii.gz", "label": f"{D}/labels/case_0001.nii.gz"})
aug = RandAffined(keys=keys, prob=1, rotate_range=0.2, scale_range=0.1, translate_range=10,
                  mode=("bilinear", "nearest"), padding_mode="zeros")
fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.1))
for i, ax in enumerate(axes):
    t = sample if i == 0 else aug(sample)
    ax.imshow(np.asarray(t["image"][0]).T, cmap="gray")
    ax.imshow(np.ma.masked_equal(np.asarray(t["label"][0]), 0).T, cmap="autumn", alpha=0.3)
    ax.set_title("original" if i == 0 else f"augmented {i}", fontsize=8)
    ax.axis("off")
fig.savefig("figures/augmentation.png", **save)

# learning curves (schematic)
e = np.arange(1, 61)
train_loss = 0.45 * np.exp(-e / 9) + 0.05 * np.exp(-e / 60) + 0.02
val_dice = 0.97 - 0.5 * np.exp(-e / 6) - 0.0009 * np.clip(e - 30, 0, None)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3))
axes[0].plot(e, train_loss, color="k")
axes[0].set_title("Training loss", fontsize=9)
axes[0].set_xlabel("epoch")
axes[1].plot(e, val_dice, color="k")
axes[1].set_title("Validation Dice", fontsize=9)
axes[1].set_xlabel("epoch")
axes[1].axvline(30, ls=":", color="0.4")
axes[1].annotate("best model", (30, 0.6), ha="center", fontsize=8)
axes[1].annotate("overfitting", (48, 0.86), ha="center", fontsize=8)
fig.tight_layout()
fig.savefig("figures/curves.png", **save)

# nodules
meta = pd.read_csv(f"{D}/nodules/metadata.csv")
fig, axes = plt.subplots(1, 6, figsize=(7.5, 1.6))
for ax, row in zip(axes, meta.sample(6, random_state=1).itertuples()):
    img = nib.load(f"{D}/nodules/images/{row.nodule_id}.nii.gz").get_fdata()
    lab = nib.load(f"{D}/nodules/labels/{row.nodule_id}.nii.gz").get_fdata()
    ax.imshow(img.T, cmap="gray", vmin=-1000, vmax=400)
    ax.contour(lab.T, levels=[0.5], colors="r", linewidths=0.8)
    ax.set_title(f"{row.diameter_mm:.0f} mm", fontsize=8)
    ax.axis("off")
fig.savefig("figures/nodules.png", **save)
