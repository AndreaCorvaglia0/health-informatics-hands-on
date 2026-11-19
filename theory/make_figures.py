"""Regenerate the figures of the handbook from the course dataset.

    python make_figures.py /path/to/luna16-lung-slices

figures/app.png is a screenshot of the web app (solutions/app.py) and is not made here.
figures/modalities.png and figures/intro_*.jpg are pictures prepared once (our dataset and Wikimedia Commons;
credits in theory.tex) and are not made here either.
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


# ------------------------------------------------------------------------------------------
# Figures added later.  Those marked "model" need a trained lung model: pass its path as the
# second argument (python make_figures.py <dataset> lung_segmenter.pth); otherwise they are skipped.
# ------------------------------------------------------------------------------------------
import torch
from scipy import ndimage as ndi
from monai.networks.nets import UNet
from monai.transforms import EnsureChannelFirst, LoadImage, Resize, ScaleIntensityRange
MODEL = sys.argv[2] if len(sys.argv) > 2 else None

# the pipeline, step by step
raw = image
scaled = (np.clip(raw, -1000, 400) + 1000) / 1400
resized = np.asarray(sample["image"][0])
fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.3))
axes[0].imshow(raw.T, cmap="gray")
axes[0].set_title("file: 512 x 512\nvalues %d to %d" % (raw.min(), raw.max()), fontsize=7)
axes[1].imshow(scaled.T, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("after rescaling\nvalues 0 to 1", fontsize=7)
axes[2].imshow(resized.T, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("after resizing\n256 x 256", fontsize=7)
axes[3].imshow(np.asarray(sample["label"][0]).T, cmap="gray")
axes[3].set_title("label after the transforms\n256 x 256, values 0 and 1", fontsize=7)
for ax in axes:
    ax.axis("off")
fig.savefig("figures/pipeline.png", **save)

# interpolation of a mask
t = torch.tensor(lung[None].astype(np.float32))
near = np.asarray(Resize((100, 100), mode="nearest")(t))[0]
bil = np.asarray(Resize((100, 100), mode="bilinear")(t))[0]
mixed = ((bil > 0.05) & (bil < 0.95)).astype(float)
count = ndi.uniform_filter(mixed, size=14)                 # where the border is most visible
cx, cy = np.unravel_index(np.argmax(count[10:-10, 10:-10]), count[10:-10, 10:-10].shape)
crop = (slice(cx + 3, cx + 17), slice(cy + 3, cy + 17))
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
for ax, arr, title in [(axes[0], near, '"nearest": only 0 and 1'), (axes[1], bil, '"bilinear": values in between')]:
    im = ax.imshow(arr[crop].T, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
fig.savefig("figures/interpolation.png", **save)

# a convolution
kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
edges = ndi.convolve(resized, kernel)
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
axes[0].imshow(resized.T, cmap="gray")
axes[0].set_title("image", fontsize=8)
axes[1].imshow(np.abs(edges).T, cmap="gray", vmax=np.abs(edges).max() * 0.6)
axes[1].set_title("output of the filter: vertical edges", fontsize=8)
for ax in axes:
    ax.axis("off")
fig.savefig("figures/convolution.png", **save)

# gradient descent with three learning rates
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.2), sharey=True)
w = np.linspace(-2.2, 2.2, 200)
for ax, (lr, title) in zip(axes, [(0.04, "too small: slow"), (0.3, "good"), (0.97, "too large: it jumps")]):
    ax.plot(w, w ** 2, color="0.6")
    p, path = -2.0, [-2.0]
    for _ in range(9):
        p = p - lr * 2 * p
        path.append(p)
    path = np.array(path)
    ax.plot(path, path ** 2, "o-", color="k", markersize=3, linewidth=0.8)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("one weight")
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].set_ylabel("loss")
fig.tight_layout()
fig.savefig("figures/learning_rate.png", **save)

# how much of the image is foreground
nodule_fraction = np.mean([nib.load(f"{D}/nodules/labels/{n}.nii.gz").get_fdata().mean() for n in meta.nodule_id[:200]])
all_meta = pd.read_csv(f"{D}/metadata.csv")
lung_fraction = (all_meta.right_lung_px + all_meta.left_lung_px).sum() / (all_meta.size_x * all_meta.size_y).sum()
fig, ax = plt.subplots(figsize=(5.4, 1.1))
for y, (name, frac) in enumerate([("nodule, in a 64 x 64 patch", nodule_fraction), ("lung, in a slice", lung_fraction)]):
    ax.barh(y, 100, color="0.88", height=0.6)
    ax.barh(y, 100 * frac, color="0.25", height=0.6)
    ax.text(101, y, f"{100 * frac:.1f}%", va="center", fontsize=8)
    ax.text(-2, y, name, fontsize=8, va="center", ha="right")
ax.set_xlim(-60, 112)
ax.axis("off")
fig.savefig("figures/imbalance.png", **save)

# Dice when the border is one pixel too wide
d_px = np.linspace(3, 200, 400)
r = d_px / 2
dice = 2 * r ** 2 / (r ** 2 + (r + 1) ** 2)
fig, ax = plt.subplots(figsize=(4.6, 2.3))
ax.plot(d_px, dice, color="k")
ax.set_xscale("log")
ax.set_xlabel("diameter of the object (pixels)")
ax.set_ylabel("Dice")
ax.set_title("Prediction one pixel too wide all around", fontsize=8)
for dd, name in [(7, "5 mm nodule"), (28, "20 mm nodule"), (160, "lung")]:
    rr = dd / 2
    val = 2 * rr ** 2 / (rr ** 2 + (rr + 1) ** 2)
    ax.plot([dd], [val], "o", color="k", markersize=4)
    ax.annotate(f"{name}\n{val:.2f}", (dd, val), textcoords="offset points", xytext=(6, -22), fontsize=7)
ax.set_ylim(0.5, 1.02)
fig.savefig("figures/dice_small.png", **save)

# a scan is a volume: three views, and what happens if the slice thickness is ignored
from scipy import ndimage
vol_file = nib.load(f"{D}/demo/volume/case_0001_image.nii.gz")
vol = vol_file.get_fdata()
sx, sy, sz = [float(v) for v in vol_file.header.get_zooms()]
nx, ny, nz = vol.shape
coronal = vol[:, ny // 2, ::-1]
sagittal = vol[nx // 3, :, ::-1]
fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.3), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
axes[0].imshow(vol[:, :, nz // 2].T, cmap="gray", vmin=-1000, vmax=400)
axes[0].set_title("axial (one slice)", fontsize=8)
axes[1].imshow(coronal.T, cmap="gray", vmin=-1000, vmax=400, aspect=sz / sx)
axes[1].set_title("coronal (from the front)", fontsize=8)
axes[2].imshow(sagittal.T, cmap="gray", vmin=-1000, vmax=400, aspect=sz / sy)
axes[2].set_title("sagittal (from the side)", fontsize=8)
axes[3].imshow(coronal.T, cmap="gray", vmin=-1000, vmax=400, aspect=1)
axes[3].set_title("coronal, thickness ignored", fontsize=8)
for ax in axes:
    ax.axis("off")
fig.savefig("figures/volume.png", **save)

# orientation: the same array shown as stored and transposed
fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.3))
axes[0].imshow(image, cmap="gray", vmin=-1000, vmax=400)
axes[0].set_title("plt.imshow(image)", fontsize=8)
axes[1].imshow(image.T, cmap="gray", vmin=-1000, vmax=400)
axes[1].set_title("plt.imshow(image.T)", fontsize=8)
for ax in axes:
    ax.axis("off")
fig.savefig("figures/orientation.png", **save)

# what the encoder does to the image: 256, 128, 64, 32, 16 pixels per side
small = ndimage.zoom((np.clip(image, -1000, 400) + 1000) / 1400, 256 / image.shape[0], order=1)
fig, axes = plt.subplots(1, 5, figsize=(7.5, 1.7))
for ax, n in zip(axes, [256, 128, 64, 32, 16]):
    ax.imshow(small.reshape(n, 256 // n, n, 256 // n).mean(axis=(1, 3)).T, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{n} x {n}", fontsize=8)
    ax.axis("off")
fig.savefig("figures/encoder.png", **save)

if MODEL:
    model = UNet(spatial_dims=2, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2), num_res_units=2)
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()

    def prepare(path, lo=-1000, hi=400):
        return Compose([LoadImage(image_only=True), EnsureChannelFirst(),
                        ScaleIntensityRange(a_min=lo, a_max=hi, b_min=0.0, b_max=1.0, clip=True),
                        Resize((256, 256), mode="bilinear")])(path)

    def predict(x):
        with torch.no_grad():
            out = model(x[None])
        return torch.softmax(out, dim=1)[0, 1].numpy(), out.argmax(dim=1)[0].numpy()

    case = "case_0003"
    x = prepare(f"{D}/images/{case}.nii.gz")
    prob, pred = predict(x)

    # scores -> probabilities -> mask
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
    axes[0].imshow(x[0].T, cmap="gray")
    axes[0].set_title("input", fontsize=8)
    im = axes[1].imshow(prob.T, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title('after softmax: probability of "lung"', fontsize=8)
    axes[2].imshow(pred.T, cmap="gray")
    axes[2].set_title("after argmax: the mask", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    fig.savefig("figures/output.png", **save)

    # same preparation or not
    x_bad = prepare(f"{D}/images/{case}.nii.gz", lo=-160, hi=240)
    _, pred_bad = predict(x_bad)
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.2))
    for ax, arr, title in [(axes[0], x[0], "prepared as in training\n(window -1000 / 400)"), (axes[1], pred, "prediction"),
                           (axes[2], x_bad[0], "prepared differently\n(window -160 / 240)"), (axes[3], pred_bad, "prediction")]:
        ax.imshow(np.asarray(arr).T, cmap="gray")
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.savefig("figures/skew.png", **save)

    # Dice of each test patient: the same split as in the project notebook (STEP 3)
    from sklearn.model_selection import train_test_split
    _, other = train_test_split(sorted(all_meta.case_id), test_size=0.3, random_state=42)
    _, cases = train_test_split(other, test_size=0.5, random_state=42)
    dices = []
    for c in cases:
        _, p = predict(prepare(f"{D}/images/{c}.nii.gz"))
        lab = nib.load(f"{D}/labels/{c}.nii.gz").get_fdata()
        true = np.asarray(Resize((256, 256), mode="nearest")(torch.tensor(((lab == 1) | (lab == 2))[None].astype(np.float32))))[0]
        dices.append(2 * (p * true).sum() / (p.sum() + true.sum()))
    dices = np.array(dices)
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.hist(dices, bins=np.linspace(0, 1, 41), color="0.35")
    ax.axvline(dices.mean(), color="k", ls="--", lw=0.8)
    ax.annotate(f"mean {dices.mean():.2f}", (dices.mean(), ax.get_ylim()[1] * 0.9), ha="right", fontsize=8,
                xytext=(-4, 0), textcoords="offset points")
    ax.annotate(f"worst patient {dices.min():.2f}", (dices.min(), 1), xytext=(dices.min(), ax.get_ylim()[1] * 0.45),
                arrowprops=dict(arrowstyle="->", lw=0.6), fontsize=8, ha="left")
    ax.set_xlabel("Dice of each test patient")
    ax.set_ylabel("patients")
    fig.savefig("figures/test_dice.png", **save)
    # what the first layer of the trained network computes: three of its 16 maps
    maps = {}
    hook = model.model[0].register_forward_hook(lambda module, inp, out: maps.setdefault("first", out))
    with torch.no_grad():
        model(x[None])
    hook.remove()
    first = maps["first"][0].numpy()
    order = np.argsort(-first.reshape(len(first), -1).std(axis=1))
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.1))
    axes[0].imshow(x[0].T, cmap="gray")
    axes[0].set_title("input", fontsize=8)
    for ax, k in zip(axes[1:], order[:3]):
        ax.imshow(first[k].T, cmap="magma", vmin=np.percentile(first[k], 1), vmax=np.percentile(first[k], 99))
        ax.set_title("one map of the first layer", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.savefig("figures/features.png", **save)

    # the three worst test patients: slice, reference mask, prediction
    worst = np.argsort(dices)[:3]
    fig, axes = plt.subplots(3, 3, figsize=(4.6, 4.6))
    for col, k in enumerate(worst):
        xk = prepare(f"{D}/images/{cases[k]}.nii.gz")
        _, pk = predict(xk)
        lab = nib.load(f"{D}/labels/{cases[k]}.nii.gz").get_fdata()
        ref = np.asarray(Resize((256, 256), mode="nearest")(torch.tensor(((lab == 1) | (lab == 2))[None].astype(np.float32))))[0]
        for row, arr in enumerate([np.asarray(xk[0]), ref, pk]):
            axes[row, col].imshow(arr.T, cmap="gray")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[0, col].set_title(f"Dice {dices[k]:.2f}", fontsize=8)
    for row, name in enumerate(["slice", "reference mask", "prediction"]):
        axes[row, 0].set_ylabel(name, fontsize=8)
    fig.savefig("figures/worst.png", **save)

    print(f"model figures done ({len(cases)} patients, mean Dice {dices.mean():.3f}, min {dices.min():.3f})")
