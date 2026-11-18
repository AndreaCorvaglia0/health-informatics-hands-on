# %% [markdown]
# # O2 - Beyond segmentation: classification, embeddings, fusion
# **Optional side track (days 3-4)** - for students, or a whole class, that are ahead of schedule.
#
# Segmentation is one task among many. Here we reuse the **same 2D slices and the same `metadata.csv`** for three
# other families of medical-AI tasks. The three parts are independent: you can do **A**, **B** or **C**
# alone (B and C need the 2D segmentation model of day 2, or the teachers' model; C needs the classifier of A and the
# embeddings of B, so do A and B first if you want the whole story).
#
# * **A. Classification** - "where in the chest is this slice?" A classifier on image slices, with a MONAI `DenseNet121`
#   (the idiom of the MONAI [MedNIST classification tutorial](https://github.com/Project-MONAI/tutorials/tree/main/2d_classification)).
#   And a lesson on **task definition**: why `has_nodule` is a bad target for one random slice.
# * **B. Embeddings** - what a neural network "knows" about an image, as a vector. We extract it with a **forward hook**,
#   map it in 2D (PCA, t-SNE), search similar images, and train a **linear probe**.
# * **C. Fusion** - combine image and **tabular** information (early vs late fusion), compare with single-modality baselines, and meet
#   the pitfall of **leaking features**.
#
# **Learning objectives**
# * turn a continuous variable into a classification target, and adapt a training loop from segmentation to classification
# * choose and interpret classification metrics (accuracy, confusion matrix, one-vs-rest ROC-AUC) and spot a badly defined task
# * extract and visualise embeddings, and tell *anatomy* from *acquisition shortcuts*
# * build early and late fusion models and know when fusion helps, and when a feature is "too good to be true"
#
# **Estimated time:** 2-3 hours (about 40-50 minutes per part; ✏️ spine of 5 exercises, 🚀 and 🌶️ optional).
#
# | Marker | Meaning |
# |---|---|
# | ✏️ Exercise | **Mandatory** (for this notebook): fill the blanks `____`, then run the check cell |
# | 🔍 Explore | Small "try this / change that" experiments |
# | 🚀 Challenge | Additional task (solution in the teachers' notebook) |
# | 🌶️ Going further | Open-ended, only pointers and starter code |
# | 🧠 Question / ⚠️ Pitfall / 📚 Read more | Concept question / classic mistake / links |

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
import json

import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import DenseNet121
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped, Lambdad, LoadImaged, RandAffined,
                              RandGaussianNoised, Resized, ScaleIntensityRanged)
from monai.utils import set_determinism
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if FAST else 2
BATCH_SIZE = 8 if FAST else 16
MAX_EPOCHS = 15 if FAST else 15
print(f"device: {device} | epochs {MAX_EPOCHS}")

# %% [markdown]
# ### Recap: the course pipeline (given)
# Same functions as in notebooks 03-06, copied verbatim, so that every notebook uses the same pre-processing. The
# image size and the HU window are read from the config of the 2D segmentation model (section B), so that classifier and
# segmentation model see the same input. We load that model right now (the canonical `find_model_dirs` / `load_model`).

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
# The 2D segmentation model (student's, otherwise the teachers' one). 3D models (spatial_dims=3) are skipped.
dirs_2d = [d for d in find_model_dirs() if json.loads((d / "config.json").read_text())["arch_kwargs"].get("spatial_dims", 2) == 2]
if not dirs_2d:
    raise FileNotFoundError("No 2D segmentation model found. Finish notebook 03 (day 2) or use DATA_DIR/pretrained/unet-baseline.")
seg_model, seg_cfg = load_model(dirs_2d[0], device)
IMG_SIZE = int(seg_cfg["preprocessing"]["img_size"][0])
HU_MIN, HU_MAX = seg_cfg["preprocessing"]["hu_min"], seg_cfg["preprocessing"]["hu_max"]
print(f"segmentation model: {dirs_2d[0].name} | input {IMG_SIZE} x {IMG_SIZE}, HU window [{HU_MIN}, {HU_MAX}]")

# %%
# Data: metadata + the course splits (subsets 0-6 train, 7-8 validation, 9 test = untouched in this notebook).
meta = pd.read_csv(DATA_DIR / "metadata.csv").set_index("case_id")
train_rows, val_rows, test_rows = make_datalists(DATA_DIR)
if FAST:                                      # tiny run: ~40 cases
    train_rows, val_rows, test_rows = train_rows[:24], val_rows[:12], test_rows[:6]
rows_all = train_rows + val_rows + test_rows
df = meta.loc[[r["case_id"] for r in rows_all]].reset_index()
df["split"] = ["train"] * len(train_rows) + ["val"] * len(val_rows) + ["test"] * len(test_rows)
print(df.split.value_counts().to_dict())
df[["case_id", "subset", "z_rel", "spacing_x", "slice_thickness", "lung_area_mm2", "has_nodule", "split"]].head()

# %% [markdown]
# ## Part A. Classification: where in the chest is this slice?
#
# Each patient has **one axial slice** at a random height of the scan. `z_rel` (0 = most inferior, 1 = most superior slice of
# the scan) tells us *where*. A slice near the lung **base** shows the diaphragm and liver dome; in the **middle** we see the heart
# and the largest lung area; at the **apex** two small lungs and the shoulder girdle. A CNN can learn to tell them apart, a
# useful toy version of real tasks like *"which anatomical region / sequence / view is this image?"* (series classification
# in PACS, DICOM metadata cleaning, quality control).
#
# We turn `z_rel` into **three classes** by tertiles: `0 = base`, `1 = middle`, `2 = apex`. The tertile *edges* are computed on the **training
# set only**.
#
# ### ✏️ Exercise A.1 - build the label from `z_rel`
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The tertiles are the quantiles 1/3 and 2/3 of the training `z_rel`. `np.digitize(x, bins)` returns, for each value, the number of bin
# edges that are <= x, i.e. 0, 1 or 2 for two edges.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `edges = train_z.quantile([1/3, 2/3]).to_numpy()` and `df["level"] = np.digitize(df.z_rel, edges)`
#
# </details>

# %%
LEVEL_NAMES = ["base", "middle", "apex"]
train_z = df.loc[df.split == "train", "z_rel"]
#@solution
edges = train_z.quantile([1 / 3, 2 / 3]).to_numpy()
df["level"] = np.digitize(df.z_rel, edges)
#@student
#| edges = train_z.quantile([____, ____]).to_numpy()      # TODO: the two tertile positions
#| df["level"] = np.digitize(df.z_rel, ____)             # TODO: use the edges as bins
#@end
meta["level"] = df.set_index("case_id")["level"].reindex(meta.index)
print("tertile edges of z_rel:", edges.round(3))
print(pd.crosstab(df.split, df.level.map(dict(enumerate(LEVEL_NAMES)))))

# %%
# Check A.1
assert set(df.level.unique()) <= {0, 1, 2}, "labels must be 0, 1 or 2"
frac = df.loc[df.split == "train", "level"].value_counts(normalize=True)
assert frac.min() > 0.2, "on the training set the three classes should be about balanced (tertiles!)"
assert df.groupby("level").z_rel.mean().is_monotonic_increasing, "level 0 = lowest z_rel, 2 = highest"
print("✅ Exercise A.1 passed")

# %% [markdown]
# **Class balance.** By construction the training set is balanced; the validation set only approximately (and with a small n,
# not even that). With a balanced problem **accuracy** is meaningful and the chance level is 1/3. With an imbalanced problem
# (e.g. 5% positives) accuracy would be misleading, and we would look at AUC, sensitivity/specificity, precision-recall.
# Let us look at one example per class (the images are already pre-processed as in the segmentation course).

# %%
def make_rows(rows, target):
    """Add the target value under the key "y" (the other keys are untouched by the dictionary transforms)."""
    return [{**r, "y": meta.at[r["case_id"], target].item()} for r in rows]

def make_loader(rows, augment=False, shuffle=False):
    ds = CacheDataset(rows, get_transforms(IMG_SIZE, HU_MIN, HU_MAX, augment=augment), num_workers=NUM_WORKERS)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)

preview = CacheDataset(make_rows(train_rows, "level"), get_transforms(IMG_SIZE, HU_MIN, HU_MAX), num_workers=0)
fig, axes = plt.subplots(3, 4, figsize=(9, 7))
for lvl in range(3):
    picks = [i for i, r in enumerate(make_rows(train_rows, "level")) if r["y"] == lvl][:4]
    for ax, i in zip(axes[lvl], picks):
        ax.imshow(preview[i]["image"][0].numpy().T, cmap="gray", vmin=0, vmax=1)     # .T: radiological view
        ax.set_title(f"{LEVEL_NAMES[lvl]}  z_rel={meta.at[train_rows[i]['case_id'], 'z_rel']:.2f}", fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### What changes with respect to segmentation (day 2)?
#
# | | Segmentation (day 2) | Classification (here) |
# |---|---|---|
# | Target `y` | mask, same size as the image | **one integer per image** (0, 1, 2) |
# | Network output | `(B, 2, H, W)` per-pixel scores | `(B, 3)` per-image scores (**logits**) |
# | Loss | `DiceCELoss` | **`CrossEntropyLoss`**: expects raw logits and **`long`** class indices (no one-hot, no softmax before) |
# | Prediction | `argmax` over the channel dimension, `AsDiscrete(to_onehot=...)` | `argmax` over **dimension 1** of `(B, 3)` |
# | Metric | Dice (needs `decollate_batch` + one-hot post-transforms) | accuracy, confusion matrix, **ROC-AUC** (plain numpy / sklearn) |
#
# ### ✏️ Exercise A.2 - adapt the training ingredients
#
# Fill in the three things that change.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# PyTorch's cross-entropy loss for multi-class problems is in `torch.nn`; its targets are class *indices* of integer type
# (64 bit).
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `torch.nn.CrossEntropyLoss()`, `torch.long`, and `logits.argmax(dim=1)`
#
# </details>

# %%
#@solution
cls_loss_fn = torch.nn.CrossEntropyLoss()
cls_target_dtype = torch.long
def predict_classes(logits):
    return logits.argmax(dim=1)
#@student
#| cls_loss_fn = ____            # TODO: the loss for multi-class classification (logits in, class indices as target)
#| cls_target_dtype = ____       # TODO: the dtype of class indices
#| def predict_classes(logits):
#|     return ____               # TODO: from (B, 3) scores to (B,) class indices
#@end

# %%
# Check A.2
_logits, _y = torch.randn(4, 3), torch.tensor([0, 1, 2, 1]).to(cls_target_dtype)
assert cls_loss_fn(_logits, _y).ndim == 0, "the loss must be a scalar"
assert _y.dtype == torch.int64, "class indices must be 64-bit integers"
assert predict_classes(_logits).shape == (4,) and predict_classes(_logits).max() <= 2, "one class index per image"
print("✅ Exercise A.2 passed")

# %% [markdown]
# Now the model and the (given) training loop. `DenseNet121` is the model of the MONAI MedNIST tutorial: with
# `spatial_dims=2, in_channels=1, out_channels=3` it takes a one-channel slice and returns three logits. It is
# *not* a U-Net: there is no decoder, only an encoder followed by global pooling and a linear layer.

# %%
def train_one_epoch(model, loader, optimizer, loss_fn, target_dtype):
    model.train()
    total = 0.0
    for batch in loader:
        x, y = batch["image"].to(device), batch["y"].to(target_dtype).to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x).squeeze(-1) if target_dtype.is_floating_point else model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def predict(model, loader):
    """Raw model outputs and targets for a whole loader, as numpy arrays."""
    model.eval()
    outs, ys = [], []
    for batch in loader:
        outs.append(model(batch["image"].to(device)).cpu())
        ys.append(batch["y"])
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()

def fit(model, train_loader, val_loader, loss_fn, target_dtype, epochs=MAX_EPOCHS, lr=1e-3, score=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, target_dtype)
        out, y = predict(model, val_loader)
        history.append((loss, score(out, y) if score else float("nan")))
        print(f"epoch {epoch:2d}/{epochs}  train loss {loss:.4f}  val score {history[-1][1]:.3f}")
    return np.array(history)

train_loader = make_loader(make_rows(train_rows, "level"), augment=True, shuffle=True)
val_loader = make_loader(make_rows(val_rows, "level"))
classifier = DenseNet121(spatial_dims=2, in_channels=1, out_channels=3).to(device)
hist_cls = fit(classifier, train_loader, val_loader, cls_loss_fn, cls_target_dtype,
               score=lambda out, y: accuracy_score(y, predict_classes(torch.tensor(out)).numpy()))

# %% [markdown]
# ### Evaluation: accuracy, confusion matrix, ROC-AUC (one-vs-rest)
# * **Confusion matrix**: *which* classes are confused? For ordered classes (base < middle < apex) confusing neighbours is
#   less serious than confusing base with apex.
# * **ROC-AUC one-vs-rest**: for each class, how well do the class probabilities separate "this class" from "the others"
#   (0.5 = chance, 1 = perfect); we average over classes (`macro`). MONAI has `monai.metrics.ROCAUCMetric`
#   (see the MedNIST tutorial); here we use `scikit-learn`, which needs no post-transforms.

# %%
def softmax_np(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def macro_auc(y, probs):
    """One-vs-rest ROC-AUC; NaN if a class is missing in `y` (possible with tiny validation sets)."""
    try:
        if probs.shape[1] == 2:
            return roc_auc_score(y, probs[:, 1])
        return roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(probs.shape[1])))
    except ValueError:
        return float("nan")

logits_val, y_val = predict(classifier, val_loader)
probs_val = softmax_np(logits_val)
pred_val = probs_val.argmax(1)
acc_cls, auc_cls = accuracy_score(y_val, pred_val), macro_auc(y_val, probs_val)
print(f"validation: accuracy {acc_cls:.3f} (chance 0.33) | macro one-vs-rest AUC {auc_cls:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay(confusion_matrix(y_val, pred_val, labels=[0, 1, 2]), display_labels=LEVEL_NAMES).plot(ax=axes[0], colorbar=False)
for c in range(3):
    if (y_val == c).any() and (y_val != c).any():
        fpr, tpr, _ = roc_curve(y_val == c, probs_val[:, c])
        axes[1].plot(fpr, tpr, label=f"{LEVEL_NAMES[c]}")
axes[1].plot([0, 1], [0, 1], "k:"); axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate")
axes[1].set_title("ROC, one-vs-rest"); axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 🔍 Explore
# * Change `MAX_EPOCHS` (top of the notebook) and look at the curve: does the model still improve? With a tiny validation set, how much can accuracy move by chance? (Each validation image is worth 1/n of accuracy.)
# * Train without augmentation (`augment=False` in the `make_loader` call). Does the validation score change?
# * Look at the images the classifier gets wrong (`np.where(pred_val != y_val)`). Are they near a tertile border?

# %% [markdown]
# ### 🧠 Task definition: why `has_nodule` is a bad target here
# `metadata.csv` also has `has_nodule`: is there at least one annotated nodule in the *patient*? Tempting target: "nodule detection"!
# But our images are **one random axial slice per patient**.
#
# > 🧠 **Question:** why is `has_nodule` a poor target for *this* input? What will the AUC be if we train on it? Name
# > two ways to define a meaningful nodule task.

# %% [markdown] tags=["solution-only"]
# **Answer.** The label is a property of the **whole scan** (any of ~300 slices), but the model sees **one** slice,
# which most of the time does not contain the nodule (nodules are a few mm to 3 cm; one slice out of 300 hits it with a small probability).
# The label does not match the input: for most positive patients the slice shows a perfectly normal lung, so the same image
# can be labelled both 0 and 1. No model can learn it: expect an **AUC near 0.5**, apart from chance and some shortcuts
# (see part B). Meaningful definitions: (1) a **slice-level / patch-level** label ("this slice or 3D patch contains a nodule", from the
# nodule coordinates in `annotations.csv`, i.e. LUNA16's own *candidate classification* task), (2) a **patient-level** task on the whole 3D scan,
# with a 3D model or an aggregation of the predictions over all slices (multiple-instance learning).
# **Lesson:** before training, check that the information needed for the label is *present in the input*.

# %% [markdown]
# ### 🚀 Challenge A.3 - try it anyway: `has_nodule`
# Train the same pipeline on the binary target `has_nodule` (2 outputs). Compute the validation AUC, compare it with 0.5, and
# say whether you can trust it **with your validation set size** (compute a bootstrap 95% interval of the AUC, or just look at n).
#
# <details>
# <summary>💡 Hint</summary>
#
# `make_rows(train_rows, "has_nodule")`, `DenseNet121(..., out_channels=2)`, and `macro_auc(y, probs)` handles two classes.
#
# </details>

# %%
#@solution
nod_train = make_loader(make_rows(train_rows, "has_nodule"), augment=True, shuffle=True)
nod_val = make_loader(make_rows(val_rows, "has_nodule"))
nodule_net = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
fit(nodule_net, nod_train, nod_val, cls_loss_fn, cls_target_dtype, epochs=MAX_EPOCHS)
logits_n, y_n = predict(nodule_net, nod_val)
probs_n = softmax_np(logits_n)
auc_nod = macro_auc(y_n, probs_n)

rng = np.random.default_rng(0)
boot = []
for _ in range(200):
    idx = rng.integers(0, len(y_n), len(y_n))
    if len(np.unique(y_n[idx])) == 2:
        boot.append(roc_auc_score(y_n[idx], probs_n[idx, 1]))
lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
print(f"has_nodule: n_val={len(y_n)}, prevalence {y_n.mean():.2f}, AUC {auc_nod:.3f}, bootstrap 95% interval [{lo:.2f}, {hi:.2f}]")
#@student
#| # TODO: your code (same recipe as above, binary target "has_nodule")
#@end

# %% [markdown] tags=["solution-only"]
# **Answer.** The AUC is around 0.5 and the bootstrap interval is wide (it contains 0.5, and with a small validation set
# it is wide enough to include values that *look* good). An AUC of, say, 0.62 on 50 patients is not evidence of anything. Two
# rules: report **uncertainty**, and compare with chance *and* with a trivial baseline.

# %% [markdown]
# ### 🚀 Challenge A.4 - regression instead of classification
# Tertiles throw away information (a slice at the border of two classes counts as an error). Predict `z_rel` directly: a
# `DenseNet121` with **one** output, an **L1 loss** (mean absolute error, robust and in the units of `z_rel`), float
# targets. Plot predicted vs true `z_rel` on the validation set, and compare the MAE with the trivial baseline "always predict the training mean".
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `out_channels=1`; `loss_fn = torch.nn.L1Loss()`; targets are `torch.float32`. The model output has shape `(B, 1)`: in `train_one_epoch`
# the loop already squeezes the last dimension for floating point targets.
#
# </details>
#
# <details>
# <summary>💡 Hint 2</summary>
#
# `make_rows(train_rows, "z_rel")`; then `predict(...)` returns `(B, 1)` outputs: use `out[:, 0]`.
#
# </details>

# %%
#@solution
reg_train = make_loader(make_rows(train_rows, "z_rel"), augment=True, shuffle=True)
reg_val = make_loader(make_rows(val_rows, "z_rel"))
regressor = DenseNet121(spatial_dims=2, in_channels=1, out_channels=1).to(device)
fit(regressor, reg_train, reg_val, torch.nn.L1Loss(), torch.float32, epochs=MAX_EPOCHS)
out_reg, y_reg = predict(regressor, reg_val)
pred_reg = out_reg[:, 0]
mae, mae_baseline = np.abs(pred_reg - y_reg).mean(), np.abs(train_z.mean() - y_reg).mean()
print(f"validation MAE of z_rel: model {mae:.3f} | baseline (training mean) {mae_baseline:.3f}")

plt.figure(figsize=(4, 4))
plt.scatter(y_reg, pred_reg, s=18)
plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], "k:")
plt.xlabel("true z_rel"); plt.ylabel("predicted z_rel"); plt.title("regression of the slice level")
plt.show()
#@student
#| # TODO: your code
#@end

# %% [markdown]
# ### 🌶️ Going further (optional): what does the classifier look at?
# **Grad-CAM** and **occlusion sensitivity** (both in `monai.visualize`) give a heat map of the image regions that drive
# a class score. They are **not proofs of causality**, but they are a good sanity check: a "base" classifier that
# looks at the liver dome and diaphragm is reassuring; one that looks at a text marker or the table at the border of the image is a shortcut.
# The cell is guarded: if it fails on your setup, the rest of the notebook is not affected.

# %%
try:
    from monai.visualize import GradCAM, OcclusionSensitivity

    classifier.eval()
    loader_one = make_loader(make_rows(val_rows, "level"))
    batch = next(iter(loader_one))
    x = batch["image"][:1].to(device)
    pred_class = int(predict_classes(classifier(x)).item())
    cam = GradCAM(nn_module=classifier, target_layers="class_layers.relu")(x=x, class_idx=pred_class)   # last feature map
    occ_map, _ = OcclusionSensitivity(nn_module=classifier, mask_size=IMG_SIZE // 4, n_batch=16)(x)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(x[0, 0].cpu().numpy().T, cmap="gray"); axes[0].set_title(f"input (predicted: {LEVEL_NAMES[pred_class]})")
    axes[1].imshow(x[0, 0].cpu().numpy().T, cmap="gray"); axes[1].imshow(cam[0, 0].cpu().numpy().T, cmap="jet", alpha=0.4); axes[1].set_title("Grad-CAM")
    axes[2].imshow(occ_map[0, pred_class].cpu().numpy().T, cmap="viridis"); axes[2].set_title("occlusion sensitivity")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
except Exception as err:             # visualisation is optional: never block the notebook
    print("Skipped (optional):", type(err).__name__, err)

# %% [markdown]
# ## Part B. Embeddings
#
# A neural network transforms an image step by step into more abstract representations. The activations of a layer, summarised as a
# vector, are an **embedding**: a point in a high-dimensional space where (hopefully) *similar images are close*. Embeddings are
# the basis of image retrieval ("find similar cases"), clustering, transfer learning (a classifier on top of a frozen model) and
# **foundation models**. We extract embeddings from two networks we already have:
#
# 1. the **bottleneck of the segmentation U-Net** (day 2): the deepest, most compressed representation, trained to find lungs, *never* told anything about
#    "base" or "apex";
# 2. the **features of the classifier** of part A, trained for the slice level.
#
# **How do we reach an intermediate layer?** With a PyTorch **forward hook**: a function attached to a module, called every time the module
# runs, with its input and output. First we need to *find* the module: print the model.

# %%
print("\n".join(str(seg_model).splitlines()[:28]), "\n...")

# %% [markdown]
# MONAI's `UNet` is a *nested* model: the `model` block contains `down`, a `SkipConnection` that contains the next level, and `up`. The bottleneck is
# the innermost block, buried several levels deep, and its name depends on the number of levels. Rather than reading the
# printout, we find it **programmatically**: a dry forward pass with a hook on every module, and we keep the module whose output has the
# **smallest spatial size** (the last one to finish at that size is the complete bottom block). We skip `SkipConnection`
# modules: they only *concatenate* the skip path with the deeper path, so their output has more channels than the real block.

# %%
def find_bottleneck(net, example):
    """Name of the module that produces the smallest feature map (the deepest block) in a forward pass."""
    seen, handles = [], []
    for name, module in net.named_modules():
        if name:
            handles.append(module.register_forward_hook(
                lambda m, i, o, n=name: seen.append((n, o.shape[2:].numel()))
                if torch.is_tensor(o) and o.ndim == 4 and type(m).__name__ != "SkipConnection" else None))
    with torch.no_grad():
        net.eval()(example)
    for h in handles:
        h.remove()
    smallest = min(size for _, size in seen)
    return [n for n, size in seen if size == smallest][-1]       # hooks fire when a module is complete: last = whole block

bottleneck_name = find_bottleneck(seg_model, torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device))
bottleneck = dict(seg_model.named_modules())[bottleneck_name]
print("bottleneck module:", bottleneck_name, "->", type(bottleneck).__name__)

# %% [markdown]
# ### ✏️ Exercise B.1 - write the hook
#
# Complete `extract_embeddings(model, layer, loader)`. The hook must store the output of `layer`, reduced to one vector per image by
# **global average pooling** (the mean over the two spatial dimensions of a `(B, C, H, W)` tensor). We also keep an optional
# `post` function for the *final output* of the model (we will need it in part C).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# A hook is `hook(module, inputs, output)`. The feature map has shape `(B, C, H, W)`: the spatial dimensions are 2 and 3.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `feats.append(output.detach().cpu().mean(dim=(2, 3)))`
#
# </details>

# %%
@torch.no_grad()
def extract_embeddings(model, layer, loader, post=None):
    """One embedding per image (global-average-pooled output of `layer`); optionally also post(model output)."""
    model.eval()
    feats, posts = [], []

    def hook(module, inputs, output):
        #@solution
        feats.append(output.detach().cpu().mean(dim=(2, 3)))
        #@student
        #| feats.append(output.detach().cpu().mean(dim=____))   # TODO: global average pooling over H and W
        #@end

    handle = layer.register_forward_hook(hook)
    try:
        for batch in loader:
            out = model(batch["image"].to(device))
            if post is not None:
                posts.append(post(out).cpu())
    finally:
        handle.remove()                       # always remove the hook, or it fires forever
    emb = torch.cat(feats).numpy()
    return (emb, torch.cat(posts).numpy()) if post is not None else emb

# %%
# One loader with ALL cases (train + val + test), in the order of df. Targets: the slice level.
all_loader = make_loader(make_rows(rows_all, "level"))

# Check B.1: a hook returns one vector per image, and the model is left clean.
_emb = extract_embeddings(seg_model, bottleneck, all_loader)
assert _emb.shape[0] == len(df) and _emb.ndim == 2, "one vector (C,) per image"
assert len(bottleneck._forward_hooks) == 0, "the hook must be removed after use"
print("✅ Exercise B.1 passed")

# %%
# Segmentation embeddings + the predicted lung area in pixels (needed for part C), in one pass over the data.
emb_seg, pred_area_px = extract_embeddings(seg_model, bottleneck, all_loader, post=lambda out: out.argmax(1).flatten(1).sum(1))
# Classifier embeddings: output of `features` (before global pooling and the linear layer) -> 1024 numbers
emb_cls = extract_embeddings(classifier, classifier.features, all_loader)
images_all = torch.cat([bt["image"][:, 0] for bt in all_loader]).numpy()    # (N, H, W), for the plots
print("segmentation-U-Net bottleneck:", emb_seg.shape, "| classifier features:", emb_cls.shape)

# %% [markdown]
# ### A 2D map of the embedding: PCA and t-SNE
# We cannot look at 256 or 1024 dimensions, so we project to 2D.
# * **PCA**: linear, fast, deterministic; keeps the directions of largest variance. Distances are meaningful.
# * **t-SNE**: non-linear; keeps *local neighbourhoods*; clusters look nice, but **distances between clusters and cluster sizes are
#   not meaningful**, and the picture changes with `perplexity` and the seed. Use it to *explore*, not to *prove*.
#   (UMAP is similar and often used, `pip install umap-learn`; we use it only if installed.)
#
# We colour the same map by four variables: `z_rel` (anatomy), pixel spacing and slice thickness (**acquisition**), and the LUNA16 `subset`.

# %%
def embed_2d(emb, method):
    x = StandardScaler().fit_transform(emb)
    n = len(x)
    if method == "pca":
        return PCA(n_components=2, random_state=0).fit_transform(x)
    if method == "tsne":
        if n < 6:
            raise ValueError("t-SNE needs a few more samples")
        perplexity = max(2.0, min(30.0, (n - 1) / 3))                # must be < n: adapt to tiny datasets
        return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=0).fit_transform(x)
    if method == "umap":
        import umap                                                  # optional dependency
        return umap.UMAP(n_neighbors=min(15, n - 1), random_state=0).fit_transform(x)
    raise ValueError(method)

COLOR_BY = {"z_rel": "viridis", "spacing_x": "plasma", "slice_thickness": "coolwarm", "subset": "tab10"}

def plot_map(emb, method, title):
    xy = embed_2d(emb, method)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (col, cmap) in zip(axes, COLOR_BY.items()):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=df[col], cmap=cmap, s=22)
        ax.set_title(f"{title} / {method}: {col}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()

plot_map(emb_seg, "pca", "U-Net bottleneck")
plot_map(emb_seg, "tsne", "U-Net bottleneck")
plot_map(emb_cls, "tsne", "classifier features")
try:
    plot_map(emb_seg, "umap", "U-Net bottleneck")
except ImportError:
    print("UMAP not installed (optional): pip install umap-learn")

# %% [markdown]
# **Quantify what you see.** Colours can fool us. A simple, honest test: how well can a *linear model on the embedding* predict
# each variable (cross-validated R², 1 = perfectly, 0 = no better than the mean, negative = worse)? If the embedding predicts the
# **pixel spacing** or the **slice thickness** as well as the anatomy (`z_rel`), it encodes how the scan was acquired.
# (`subset` is a category, not a number, so it is left out of this table: look at its colours in the maps.)

# %%
from sklearn.linear_model import RidgeCV

def r2_of(emb, target, folds=5):
    folds = max(2, min(folds, len(emb) // 8))                    # guard for tiny n
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 13)))
    return cross_val_score(model, emb, target, cv=folds, scoring="r2").mean()

probe_table = pd.DataFrame({name: {col: r2_of(emb, df[col].to_numpy(dtype=float)) for col in ["z_rel", "spacing_x", "slice_thickness"]}
                            for name, emb in {"U-Net bottleneck": emb_seg, "classifier features": emb_cls}.items()})
probe_table.round(2)

# %% [markdown]
# > 🧠 **Question:** what structure do you see in the maps and in the table: **anatomy** (`z_rel`) or **acquisition parameters**
# > (spacing, thickness, subset)? Which is good, and which is a warning sign? Why is a warning sign worse than bad accuracy?

# %% [markdown] tags=["solution-only"]
# **Answer.** On real data you typically see a smooth gradient of `z_rel` (apex <-> base) along the main axes, because anatomy is
# the dominant source of variation in a chest slice: *good*. If `spacing_x`, `slice_thickness` or `subset` (which in LUNA16 also groups
# sites/scanners) are also easy to read from the embedding, the network "knows" the scanner. This is a **shortcut / batch effect**: a model
# on top of it might use "this looks like scanner X" instead of the disease, and then fails in another hospital, or learns a
# spurious correlation if scanner and label are associated (e.g. a hospital that sees more sick patients). It is worse than low accuracy because it
# is **invisible in a random train/validation split** (same scanners on both sides). On our *synthetic* data the numbers have no clinical meaning;
# with 40 cases the R² values are also very noisy (compare them with the next run, they can be negative). The remedies are: harmonise
# acquisition (resampling), intensity augmentation, ComBat-like harmonisation, splits by site, or testing on an external centre.

# %% [markdown]
# ### Image retrieval: "find me similar slices"
# With embeddings, similarity search is a distance computation. **Cosine similarity** (the cosine of the angle between two vectors) ignores
# the vector length, so it is the usual choice. In a clinical setting this is "case-based retrieval": show the radiologist similar cases with their diagnosis.
#
# ### ✏️ Exercise B.2 - cosine similarity
# Given an embedding matrix `emb` of shape `(N, D)` and a query index `q`, return the indices of the 5 most similar images (excluding the query).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# First normalise each row to length 1 (`np.linalg.norm(..., axis=1, keepdims=True)`): then the cosine similarity is just a dot product.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `sims = emb_n @ emb_n[q]`, then `np.argsort(-sims)` sorts from most to least similar: the first one is the query itself.
#
# </details>

# %%
def nearest_neighbours(emb, q, k=5):
    #@solution
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = emb_n @ emb_n[q]
    order = np.argsort(-sims)
    return [i for i in order if i != q][:k]
    #@student
    #| emb_n = ____                      # TODO: rows normalised to unit length
    #| sims = ____                       # TODO: cosine similarity of every image with image q
    #| order = np.argsort(-sims)
    #| return [i for i in order if i != q][:k]
    #@end

# %%
# Check B.2
_nn = nearest_neighbours(emb_seg, 0, k=5)
assert len(_nn) == 5 and 0 not in _nn and len(set(_nn)) == 5, "5 distinct neighbours, not the query itself"
_v = np.random.default_rng(0).normal(size=(6, 8)); _v[3] = 5 * _v[0]            # image 3 = scaled copy of image 0
assert nearest_neighbours(_v, 0, k=1) == [3], "cosine similarity ignores the length of the vector"
print("✅ Exercise B.2 passed")

# %%
def show_retrieval(emb, q, title, k=5):
    nn_idx = nearest_neighbours(emb, q, k)
    fig, axes = plt.subplots(1, k + 1, figsize=(2.2 * (k + 1), 2.6))
    for ax, i, name in zip(axes, [q] + nn_idx, ["query"] + [f"#{j + 1}" for j in range(k)]):
        ax.imshow(images_all[i].T, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name}\nz_rel={df.z_rel[i]:.2f}", fontsize=9); ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

q = int(np.argmin(np.abs(df.z_rel - df.z_rel.median())))                   # a slice in the middle of the chest
show_retrieval(emb_seg, q, "neighbours in the U-Net bottleneck space")
show_retrieval(emb_cls, q, "neighbours in the classifier feature space")

# Quantitative view: mean |z_rel difference| between a query and its 5 neighbours, over all queries (lower = better), vs random neighbours
rng = np.random.default_rng(0)
for name, emb in {"U-Net bottleneck": emb_seg, "classifier features": emb_cls}.items():
    err = np.mean([np.abs(df.z_rel[nearest_neighbours(emb, i, 5)].to_numpy() - df.z_rel[i]).mean() for i in range(len(df))])
    rand = np.mean([np.abs(df.z_rel[rng.choice(len(df), 5)].to_numpy() - df.z_rel[i]).mean() for i in range(len(df))])
    print(f"{name:22s} mean |z_rel difference| to the 5 neighbours: {err:.3f}  (random neighbours: {rand:.3f})")

# %% [markdown]
# ### Linear probe: how much does the frozen embedding know?
# A **linear probe** trains only a linear classifier (here logistic regression) on **frozen** embeddings. It measures how much
# of the information is *linearly accessible* in the representation. Compare:
#
# * the probe on the **U-Net bottleneck** (the network never saw the slice level!),
# * the probe on the **classifier features** (trained for the task: expect the best probe, but it is not independent: those features were trained on the same labels),
# * the **end-to-end classifier** of part A.

# %%
tr, va = (df.split == "train").to_numpy(), (df.split == "val").to_numpy()
y_level = df.level.to_numpy()

def probe(emb):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    clf.fit(emb[tr], y_level[tr])
    p = clf.predict_proba(emb[va])
    return accuracy_score(y_level[va], p.argmax(1)), macro_auc(y_level[va], p), p

probe_seg, probe_cls = probe(emb_seg), probe(emb_cls)
table_b = pd.DataFrame({"accuracy": [probe_seg[0], probe_cls[0], acc_cls], "macro AUC": [probe_seg[1], probe_cls[1], auc_cls]},
                       index=["linear probe on U-Net bottleneck", "linear probe on classifier features", "end-to-end classifier (part A)"])
table_b.round(3)

# %% [markdown]
# ### 🌶️ Going further (optional)
# * **Self-supervised and foundation-model embeddings** are trained *without labels* on huge unlabeled image collections (contrastive learning such as SimCLR, masked image modelling,
#   DINO-like distillation) and give far better embeddings than a segmentation bottleneck. For CT, look at MONAI's self-supervised tutorials (`self_supervised_pretraining` in the
#   [MONAI tutorials](https://github.com/Project-MONAI/tutorials)) and at medical foundation models published on Hugging Face (check the license and the training data).
#   Try replacing `emb_seg` with embeddings of a pretrained model and re-run the maps and the probe.
# * Cluster the embeddings (k-means, k=3) and check how the clusters line up with `level`, with `subset` and with `slice_thickness`. Which wins?
# * Plot the ROC of the linear probe vs the end-to-end classifier on the same validation images.

# %% [markdown]
# ## Part C. Fusion: images + tabular data
#
# Real clinical decisions combine an **image** with other information: age, sex, labs, history, reports. Combining modalities is
# called **multimodal fusion**. Two basic strategies:
#
# * **Early fusion** (feature level): concatenate the image embedding with the tabular features and train **one** model on the concatenation.
# * **Late fusion** (decision level): train **one model per modality**, and combine their *outputs* (average probabilities, or a model on top).
#
# Our toy question: predict the slice level (`base / middle / apex`) from the slice **embedding** plus tabular information about the scan.
# What do we have in `metadata.csv`?
#
# | Column | Usable? |
# |---|---|
# | `spacing_x`, `slice_thickness`, `size_x` | yes: **acquisition** parameters, known for every new image (in the DICOM header) |
# | `lung_area_mm2` | **careful!** derived from the *label mask*. At inference time there is no label: use the area of the **predicted** mask |
# | `slice_index`, `n_slices`, `z_rel` | **forbidden, they contain the target** (`z_rel = slice_index / n_slices`...) |
# | `right_lung_px`, `left_lung_px`, `trachea_px` | derived from the label mask: same problem as `lung_area_mm2` |
#
# **Predicted lung area.** We compute it from the segmentation model's output (`pred_area_px`, section B): pixels of the predicted lung mask
# x the area of one resized pixel in mm² (`spacing x size / IMG_SIZE`, each axis). It exists for *any* new image, as it will in the real application. One caveat: the segmentation model was trained on the training patients, so on
# them its mask is a bit better than on new patients. A cleaner pipeline would use out-of-fold predictions.

# %%
pixel_area_mm2 = (df.size_x * df.spacing_x / IMG_SIZE) * (df.size_y * df.spacing_y / IMG_SIZE)
df["lung_area_pred_mm2"] = pred_area_px * pixel_area_mm2
TAB_COLS = ["spacing_x", "slice_thickness", "size_x", "lung_area_pred_mm2"]
print("predicted vs label-derived lung area, correlation:", round(np.corrcoef(df.lung_area_pred_mm2, df.lung_area_mm2)[0, 1], 3))
df[TAB_COLS + ["level"]].describe().T[["mean", "std", "min", "max"]].round(1)

# %% [markdown]
# > ⚠️ **Pitfall (leakage by features):** a feature is *leaking* when it is not available at prediction time, or when it is a
# > (near) function of the target. Look how "good" a tabular model becomes when we feed it forbidden columns. A score that is too
# > good is a reason to **doubt**, not to celebrate.

# %%
from sklearn.ensemble import GradientBoostingClassifier

def tab_model_scores(cols):
    gb = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=0).fit(df.loc[tr, cols], y_level[tr])
    p = gb.predict_proba(df.loc[va, cols])
    return accuracy_score(y_level[va], p.argmax(1)), macro_auc(y_level[va], p), p

for name, cols in {"legit tabular (4 columns)": TAB_COLS, "+ z_rel (the target itself!)": TAB_COLS + ["z_rel"],
                   "slice_index + n_slices (z_rel in disguise)": ["slice_index", "n_slices"]}.items():
    acc, auc, _ = tab_model_scores(cols)
    print(f"{name:45s} val accuracy {acc:.2f} | AUC {auc:.2f}")

# %% [markdown]
# ### The comparison
# We compare on the same validation images:
#
# | Model | Information |
# |---|---|
# | tabular only | 4 acquisition features (gradient boosting) |
# | image only | PCA(16) of the U-Net bottleneck + logistic regression |
# | image only, end-to-end | the classifier of part A |
# | **early fusion** | concatenation of PCA(16) embedding and scaled tabular features -> small MLP |
# | **late fusion** | average of the probabilities of "tabular only" and "image only" |
#
# We reduce the 256-D embedding to 16 dimensions with PCA (fitted on the **training** set only) so that the 4 tabular features are not drowned among 256 image features.
#
# ### ✏️ Exercise C.1 - early fusion: concatenate and train
# Build `X_fused` = [image features, tabular features], both standardised with scalers **fitted on the training rows only** (why?), and fit
# the MLP.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `np.concatenate([A, B], axis=1)` stacks the feature columns of two arrays with the same number of rows (images).
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `X_fused = np.concatenate([Z_img, Z_tab], axis=1)`, `mlp.fit(X_fused[tr], y_level[tr])`, `p_early = mlp.predict_proba(X_fused[va])`.
#
# </details>

# %%
from sklearn.neural_network import MLPClassifier

n_pca = max(2, min(16, int(tr.sum()) - 1))
emb_scaler, tab_scaler = StandardScaler().fit(emb_seg[tr]), StandardScaler().fit(df.loc[tr, TAB_COLS])
pca = PCA(n_components=n_pca, random_state=0).fit(emb_scaler.transform(emb_seg[tr]))
Z_img = pca.transform(emb_scaler.transform(emb_seg))                    # (N, 16) image features
Z_img = Z_img / Z_img[tr].std(0)                                         # unit variance, like the tabular ones
Z_tab = tab_scaler.transform(df[TAB_COLS])                               # (N, 4) tabular features (scaled on train only)

mlp = MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=1000, random_state=0)
#@solution
X_fused = np.concatenate([Z_img, Z_tab], axis=1)
mlp.fit(X_fused[tr], y_level[tr])
p_early = mlp.predict_proba(X_fused[va])
#@student
#| X_fused = ____                            # TODO: concatenate image and tabular features (axis=1)
#| mlp.fit(____, y_level[tr])                # TODO: fit on the training rows of X_fused
#| p_early = mlp.predict_proba(____)         # TODO: probabilities on the validation rows
#@end

# %%
# Check C.1
assert X_fused.shape == (len(df), Z_img.shape[1] + Z_tab.shape[1]), "one row per image, image + tabular columns"
assert p_early.shape == (va.sum(), len(np.unique(y_level[tr]))) and np.allclose(p_early.sum(1), 1), "class probabilities"
print("✅ Exercise C.1 passed")

# %%
# Single-modality models and late fusion (given).
_, _, p_tab = tab_model_scores(TAB_COLS)
img_lr = LogisticRegression(max_iter=2000, C=0.1).fit(Z_img[tr], y_level[tr])
p_img = img_lr.predict_proba(Z_img[va])
p_late = (p_img + p_tab) / 2                                  # late fusion: average the probabilities of the two models
p_e2e = probs_val                                              # end-to-end classifier of part A (same validation images)

def row(p):
    return {"accuracy": accuracy_score(y_level[va], p.argmax(1)), "macro AUC": macro_auc(y_level[va], p)}

fusion_table = pd.DataFrame({"tabular only": row(p_tab), "image only (frozen embedding + LR)": row(p_img),
                             "image only (end-to-end, part A)": row(p_e2e), "early fusion (concat + MLP)": row(p_early),
                             "late fusion (average)": row(p_late)}).T
print(f"validation images: {va.sum()}  (differences of a few points are NOT significant with so few images)")
fusion_table.round(3)

# %% [markdown]
# ### 🔍 Explore
# * Remove `lung_area_pred_mm2` from `TAB_COLS` and re-run: does the tabular model drop to chance? How much do early and late fusion change?
# * **Missing modality.** In practice the tabular data is often missing for some patients. With *late* fusion you can simply use the image model alone when the table is absent. How would you do it with the early-fusion MLP? (Ideas: imputation, "modality dropout" during training, a separate model for each availability pattern.)
# * Replace the lung area computed from the *prediction* by the label-derived `lung_area_mm2` (**do not use in a real project**). How much "better" does fusion look? That gap is the optimism caused by leakage.

# %% [markdown]
# > 🧠 **Question:** when does fusion help? In which situations would the early-fusion model be *no better* than the image alone?
# > And what would a *real* clinical fusion model combine?

# %% [markdown] tags=["solution-only"]
# **Answer.** Fusion helps when the modalities carry **complementary** information (different parts of the truth) and not only *redundant*
# information. Here the image already contains the lung size (the tabular area is a function of the image), and the spacing only
# tells how to convert pixels into mm: the complementary part is small, so fusion gives a modest gain at best (the area in **mm²** helps because the image alone
# does not know the physical scale: a patient with a small chest has a small area at any level). Early fusion cannot help when the tabular part is
# redundant, uninformative or if the training set is too small to learn the interaction, and it can *hurt* through overfitting. In real clinical models: imaging
# (CT/MRI, radiomics or deep embeddings) + labs, vital signs, medication, genomics, **clinical notes** (text embeddings), time series.
# The real problems are **missing modalities** (not every patient has every test, and *missingness itself is informative*), different sampling times, and the same leakage
# issue: a "diagnosis code" or a "treatment" column often *follows* the outcome. For each feature ask: *would I know it at the moment of the prediction?*

# %% [markdown]
# ### 🌶️ Going further (optional): a joint end-to-end fusion network
# Early fusion above used a **frozen** image embedding. A **joint** network learns the image features *together with* the tabular branch, with one loss.
# The starter code below defines the model and checks a forward pass. To use it: build a loader that also returns the tabular row (a
# `"tab"` key in the dictionaries), train with `CrossEntropyLoss` as in part A, and compare with the table above. Think about **regularisation**
# (dropout on the tabular branch, "modality dropout": zero a whole branch at random during training so the model also works with one modality missing).

# %%
class FusionNet(torch.nn.Module):
    """Image branch (DenseNet121 features + global pooling) and tabular branch (small MLP), concatenated -> classifier."""

    def __init__(self, n_tab, n_classes=3, tab_hidden=16):
        super().__init__()
        self.image_branch = DenseNet121(spatial_dims=2, in_channels=1, out_channels=n_classes).features
        self.tab_branch = torch.nn.Sequential(torch.nn.Linear(n_tab, tab_hidden), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(1024 + tab_hidden, n_classes))

    def forward(self, image, tab):
        z_img = self.image_branch(image).mean(dim=(2, 3))          # (B, 1024) global average pooling
        return self.head(torch.cat([z_img, self.tab_branch(tab)], dim=1))

fusion_net = FusionNet(n_tab=len(TAB_COLS)).to(device)
dummy = fusion_net(torch.zeros(2, 1, IMG_SIZE, IMG_SIZE, device=device), torch.zeros(2, len(TAB_COLS), device=device))
print("forward pass OK, output", tuple(dummy.shape), "|", sum(p.numel() for p in fusion_net.parameters()), "parameters")

# %% [markdown]
# ## Wrap-up
# * **Task definition first**: the label must be *in the input* (one slice cannot tell if a patient has a nodule). Check class balance and use metrics that fit the problem, with uncertainty.
# * **Classification vs segmentation** in MONAI: another head and loss (`CrossEntropyLoss`), integer targets, simple metrics.
# * **Embeddings** come from any layer through a forward hook. They enable retrieval and probes, and show what the model really encodes, including **shortcuts** (scanner, protocol).
# * **Fusion** helps when modalities are complementary; compare with single-modality baselines on the *same* data, and hunt for **leaking features**.
#
# > 📚 **Read more:** the MONAI [MedNIST classification tutorial](https://github.com/Project-MONAI/tutorials/tree/main/2d_classification),
# > the [scikit-learn documentation on t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), and the paper
# > *Metrics Reloaded* ([arXiv 2206.01653](https://arxiv.org/abs/2206.01653)) for choosing metrics.
