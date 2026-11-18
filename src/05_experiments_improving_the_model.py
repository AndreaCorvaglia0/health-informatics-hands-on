# %% [markdown]
# # 05 - Experiments: improving the model like a researcher
# **Day 3 (second half) and Day 4 (first half)** - about 3 to 3.5 hours
#
# You have a baseline U-Net (notebook 03) and you know how to evaluate it honestly (notebook 04).
# Now the real work of a data scientist starts: *"can I make it better, and how do I know?"*
# In this notebook you do not follow one fixed path. You get a **solid, given experiment harness**
# and a **menu of experiments**. Every experiment follows the same loop:
# *hypothesis -> change ONE thing -> run -> track -> compare fairly -> conclude*.
# At the end you pick your best model, test it **once** on the test set, and save it in the
# course contract format with a personal name: on day 5 the web app will let users choose among
# the models of the whole class.
#
# **Learning objectives**
# * Formulate a hypothesis *before* running an experiment and change only one thing at a time.
# * Use a harness that logs every run (TensorBoard + a CSV table) so results are comparable.
# * Measure seed-to-seed noise and use it to decide whether a difference is real.
# * Explore the effect of augmentation, training-set size, loss, architecture, optimisation and resolution.
# * Choose a model for deployment by accuracy **and** speed **and** size, and report it on the test set once.
#
# **Legend**
#
# | Marker | Meaning |
# |---|---|
# | ✏️ Exercise | **Mandatory**, with blanks (`____`) and hidden hints. Followed by a check cell. |
# | 🔍 Explore | Small "change this and re-run" experiments. |
# | 🚀 Challenge | **Additional** experiments for those who finish early (solutions in the teachers' notebook). |
# | 🌶️ Going further | **Optional**, open-ended, starter code only. |
# | 🧠 Question | Answer in words (in your notes, or with your neighbour). |
#
# **What is mandatory?** Section 3 (baseline + seed noise), experiments **A** (augmentation), **B** (data size) and
# **C** (loss) - or D / E instead of C if you prefer: pick *at least one of C-E* - and sections 5-7.
# Everything else is a menu: choose what interests you, and split it with your neighbour.
#
# **Time plan (real run, free GPU)**
# | Part | Minutes |
# |---|---|
# | 0-1 Setup, method | 15 |
# | 2 The harness (read, 3 small exercises) | 30 |
# | 3 Experiment 0: baseline + noise | 20 |
# | 4 Menu: A + B (mandatory) | 45 |
# | 4 Menu: C + (D or E) | 40 |
# | 4 Menu: F-J (additional / optional) | 0-60 |
# | 5-7 Compare, final model, save | 30 |

# %% [markdown]
# ## 0. Setup

# %%
# Setup: install what is missing (quietly). Kaggle and Colab already have most of it.
import importlib.util
import subprocess
import sys


def ensure(pip_name, module=None):
    if importlib.util.find_spec(module or pip_name) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)


for pkg, mod in [("monai", "monai"), ("nibabel", "nibabel"), ("tensorboard", "tensorboard"), ("pandas", "pandas")]:
    ensure(pkg, mod)

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
import datetime as dt
import json
import re
import shutil
import time
import warnings

import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.transforms import (AsDiscrete, Compose, EnsureChannelFirstd, EnsureTyped, Lambdad, LoadImaged,
                              RandAffined, RandGaussianNoised, Resized, ScaleIntensityRanged)
from monai.utils import set_determinism
from torch.utils.data import Subset

warnings.filterwarnings("ignore", category=FutureWarning, module="monai")   # deprecation noise from the metrics
warnings.filterwarnings("ignore", message=".*ground truth of class.*")
set_determinism(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if FAST else 2
IMG_SIZE = 64 if FAST else 256
EPOCHS = 2 if FAST else 30
STUDENT = "your-name"   # <- put your name here, it ends up in the config.json of your model
print(f"device: {device}   img size: {IMG_SIZE}   epochs per run: {EPOCHS}   MONAI {monai.__version__}")
if device.type == "cpu" and not FAST:
    print("WARNING: no GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2 (or P100). On Colab: Runtime -> Change runtime type.")

# %% [markdown]
# ### Recap cell: the canonical pipeline from notebooks 03-04
# `lungs_only` and `make_datalists` are **exactly** the functions of the previous days, so that every notebook and the
# web app agree on the pre-processing. `get_transforms` is the same function **extended with options that are off by
# default** (`multiclass`, `windows`, `img_size=None`); with the defaults it builds the identical pipeline. The random
# augmentations moved to their own small function `augmentations`, because the harness caches the deterministic part
# of the pipeline once and applies the random part on top (fast repeated runs).

# %%
def lungs_only(x):
    """1 = right lung, 2 = left lung -> 1; everything else (incl. trachea) -> 0."""
    return ((x == 1) | (x == 2)).to(x.dtype)


def augmentations(img_size):
    """The random (training-only) transforms of notebook 03."""
    keys = ["image", "label"]
    return [
        RandAffined(keys=keys, prob=0.5, rotate_range=0.15, scale_range=0.1, translate_range=img_size * 0.05,
                    mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandGaussianNoised(keys="image", prob=0.2, std=0.02),
    ]


def get_transforms(img_size, hu_min=-1000, hu_max=400, augment=False, multiclass=False, windows=None):
    """Default behaviour = notebook 03. Options (all off by default):
    multiclass: keep labels 0-3 instead of collapsing them to lungs
    windows:    list of (hu_min, hu_max) -> one input channel per CT window
    img_size=None: keep the native resolution (used only for evaluation)"""
    keys = ["image", "label"]
    t = [LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)]
    if not multiclass:
        t += [Lambdad(keys="label", func=lungs_only)]
    if windows is None:
        t += [ScaleIntensityRanged(keys="image", a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True)]
    else:
        t += [Lambdad(keys="image", func=lambda x: torch.cat(
            [(x.float().clamp(lo, hi) - lo) / (hi - lo) for lo, hi in windows], dim=0))]
    if img_size is not None:
        t += [Resized(keys=keys, spatial_size=(img_size, img_size), mode=("bilinear", "nearest"))]
    if augment:
        t += augmentations(img_size)
    t += [EnsureTyped(keys=keys)]
    return Compose(t)


def make_datalists(data_dir: Path):
    meta = pd.read_csv(data_dir / "metadata.csv")
    def rows(mask):
        return [{"image": str(data_dir / "images" / f"{c}.nii.gz"),
                 "label": str(data_dir / "labels" / f"{c}.nii.gz"), "case_id": c}
                for c in meta.loc[mask, "case_id"]]
    return rows(meta.subset <= 6), rows(meta.subset.isin([7, 8])), rows(meta.subset == 9)


META = pd.read_csv(DATA_DIR / "metadata.csv").set_index("case_id", drop=False)
train_files, val_files, test_files = make_datalists(DATA_DIR)
if FAST:   # tiny subsets so that the whole notebook runs in a few minutes on a CPU
    train_files, val_files, test_files = train_files[:24], val_files[:12], test_files[:8]
print(f"patients: train {len(train_files)}  val {len(val_files)}  test {len(test_files)} (the test set stays closed until section 6)")

# %% [markdown]
# ## 1. How to run experiments (short and opinionated)
#
# A model that "got 0.981" tells you nothing. A **table of controlled experiments** tells you a lot.
# The rules we follow in this notebook:
#
# 1. **Write the hypothesis first.** "I expect augmentation to help *more* with little data, because ...". If you
#    write it after seeing the numbers you will always be right, and you will learn nothing.
# 2. **Change ONE thing at a time.** If you change the loss *and* the learning rate and the result improves, you do not
#    know which one helped. The harness below makes this easy: every run is the baseline config plus a small diff.
# 3. **Fix everything else**: same data split, same validation set, same number of epochs, same seed.
# 4. **Measure the noise.** Train the *same* config with two seeds. The difference is the noise floor; any improvement
#    smaller than that is not an improvement (section 3).
# 5. **The test set is not for choosing.** We select with the validation set. The test set is opened **once**, at the
#    end, for the one model we ship (section 6). Every peek at the test set while choosing makes it a second validation set.
# 6. **Log everything automatically** (config, metrics, time, size). Your memory and your notes will not be enough at run 15.
# 7. **The results table is the deliverable**, not the code that produced it.
#
# **Compute budget.** A free GPU gives you about 9-12 hours per week on Kaggle (and a few hours on Colab); today you have
# roughly 2 hours for this notebook. So: *measure the time of one epoch first*, multiply, and only then decide how many
# runs you can afford. Experiments that are cheap and informative (tiny data, small images) come first.
#
# > ⚠️ **Pitfall:** "I tried 20 things and kept the best" is itself an experiment with a selection bias (the *winner's curse*):
# > the best of many noisy numbers is on average optimistic. We will come back to this in sections 5 and 6.
#
# > 📚 **Read more:** the nnU-Net paper (https://arxiv.org/abs/1809.10486) shows that a well-engineered *pipeline*
# > (pre-processing, augmentation, training schedule, post-processing) usually beats fancy new architectures.

# %% [markdown]
# ## 2. The harness (given - read it!)
# The harness is a small library of four ideas:
# * a **`CONFIG` dictionary** that fully describes one run;
# * **factories** that build model, loss and optimiser from names;
# * **`run_experiment(config)`** that trains, validates every epoch, keeps the best weights, logs to TensorBoard,
#   saves a model folder in the **course contract format** and appends one row to `experiments.csv`;
# * helpers to **compare** runs.
#
# Read each cell, answer the 🧠 questions, and fill the three small ✏️ blanks. They are small on purpose, but they are
# the places where experiments silently go wrong in real projects.
#
# ### 2.1 The config
# Every key of `BASE_CONFIG` is a knob you can turn. A run is always `{**BASE_CONFIG, "name": "my-run", <the one change>}`.
#
# | key | meaning |
# |---|---|
# | `name` | unique run name: folder names for TensorBoard (`runs/<name>`) and the model (`models/<name>`) |
# | `arch`, `arch_kwargs` | class name in `monai.networks.nets` and its arguments (the contract!) |
# | `loss` | `dice`, `ce`, `dice_ce`, `dice_focal`, `tversky`, `generalized_dice` |
# | `lr`, `optimizer`, `weight_decay`, `scheduler`, `batch_size`, `epochs`, `amp` | optimisation (`amp` = mixed precision, GPU only) |
# | `img_size` | the images are resized to `img_size x img_size` |
# | `augment` | random affine + noise on the training set only |
# | `train_fraction` | fraction of the training *patients* used (nested subsets, always the same order) |
# | `seed` | seed for initial weights, batch order and augmentation |
# | `windows`, `fold` | optional: multi-window input channels (H) and cross-validation fold (I); `None` = default |
#
# > 🧠 **Question:** the baseline uses `epochs=30` and the cosine schedule. If an experiment trains for 60 epochs, can you
# > still call it a fair comparison with the baseline? Why does `epochs` belong to the config, too?

# %%
def unet_kwargs(channels=(16, 32, 64, 128, 256), num_res_units=2, out_channels=2):
    """arch_kwargs of a 2D MONAI UNet: one stride-2 step between consecutive levels."""
    return dict(spatial_dims=2, in_channels=1, out_channels=out_channels, channels=list(channels),
                strides=[2] * (len(channels) - 1), num_res_units=num_res_units)


# If your notebook 03 used other values (lr, batch size, epochs...), copy them here: BASE_CONFIG IS your baseline.
BASE_CONFIG = dict(
    name="baseline",
    arch="UNet", arch_kwargs=unet_kwargs(),
    loss="dice_ce",
    lr=1e-3, optimizer="adam", weight_decay=0.0, scheduler="cosine",
    batch_size=16, epochs=EPOCHS, amp=False,
    img_size=IMG_SIZE, augment=True, train_fraction=1.0, seed=42,
    windows=None, fold=None,
)


def changed_keys(cfg, ref=None):
    """Which keys differ from the reference config (default: the baseline)? 'name' does not count."""
    ref = BASE_CONFIG if ref is None else ref
    return {k for k in BASE_CONFIG if k != "name" and cfg.get(k) != ref[k]}


# %% [markdown]
# ### 2.2 Factories: model, loss, optimiser
# `build_model` is the same one-liner the web app will use to rebuild your network from `config.json`.
#
# ### ✏️ Exercise 2.1 - complete the loss zoo
# The losses live in a dictionary of small functions (`lambda: ...`), so that a *new* loss object is created for every run.
# Add `dice_focal`: a `DiceFocalLoss` for a **segmentation with softmax** where the label is an integer map
# (so MONAI has to one-hot it).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Look at the entry `dice_ce` right above: `DiceFocalLoss` takes the same two arguments (`to_onehot_y`, `softmax`).
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `DiceFocalLoss(to_onehot_y=True, softmax=True)`
#
# </details>

# %%
class CELoss(torch.nn.Module):
    """Plain cross-entropy for (B, C, H, W) logits and (B, 1, H, W) integer labels."""
    def forward(self, logits, y):
        return F.cross_entropy(logits, y.squeeze(1).long())


LOSSES = {
    "dice": lambda: DiceLoss(to_onehot_y=True, softmax=True),
    "ce": lambda: CELoss(),
    "dice_ce": lambda: DiceCELoss(to_onehot_y=True, softmax=True),
    #@solution
    "dice_focal": lambda: DiceFocalLoss(to_onehot_y=True, softmax=True),
    #@student
    #| "dice_focal": lambda: ____,  # TODO: a DiceFocalLoss for integer labels + softmax outputs
    #@end
    # alpha weights false positives, beta false negatives: beta > alpha = "missing lung is worse than extra lung"
    "tversky": lambda: TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.3, beta=0.7),
    "generalized_dice": lambda: GeneralizedDiceLoss(to_onehot_y=True, softmax=True),
}


def build_model(arch, arch_kwargs):
    """Rebuild a network from the contract: the class name in monai.networks.nets + its kwargs."""
    return getattr(monai.networks.nets, arch)(**arch_kwargs)


def build_loss(name):
    return LOSSES[name]()


def build_optimizer(cfg, params):
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(params, cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(params, cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, cfg["lr"], momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])
    raise ValueError(f"unknown optimizer {cfg['optimizer']}")


# %%
# Check 2.1
_logits, _y = torch.randn(2, 2, 32, 32), (torch.rand(2, 1, 32, 32) > 0.5).long()
for _name in LOSSES:
    _value = build_loss(_name)(_logits, _y)
    assert _value.ndim == 0 and torch.isfinite(_value), _name
assert isinstance(build_loss("dice_focal"), DiceFocalLoss), "dice_focal must be a DiceFocalLoss"
_n_params = sum(p.numel() for p in build_model(BASE_CONFIG["arch"], BASE_CONFIG["arch_kwargs"]).parameters())
print(f"✅ Exercise 2.1 passed (the baseline U-Net has {_n_params:,} parameters)")

# %% [markdown]
# > 🧠 **Question:** `CELoss` has to call `y.squeeze(1).long()` while `DiceLoss` takes `y` as it is and `to_onehot_y=True`.
# > What is the shape and dtype that `F.cross_entropy` expects for the target, and what does `to_onehot_y` do instead?

# %% [markdown] tags=["solution-only"]
# **Answer.** `F.cross_entropy` wants class indices of shape `(B, H, W)` and dtype `long`; the loss compares them with
# `(B, C, H, W)` logits. MONAI's Dice-type losses need a one-hot target `(B, C, H, W)` to multiply with the softmax
# probabilities: `to_onehot_y=True` creates it from the `(B, 1, H, W)` integer map. In `DiceCELoss` MONAI does both
# internally.

# %% [markdown]
# ### 2.3 Data: cache once, reuse in every run
# Loading and resizing 600 NIfTI files takes a while, and it is **deterministic**: the same for every run with the same
# `img_size`. So we cache that part once (`CacheDataset`) and keep it in a dictionary. Only the random augmentation is
# applied on the fly, on top of the cache. A `train_fraction < 1` takes a **nested** subset of the patients (the 5% are
# contained in the 10%, which are contained in the 25%...) using one fixed random order, so the learning curve in
# experiment B compares like with like.
#
# ### ✏️ Exercise 2.2 - how many patients?
# Compute `n_train`: the number of training patients for a given `train_fraction` (at least one patient!).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `fraction * len(...)` is a float: round it to an integer, and protect against 0 with `max(1, ...)`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `n_train = max(1, int(round(cfg["train_fraction"] * len(all_train))))`
#
# </details>

# %%
_CACHE = {}                                                                     # (split, fold, img_size, multiclass, windows) -> CacheDataset
_ORDER = np.random.RandomState(0).permutation(len(train_files))                 # fixed patient order for nested subsets
SUBSET = META["subset"].to_dict()                                               # case_id -> official LUNA16 subset (0-9)


def get_files(split, fold=None):
    """Default: subsets 0-6 train / 7-8 validation. With fold=k: validate on subset k, train on the other dev subsets (0-8)."""
    if fold is None:
        return {"train": train_files, "val": val_files}[split]
    dev = train_files + val_files
    return [f for f in dev if (SUBSET[f["case_id"]] != fold) == (split == "train")]


def get_base_dataset(split, cfg):
    """Deterministic, cached part of the pipeline (load, HU window, resize)."""
    multiclass = cfg["arch_kwargs"]["out_channels"] > 2
    key = (split, cfg["fold"], cfg["img_size"], multiclass, json.dumps(cfg["windows"]))
    if key not in _CACHE:
        tf = get_transforms(cfg["img_size"], multiclass=multiclass, windows=cfg["windows"])
        _CACHE[key] = CacheDataset(get_files(split, cfg["fold"]), tf, num_workers=max(1, NUM_WORKERS), progress=False)
    return _CACHE[key]


def make_loaders(cfg):
    """Returns train_loader, val_loader and the number of training patients actually used."""
    all_train = get_base_dataset("train", cfg)
    #@solution
    n_train = max(1, int(round(cfg["train_fraction"] * len(all_train))))
    #@student
    #| n_train = ____  # TODO: number of patients = fraction * all training patients (at least 1)
    #@end
    order = _ORDER if len(all_train) == len(_ORDER) else np.random.RandomState(0).permutation(len(all_train))
    train_ds = Subset(all_train, sorted(order[:n_train].tolist()))
    if cfg["augment"]:
        train_ds = Dataset(train_ds, transform=Compose(augmentations(cfg["img_size"])))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(get_base_dataset("val", cfg), batch_size=cfg["batch_size"], num_workers=NUM_WORKERS)
    return train_loader, val_loader, n_train


# %%
# Check 2.2
_, _, _n_half = make_loaders({**BASE_CONFIG, "train_fraction": 0.5})
_, _, _n_tiny = make_loaders({**BASE_CONFIG, "train_fraction": 0.0001})
assert _n_half == max(1, round(0.5 * len(train_files))), "wrong number of patients for fraction 0.5"
assert _n_tiny == 1, "even a tiny fraction must keep at least one patient"
print(f"✅ Exercise 2.2 passed (50% of {len(train_files)} patients = {_n_half})")

# %% [markdown]
# > 🧠 **Question:** why do we draw the patient order **once** with a fixed seed (`RandomState(0)`) instead of taking
# > a fresh random subset in every run? And why is it the *patients* that we subsample, not the pixels?

# %% [markdown] tags=["solution-only"]
# **Answer.** (1) A fresh subset per run would add "which patients happened to be drawn" to the noise and would make
# the 10% and 25% runs incomparable; nested subsets isolate the effect of *how many* patients. (2) The unit of
# independent information is the patient: neighbouring pixels (or slices) of one patient are highly correlated, and the
# split must be by patient to avoid leakage. Here we also have exactly one slice per patient, so "patients" and
# "images" coincide - in a 3D dataset they would not.

# %% [markdown]
# ### 2.4 `run_experiment`: the heart of the harness
# What happens in a run, in order: set the seed -> build data, model, loss, optimiser, scheduler -> for every epoch: train,
# validate (Dice on the **validation** set) -> remember the best weights *in memory* -> at the end restore the best
# weights, save the model folder (contract), the TensorBoard logs, the history and one CSV row.
#
# ### ✏️ Exercise 2.3 - remember the best weights
# In the loop we keep the weights of the best epoch. Fill the blank so that `best_state` is a **copy** of the weights.
# The harness re-validates the restored weights at the end (`reload_dice`) and it must reproduce the best Dice.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `model.state_dict()` returns a dictionary whose tensors are *references* to the live parameters. They keep changing
# when training continues. You need an independent copy.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `copy.deepcopy(model.state_dict())`
#
# </details>

# %%
RESULTS = {}   # name -> dict returned by run_experiment (history, state_dict...), for the current session only
RESULTS_CSV = WORK_DIR / "experiments.csv"


def contract_labels(n_classes):
    if n_classes == 2:
        return {"0": "background", "1": "lung"}
    return {"0": "background", "1": "right lung", "2": "left lung", "3": "trachea"}


def run_experiment(config, record=True, verbose=True):
    """Train one configuration. Returns a dict with the history, the best weights and the summary numbers.
    record=False: no files at all (used for quick timing tests)."""
    unknown = set(config) - set(BASE_CONFIG)
    assert not unknown, f"unknown config keys {unknown} - typo? valid keys: {sorted(BASE_CONFIG)}"
    cfg = {**BASE_CONFIG, **config}
    n_classes = cfg["arch_kwargs"]["out_channels"]
    set_determinism(seed=cfg["seed"])

    train_loader, val_loader, n_train = make_loaders(cfg)
    model = build_model(cfg["arch"], cfg["arch_kwargs"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = build_loss(cfg["loss"])
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) if cfg["scheduler"] == "cosine" else None
    use_amp = bool(cfg["amp"]) and device.type == "cuda"          # mixed precision only makes sense on a GPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metric = DiceMetric(include_background=False, reduction="mean_batch")   # one Dice per foreground class
    post_pred, post_label = AsDiscrete(argmax=True, to_onehot=n_classes), AsDiscrete(to_onehot=n_classes)

    writer = None
    if record:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = WORK_DIR / "runs" / cfg["name"]
            shutil.rmtree(log_dir, ignore_errors=True)            # re-running a name overwrites the old curves
            writer = SummaryWriter(log_dir)
        except ImportError:
            print("tensorboard not installed: logging only to experiments.csv and the history file")

    def validate():
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["image"].to(device), batch["label"].to(device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                metric(y_pred=[post_pred(p) for p in decollate_batch(logits)], y=[post_label(l) for l in decollate_batch(y)])
        per_class = metric.aggregate().tolist()
        metric.reset()
        return float(np.nanmean(per_class)), per_class

    history = {"epoch": [], "train_loss": [], "val_dice": [], "lr": [], "val_dice_per_class": []}
    best = {"dice": -1.0, "epoch": 0}
    best_state = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        val_dice, per_class = validate()
        train_loss = running / len(train_loader)
        for key, value in zip(history, [epoch, train_loss, val_dice, lr_now, per_class]):
            history[key].append(value)
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("dice/val", val_dice, epoch)
            writer.add_scalar("lr", lr_now, epoch)
        if val_dice > best["dice"]:
            #@solution
            best_state = copy.deepcopy(model.state_dict())
            #@student
            #| best_state = ____  # TODO: an independent COPY of the best weights
            #@end
            best.update(dice=val_dice, epoch=epoch)
        if verbose and (epoch % 5 == 0 or epoch in (1, cfg["epochs"])):
            print(f"  [{cfg['name']}] epoch {epoch:3d}/{cfg['epochs']}  loss {train_loss:.4f}  val dice {val_dice:.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.time() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    if writer:
        writer.close()

    # restore the best weights and check that they really reproduce the best validation Dice
    model.load_state_dict(best_state)
    reload_dice, _ = validate()
    if abs(reload_dice - best["dice"]) > 2e-3:
        print(f"  WARNING: restored weights give {reload_dice:.4f}, not {best['dice']:.4f}: is best_state a real copy?")

    result = dict(config=cfg, history=history, best_val_dice=best["dice"], best_epoch=best["epoch"],
                  best_per_class=history["val_dice_per_class"][best["epoch"] - 1], reload_dice=reload_dice,
                  train_time_s=train_time, n_params=n_params, n_train=n_train, peak_mem_mb=peak_mem_mb,
                  state_dict={k: v.detach().cpu() for k, v in best_state.items()})
    if record:
        save_run(result)
        RESULTS[cfg["name"]] = result
    if verbose:
        print(f"-> {cfg['name']}: best val dice {best['dice']:.4f} at epoch {best['epoch']}  "
              f"({train_time:.0f} s, {n_params:,} params, {n_train} patients)")
    return result


def save_run(result):
    """Contract model folder + history + one row in experiments.csv."""
    cfg = result["config"]
    model_dir = WORK_DIR / "models" / cfg["name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result["state_dict"], model_dir / "model.pt")
    kwargs = {k: v for k, v in cfg["arch_kwargs"].items()}
    if "pretrained" in kwargs:
        kwargs["pretrained"] = False          # the weights are in model.pt: the app must NOT download anything
    preprocessing = {"hu_min": -1000, "hu_max": 400, "img_size": [cfg["img_size"]] * 2}
    if cfg["windows"] is not None:
        preprocessing["windows"] = cfg["windows"]
    contract = {
        "name": cfg["name"], "arch": cfg["arch"], "arch_kwargs": kwargs, "preprocessing": preprocessing,
        "labels": contract_labels(cfg["arch_kwargs"]["out_channels"]),
        "metrics": {"val_dice": round(result["best_val_dice"], 4)},
        "trained_by": STUDENT, "date": dt.date.today().isoformat(), "epochs": cfg["epochs"], "loss": cfg["loss"],
    }
    (model_dir / "config.json").write_text(json.dumps(contract, indent=2))
    run_dir = WORK_DIR / "runs" / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(json.dumps(result["history"]))
    row = {"name": cfg["name"], "date": dt.datetime.now().isoformat(timespec="seconds"), "fast": FAST,
           "arch": cfg["arch"], "loss": cfg["loss"], "lr": cfg["lr"], "optimizer": cfg["optimizer"],
           "weight_decay": cfg["weight_decay"], "scheduler": cfg["scheduler"], "batch_size": cfg["batch_size"],
           "epochs": cfg["epochs"], "amp": cfg["amp"], "img_size": cfg["img_size"], "augment": cfg["augment"],
           "train_fraction": cfg["train_fraction"], "n_train": result["n_train"], "seed": cfg["seed"],
           "n_classes": cfg["arch_kwargs"]["out_channels"], "fold": cfg["fold"],
           "best_val_dice": round(result["best_val_dice"], 5), "best_epoch": result["best_epoch"],
           "train_time_s": round(result["train_time_s"], 1), "n_params": result["n_params"],
           "peak_mem_mb": round(result["peak_mem_mb"]), "config_json": json.dumps(cfg)}
    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)


# %%
# Check 2.3: a smoke test (tiny, nothing is recorded)
_smoke = run_experiment({**BASE_CONFIG, "name": "_smoke", "epochs": 2, "train_fraction": 0.25}, record=False)
assert len(_smoke["history"]["val_dice"]) == 2 and _smoke["n_params"] > 0
assert abs(_smoke["reload_dice"] - _smoke["best_val_dice"]) < 2e-3, "the restored best weights do not reproduce the best Dice: is best_state a copy?"
print("✅ Exercise 2.3 passed")

# %% [markdown]
# > 🧠 **Questions:**
# > 1. Why do we set the seed at the *start of every run* and not only once at the top of the notebook?
# > 2. We validate after every epoch and keep the best epoch. Is the "best val Dice" an unbiased estimate of the performance
# >    on new patients? (Think: we *selected* the epoch on that very set.)
# > 3. The loop only restores the best weights at the end. What would be different if we saved the *last* epoch instead?

# %% [markdown] tags=["solution-only"]
# **Answers.** (1) With one seed at the top, run *k* would start from a random state that depends on how many random
# numbers the previous runs consumed: results would depend on the order in which you ran the cells and could not be
# reproduced individually. (2) No: it is optimistic, because we pick the maximum over epochs of a noisy curve (a small
# selection bias; with 30 epochs and a smooth curve it is small). Only the test set - used once - is unbiased.
# (3) With a cosine schedule the last epoch is usually (almost) the best; with a constant learning rate it is often
# noisier. Keeping the best is cheap insurance, but both should be close; a big gap means unstable training.

# %% [markdown]
# ### 2.5 Evaluation and comparison helpers
# * `evaluate_cases` - per-case Dice and HD95 in **millimetres at native resolution** (the model prediction is upsampled to
#   the original grid), so models trained at different `img_size` are compared on the same footing. It is a compact version
#   of what you built in notebook 04.
# * `load_results`, `compare`, `learning_curve` - tables and plots from `experiments.csv` and the saved histories.
# * `find_model_dirs` / `load_model` - the course contract loader (identical in notebooks 04-06 and in the app).

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


class SoftmaxEnsemble(torch.nn.Module):
    """Average of the softmax probabilities of several models (used in the optional experiment I)."""
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([m(x).softmax(1) for m in self.models]).mean(0)


def dice_np(a, b):
    """Dice of two boolean arrays (NaN if both are empty)."""
    denom = a.sum() + b.sum()
    return float("nan") if denom == 0 else float(2 * (a & b).sum() / denom)


@torch.no_grad()
def evaluate_cases(model, files, img_size, windows=None):
    """Per-case metrics at NATIVE resolution: dice_lung, hd95_mm (+ dice per class for 4-class models)."""
    model = model.to(device).eval()
    tf_in, tf_native = get_transforms(img_size, windows=windows), get_transforms(None, multiclass=True)
    rows = []
    for f in files:
        x = tf_in(f)["image"].unsqueeze(0).to(device)
        gt = tf_native(f)["label"][0].numpy().astype(int)                       # native-size label 0-3
        logits = F.interpolate(model(x).float(), size=gt.shape, mode="bilinear", align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy()
        n_classes = logits.shape[1]
        gt_lung = (gt == 1) | (gt == 2)
        pred_lung = (pred == 1) if n_classes == 2 else ((pred == 1) | (pred == 2))
        sx, sy = META.loc[f["case_id"], ["spacing_x", "spacing_y"]]
        onehot = lambda m: torch.from_numpy(np.stack([~m, m])[None].astype(np.float32))
        hd = compute_hausdorff_distance(onehot(pred_lung), onehot(gt_lung), include_background=False,
                                        percentile=95, spacing=(float(sx), float(sy))).item() if pred_lung.any() and gt_lung.any() else float("nan")
        row = {"case_id": f["case_id"], "dice_lung": dice_np(pred_lung, gt_lung), "hd95_mm": hd}
        if n_classes > 2:
            row.update({f"dice_c{c}": dice_np(pred == c, gt == c) for c in range(1, n_classes)})
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_model_dir(model_dir, files):
    """Everything comes from the model folder (contract): architecture, img_size, windows."""
    model, cfg = load_model(model_dir, device)
    pre = cfg["preprocessing"]
    return evaluate_cases(model, files, pre["img_size"][0], pre.get("windows"))


def load_results():
    """experiments.csv -> one row per run name (the latest wins), only runs of the current mode (FAST or real)."""
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["fast"] == FAST].drop_duplicates("name", keep="last")
    return df.set_index("name")


def config_of(name):
    return json.loads(load_results().loc[name, "config_json"])


def load_history(name):
    return json.loads((WORK_DIR / "runs" / name / "history.json").read_text())


NOISE = None   # seed-to-seed spread of the baseline: set in section 3, used by compare()
TABLE_COLS = ["arch", "loss", "lr", "img_size", "augment", "train_fraction", "n_train", "n_params", "train_time_s",
              "best_epoch", "best_val_dice"]


def compare(names, title="", baseline="baseline-s42"):
    """Table (with the difference from the baseline and whether it exceeds the noise) + curves on the same axes + bars."""
    res = load_results()
    t = res.loc[names, TABLE_COLS].copy()
    if baseline in res.index:
        t["delta_vs_base"] = t["best_val_dice"] - res.loc[baseline, "best_val_dice"]
        if NOISE is not None:
            t["beyond_noise"] = t["delta_vs_base"].abs() > NOISE
    display(t.round(4))
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for n in names:
        h = load_history(n)
        ax[0].plot(h["epoch"], h["val_dice"], marker="o" if len(h["epoch"]) < 5 else None, label=n)
    ax[0].set(xlabel="epoch", ylabel="val Dice", title=f"{title} - validation curves")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    vals = t["best_val_dice"]
    ax[1].bar(range(len(names)), vals, color="tab:blue")
    if baseline in res.index and NOISE is not None:
        b = res.loc[baseline, "best_val_dice"]
        ax[1].axhspan(b - NOISE, b + NOISE, color="tab:orange", alpha=0.3, label="baseline +/- seed noise")
        ax[1].legend(fontsize=8)
    ax[1].set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax[1].set(ylim=(max(0, vals.min() - 0.02), min(1.0, vals.max() + 0.01)), ylabel="best val Dice", title="best epoch")
    plt.tight_layout()
    plt.show()
    return t


def estimate_run_time(cfg, n_epochs=1):
    """Time a few epochs of a config (nothing is saved) and project the time of a full run."""
    _ = get_base_dataset("train", {**BASE_CONFIG, **cfg})           # exclude the one-off caching from the timing
    r = run_experiment({**cfg, "name": "_timing", "epochs": n_epochs}, record=False, verbose=False)
    per_epoch = r["train_time_s"] / n_epochs
    n_full = {**BASE_CONFIG, **cfg}["epochs"]
    full = per_epoch * n_full
    print(f"{per_epoch:.1f} s per epoch  ->  a full run of {n_full} epochs: about {full / 60:.1f} min")
    return full / 60

# %% [markdown]
# ### 2.6 TensorBoard (and what to do when it does not work)
# Every run writes scalars (train loss, val Dice, learning rate) to `WORK_DIR/runs/<name>`. TensorBoard shows all runs on the
# same axes while they train.
#
# * **Colab**: works out of the box. In a cell: `%load_ext tensorboard` then `%tensorboard --logdir outputs/runs`
#   (use your `WORK_DIR`).
# * **VS Code**: with the Python extension, run the command *"Python: Launch TensorBoard"* and select the `runs` folder.
# * **Kaggle**: the inline magic is not reliable (it is not a supported feature of Kaggle notebooks). Use the **fallback**:
#   the helpers `compare(...)` and `learning_curve(...)` plot the same curves from the saved histories. You can also
#   download the `runs` folder (it is small) and run `tensorboard --logdir runs` on your laptop.
#
# The cell below tries the magic outside FAST mode and does nothing harmful if it fails.

# %%
if not FAST:
    try:
        get_ipython().run_line_magic("load_ext", "tensorboard")
        get_ipython().run_line_magic("tensorboard", f"--logdir {WORK_DIR / 'runs'}")
    except Exception as err:   # Kaggle, plain scripts...
        print(f"TensorBoard inline not available here ({type(err).__name__}); use compare() / learning_curve() instead.")

# %% [markdown]
# ### ✏️ Exercise 2.4 - your compute budget
# Time **one epoch** of the baseline config, then work out how many full runs fit in your budget.
# Planning rule for today: about **90 minutes of GPU time** for all menu experiments together.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `estimate_run_time(cfg)` returns the minutes of one *full* run (it times one epoch and multiplies).
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `minutes_per_run = estimate_run_time(BASE_CONFIG)` and `max_runs = int(BUDGET_MIN // minutes_per_run)`
#
# </details>

# %%
BUDGET_MIN = 90
#@solution
minutes_per_run = estimate_run_time(BASE_CONFIG)
max_runs = int(BUDGET_MIN // max(minutes_per_run, 1e-3))
#@student
#| minutes_per_run = ____   # TODO: time one epoch and project a full run (minutes)
#| max_runs = ____          # TODO: how many full runs fit in BUDGET_MIN?
#@end
print(f"-> about {max_runs} full runs fit in {BUDGET_MIN} min. Small-data and small-image runs are much cheaper: plan accordingly.")

# %%
# Check 2.4
assert minutes_per_run > 0 and isinstance(max_runs, int) and max_runs >= 0
print("✅ Exercise 2.4 passed")

# %% [markdown] tags=["solution-only"]
# **Teacher note.** On a Kaggle T4 (256x256, ~620 training slices, batch 16, 30 epochs, no AMP) expect roughly 2-5 minutes per run
# (~5-10 s/epoch; please measure before class - it depends on the GPU lottery and on data loading from `/kaggle/input`).
# That gives ~20-40 full runs in 90 minutes; the 512x512 run costs about 4x more per epoch, so do it with fewer epochs or skip it.

# %% [markdown]
# ## 3. Experiment 0 (mandatory): reproduce the baseline and measure the noise
# Before changing anything, check that the harness reproduces notebook 03 (or 04's teachers' model) and measure how much
# the result moves **when nothing changes except the random seed**. This number calibrates every later comparison.
#
# ### ✏️ Exercise 3.1 - the baseline through the harness
# Run `BASE_CONFIG` under the name `baseline-s42` (seed 42), **and the same config with seed 43** under `baseline-s43`.
# Compare the first with your Dice of day 2 (it should be close; small differences are due to the GPU being non-deterministic
# and to the slightly different training loop).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# A config is a dictionary: `{**BASE_CONFIG, "name": ..., "seed": ...}` copies the baseline and overrides two keys.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `run_experiment({**BASE_CONFIG, "name": "baseline-s43", "seed": 43})`
#
# </details>

# %%
#@solution
r_base = run_experiment({**BASE_CONFIG, "name": "baseline-s42", "seed": 42})
r_seed = run_experiment({**BASE_CONFIG, "name": "baseline-s43", "seed": 43})
#@student
#| r_base = run_experiment({**BASE_CONFIG, "name": "baseline-s42", "seed": 42})
#| r_seed = run_experiment({**BASE_CONFIG, "name": "baseline-s43", "seed": ____})   # TODO: a different seed
#@end

# %%
# Check 3.1
assert r_base["config"]["seed"] != r_seed["config"]["seed"], "use two different seeds"
assert changed_keys(r_seed["config"], r_base["config"]) == {"seed"}, "only the seed may differ"
print(f"✅ Exercise 3.1 passed: val Dice {r_base['best_val_dice']:.4f} (seed 42) vs {r_seed['best_val_dice']:.4f} (seed 43)")
if not FAST:
    print("day-2 reference: compare with the val Dice you obtained in notebook 03")

# %% [markdown]
# ### ✏️ Exercise 3.2 - the noise floor
# Compute `NOISE`: the spread (max - min) of the best val Dice over the baseline runs with different seeds. From now on
# `compare()` marks a difference as `beyond_noise` only if it is larger than this.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Select the rows whose name starts with `baseline-s` in `load_results()` and look at the column `best_val_dice`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `d = res.loc[res.index.str.startswith("baseline-s"), "best_val_dice"]` and `NOISE = d.max() - d.min()`
#
# </details>

# %%
res = load_results()
seed_runs = res.loc[res.index.str.startswith("baseline-s"), "best_val_dice"]
#@solution
NOISE = float(seed_runs.max() - seed_runs.min())
#@student
#| NOISE = ____   # TODO: spread (max - min) of seed_runs
#@end
print(seed_runs.round(4).to_dict())
print(f"seed-to-seed noise: {NOISE:.4f} Dice")

# %%
# Check 3.2
assert NOISE >= 0
print("✅ Exercise 3.2 passed")

# %% [markdown]
# > 🧠 **Question:** with two seeds the "spread" is a very crude noise estimate. How would you estimate it better,
# > and what does it mean for an experiment that shows +0.002 Dice over the baseline? What if it shows +0.02?

# %% [markdown] tags=["solution-only"]
# **Answer.** Use more seeds (3-5) and report mean +/- standard deviation, or - better - a paired comparison per validation case
# with a confidence interval. If the seed noise is of the order of 0.002-0.005 Dice (typical for this task: lung segmentation is
# nearly saturated), a +0.002 difference is indistinguishable from "a different seed" and should be reported as "no
# evidence of an effect". +0.02 would be large relative to the noise, but then ask whether it is a bug (leakage!) before celebrating.
# Note also that the validation set is finite (~100 patients): its sampling error adds to the noise.
#
# **Expected on real data (qualitative, check against your run).** Baseline val Dice around 0.97-0.99, seed spread of a few
# thousandths. Some slices (e.g. very high or low slices with little lung, or with consolidation) are much harder than the rest,
# so the *mean* hides a tail: that is what notebook 04 showed.

# %% [markdown]
# ## 4. The menu of experiments
# Pick by level and by interest. Each experiment is self-contained and has the same shape:
#
# **motivation -> hypothesis (write it down!) -> config diff -> run -> compare -> 🧠 conclusion**
#
# | Exp. | Question | Level | Cost (real) |
# |---|---|---|---|
# | **A** | Does augmentation help, especially with little data? | ✏️ mandatory | 3 runs |
# | **B** | How many annotated patients do we really need? | ✏️ mandatory | 4 short runs |
# | **C** | Which loss function? | ✏️ mandatory (or D / E: pick at least one of C-E) | 2-3 runs |
# | **D** | Architecture and capacity: is bigger better? | 🚀 additional | 3-6 runs |
# | **E** | Optimisation: learning rate, scheduler, AMP... | 🚀 additional | 2-6 runs |
# | **F** | Input resolution: 128 / 256 / 512 | 🚀 additional | 2 runs (512 is slow) |
# | **G** | Multi-class: right lung / left lung / trachea | 🚀 additional | 1 run |
# | **H** | Transfer learning, multi-window input | 🌶️ optional | 1-2 runs |
# | **I** | Optuna, cross-validation, ensembling | 🌶️ optional | open |
# | **J** | Tracking with MLflow / Weights & Biases | 🌶️ optional | open |
#
# Work in pairs if you like: split the menu, and compare notes (not the code: the *tables*).
# Everything goes through `run_experiment`, so everything ends up in `experiments.csv`.

# %% [markdown]
# ## Experiment A (mandatory): augmentation on vs off
# **Motivation.** Augmentation (random rotations, scaling, shifts, noise) invents plausible new training images. It is
# supposed to reduce **overfitting**, i.e. a model that memorises its training patients. How much it helps should depend on how much
# data you have: with 600 patients maybe little, with 60 a lot. We test both.
#
# **Hypothesis (write it before you run!).** Use the template: *"I expect ... because ...; the effect will be larger/smaller with
# little data, because ..."*

# %% [markdown] tags=["student-only"]
# **My hypothesis:** _(double-click and write it here before running the next cells)_

# %% [markdown] tags=["solution-only"]
# **My hypothesis (example):** with the full training set the difference will be within the seed noise, because 600 slices already
# cover the variability of lung shapes; with 10% of the patients augmentation should help by a visible margin, because the
# unaugmented model memorises the few training shapes (train loss keeps falling while the val Dice stalls).

# %% [markdown]
# ### ✏️ Exercise A.1 - config diffs
# Two comparisons, each changes **one** thing from its own control:
# 1. full data: baseline (augment on) vs `aug-off`;
# 2. little data (`SMALL_FRAC` of the patients): `small-aug-on` vs `small-aug-off`.
#
# Fill in the config diffs. The check cell verifies that you changed exactly one thing.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `augment` is a boolean in the config. The "small" runs also set `train_fraction`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `{**BASE_CONFIG, "name": "aug-off", "augment": False}`
#
# </details>

# %%
SMALL_FRAC = 0.25 if FAST else 0.1
#@solution
cfg_aug_off = {**BASE_CONFIG, "name": "aug-off", "augment": False}
cfg_small_on = {**BASE_CONFIG, "name": "small-aug-on", "train_fraction": SMALL_FRAC, "augment": True}
cfg_small_off = {**BASE_CONFIG, "name": "small-aug-off", "train_fraction": SMALL_FRAC, "augment": False}
#@student
#| cfg_aug_off = {**BASE_CONFIG, "name": "aug-off", "augment": ____}                                  # TODO
#| cfg_small_on = {**BASE_CONFIG, "name": "small-aug-on", "train_fraction": SMALL_FRAC, "augment": ____}   # TODO
#| cfg_small_off = {**BASE_CONFIG, "name": "small-aug-off", "train_fraction": SMALL_FRAC, "augment": ____}  # TODO
#@end
for c in (cfg_aug_off, cfg_small_on, cfg_small_off):
    run_experiment(c)

# %%
# Check A.1
small_ref = {**BASE_CONFIG, "train_fraction": SMALL_FRAC}
assert changed_keys(cfg_aug_off) == {"augment"} and cfg_aug_off["augment"] is False
assert changed_keys(cfg_small_on, small_ref) == set() and changed_keys(cfg_small_off, small_ref) == {"augment"}
print("✅ Exercise A.1 passed")

# %%
compare(["baseline-s42", "aug-off", "small-aug-on", "small-aug-off"], "A: augmentation")
# Overfitting check: train loss vs val Dice for the little-data runs (look at the curves above)

# %% [markdown]
# > 🧠 **Conclusion (write 2-3 sentences):** was your hypothesis confirmed *beyond the noise*? Is the effect different with little
# > data? What would you look at to see overfitting (hint: the training loss)?
#
# > ⚠️ **Pitfall:** augmentation must be **medically plausible**. A 180-degree rotation or a vertical flip produces chest CTs
# > that never occur (heart on the wrong side, spine on top). Our augmentation is mild on purpose. We come back to flips in G.

# %% [markdown] tags=["solution-only"]
# **Discussion.** *Expected on real data (qualitative, not measured here).* With the full training set the difference between
# augmentation on and off is typically within a few thousandths of Dice - about the seed noise - because the task is easy and
# 600 slices are plenty. With 10% of the patients (~60) the unaugmented model overfits earlier (training loss lower, val Dice
# plateauing) and augmentation usually helps by a small but visible margin (order of a point of Dice, or less). Honest message:
# *for this easy task most variants saturate around 0.97-0.99; the differences live in the small-data regime and in the
# hard cases (tails, boundaries), not in the mean on the full data.* If a pair of students finds the opposite, check seeds first.
# In FAST mode (synthetic ellipses, 2 epochs) the numbers mean nothing: only the plumbing is tested.

# %% [markdown]
# ## Experiment B (mandatory): data efficiency, the learning curve
# **Motivation.** Annotating a CT slice takes an expert minutes; annotating hundreds of patients takes months. Before
# paying for it we want to know the shape of the **learning curve**: validation performance as a function of the number of
# annotated patients. Plot it with a logarithmic x axis: gains usually follow the number of *doublings* of data.
#
# **Hypothesis.** Where do you expect the curve to flatten (5 patients? 50? 500?) and what Dice do you expect with 5% of the data?

# %% [markdown] tags=["student-only"]
# **My hypothesis:** _(write it here)_

# %% [markdown] tags=["solution-only"]
# **My hypothesis (example):** Dice rises steeply up to ~10% of the patients (about 60) and then flattens; with 5% I expect about
# 0.95; the last doublings (50% -> 100%) will gain less than the seed noise.

# %% [markdown]
# ### ✏️ Exercise B.1 - the learning curve
# Train on 5%, 10%, 25% and 50% of the training patients (the 100% point is your baseline run `baseline-s42`). The runs are
# short because the data are small, so you can afford them.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Loop over the fractions, and give every run a name that contains the fraction, e.g. `frac-0.25`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `run_experiment({**BASE_CONFIG, "name": f"frac-{f}", "train_fraction": f})`
#
# </details>

# %%
FRACTIONS = [0.25, 0.5] if FAST else [0.05, 0.1, 0.25, 0.5]
#@solution
for f in FRACTIONS:
    run_experiment({**BASE_CONFIG, "name": f"frac-{f}", "train_fraction": f})
#@student
#| for f in ____:   # TODO: the list of fractions
#|     run_experiment({**BASE_CONFIG, "name": f"frac-{f}", "train_fraction": ____})   # TODO
#@end

# %%
# Check B.1
_res = load_results()
for f in FRACTIONS:
    assert f"frac-{f}" in _res.index, f"run frac-{f} is missing"
    assert abs(_res.loc[f"frac-{f}", "train_fraction"] - f) < 1e-9
print("✅ Exercise B.1 passed")


# %%
def learning_curve(names, title="B: learning curve"):
    """Best val Dice vs number of training patients (log x), with the seed-noise band around the full-data baseline."""
    res = load_results().loc[names].sort_values("n_train")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res["n_train"], res["best_val_dice"], "o-")
    for n, d in zip(res["n_train"], res["best_val_dice"]):
        ax.annotate(f"{d:.3f}", (n, d), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    if NOISE is not None:
        top = res["best_val_dice"].iloc[-1]
        ax.axhspan(top - NOISE, top + NOISE, color="tab:orange", alpha=0.3, label="full data +/- seed noise")
        ax.legend()
    ax.set(xscale="log", xlabel="number of training patients (log)", ylabel="best val Dice", title=title)
    ax.grid(alpha=0.3, which="both")
    plt.show()
    display(res[["n_train", "train_fraction", "best_val_dice", "train_time_s"]].round(4))


learning_curve([f"frac-{f}" for f in FRACTIONS] + ["baseline-s42"])

# %% [markdown]
# > 🧠 **Conclusion:** how many annotated patients did we *really* need to get within the seed noise of the full-data model? What
# > would that mean for a **hospital that wants to build its own model** (cost of annotation, which patients to annotate,
# > what if the hospital's CT scanner differs from LUNA16's)?
# >
# > ⚠️ **Pitfall:** the validation set is the same for every point, so the *curve* is comparable; but with 5% of the patients the
# > result depends heavily on **which** patients you drew. A single run per point has a large error bar there.

# %% [markdown] tags=["solution-only"]
# **Discussion.** *Expected (qualitative).* For lung segmentation the curve rises very quickly and is flat well before 100%:
# a few dozen patients already give a Dice in the high 0.9s; the last doublings gain less than the noise. For a hospital: a
# few dozen well-chosen, carefully annotated cases may suffice to *adapt* a model (fine-tune a public one), and the budget
# is better spent on *diversity* (scanners, kernels, pathologies) and on a good *validation set* than on more of the same
# easy slices. It is also the place to mention active learning and pre-annotation with an existing model (MONAI Label),
# with the silver-standard caveat: our labels came from an algorithm, not from experts. For harder tasks (nodules, tumours)
# the curve is far less saturated: do not generalise "50 patients are enough".
# Extra point: nested subsets and one seed - the 5% point has a big error bar.

# %% [markdown]
# ## Experiment C (mandatory, or D / E): loss functions
# **Motivation.** The loss is the only thing the optimiser "wants". **Dice loss** optimises the overlap directly (and is insensitive to
# class imbalance), **cross-entropy** gives smooth per-pixel gradients, **Dice+CE** (the baseline) combines both. **Focal** and
# **Tversky** emphasise hard pixels or missed foreground. For a large organ like the lung, with ~20% of the pixels, the imbalance
# is mild: what do you expect?
#
# **Hypothesis.** Write which loss you expect to be best, worst, and why.

# %% [markdown] tags=["student-only"]
# **My hypothesis:** _(write it here)_

# %% [markdown] tags=["solution-only"]
# **My hypothesis (example):** all three end within the noise on the mean Dice because lungs are large and easy; Dice alone converges
# more slowly at the very beginning (flat gradients while predictions are poor) and CE alone is slightly worse at the borders.

# %% [markdown]
# ### ✏️ Exercise C.1 - three losses
# Train with `dice` and `ce` (the baseline already is `dice_ce`). Then, if you have time, try `dice_focal` or `tversky` too
# (each costs one run).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The keys of the dictionary `LOSSES` are the valid names of `cfg["loss"]`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `{**BASE_CONFIG, "name": "loss-dice", "loss": "dice"}`
#
# </details>

# %%
#@solution
cfg_dice = {**BASE_CONFIG, "name": "loss-dice", "loss": "dice"}
cfg_ce = {**BASE_CONFIG, "name": "loss-ce", "loss": "ce"}
#@student
#| cfg_dice = {**BASE_CONFIG, "name": "loss-dice", "loss": ____}   # TODO
#| cfg_ce = {**BASE_CONFIG, "name": "loss-ce", "loss": ____}       # TODO
#@end
loss_runs = [cfg_dice, cfg_ce]
if not FAST:
    loss_runs += [{**BASE_CONFIG, "name": "loss-tversky", "loss": "tversky"}]   # 🚀 optional third loss
for c in loss_runs:
    run_experiment(c)

# %%
# Check C.1
assert changed_keys(cfg_dice) == {"loss"} and changed_keys(cfg_ce) == {"loss"}
assert {cfg_dice["loss"], cfg_ce["loss"]} == {"dice", "ce"}
print("✅ Exercise C.1 passed")

# %%
compare(["baseline-s42"] + [c["name"] for c in loss_runs], "C: loss functions");

# %% [markdown]
# > 🧠 **Conclusion:** is any loss better than the others *beyond the noise*? Look at the curves too: do some converge faster? Mean
# > Dice is a coarse summary: which loss would you expect to behave differently for **tiny** structures (a nodule, the trachea in G)?
#
# > 📚 **Read more:** "Loss odyssey in medical image segmentation" (Ma et al., Medical Image Analysis 2021),
# > https://doi.org/10.1016/j.media.2021.102035 - 20 losses compared; the conclusions are more nuanced than the claims of single papers.

# %% [markdown] tags=["solution-only"]
# **Discussion.** *Expected (qualitative).* On real lungs the three losses land within a few thousandths of Dice of each other; CE alone
# is often slightly worse on the *boundary* metrics and slower to reach a high Dice, Dice-only trains less smoothly early on.
# The lesson: loss choice matters most for **small structures and strongly imbalanced classes** (e.g. the trachea in G, nodules),
# not for lungs. Tversky with beta > alpha trades false positives for fewer missed pixels: useful when under-segmentation is
# the costly error (e.g. lesion detection), less so here.

# %% [markdown]
# ## 🚀 Experiment D (additional): architecture and capacity
# **Motivation.** Is a bigger network better? `UNet` has two knobs: `channels` (width and depth) and `num_res_units` (residual units
# per level). MONAI also provides `AttentionUnet`, `SegResNet` and `BasicUNet`. Any class of `monai.networks.nets` works with the
# harness, because the **contract** is `arch` + `arch_kwargs`. What counts for the **web app** is not only Dice but also
# parameters and speed: the app will run on a *small CPU machine*.
#
# **Hypothesis.** Predict the ranking by Dice, and by parameters. Do you expect the small U-Net to lose much?

# %% [markdown] tags=["student-only"]
# **My hypothesis:** _(write it here)_

# %% [markdown] tags=["solution-only"]
# **My hypothesis (example):** a U-Net with 4x fewer channels per layer loses less than the noise; the large one gains nothing; residual
# units matter little; attention gates do not help a task where the target is big and contrasted.

# %% [markdown]
# ### ✏️ Exercise D.1 - a small and a different network
# 1. `unet-small`: a U-Net with narrower layers: `channels = (8, 16, 32, 64, 128)` (half of the baseline), same `num_res_units`.
# 2. `basicunet`: another family. Use `ARCH_ZOO["basicunet"]`.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `unet_kwargs(channels, num_res_units)` returns a complete `arch_kwargs` dictionary; `ARCH_ZOO[name]` is a pair `(arch, arch_kwargs)`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `{**BASE_CONFIG, "name": "unet-small", "arch_kwargs": unet_kwargs((8, 16, 32, 64, 128), 2)}`
#
# </details>

# %%
ARCH_ZOO = {   # name: (class in monai.networks.nets, kwargs) - all 2D, 1 input channel, 2 classes
    "unet-large": ("UNet", unet_kwargs((32, 64, 128, 256, 512), 2)),
    "unet-nores": ("UNet", unet_kwargs((16, 32, 64, 128, 256), 0)),
    "attunet": ("AttentionUnet", dict(spatial_dims=2, in_channels=1, out_channels=2, channels=[16, 32, 64, 128], strides=[2, 2, 2])),
    "segresnet": ("SegResNet", dict(spatial_dims=2, in_channels=1, out_channels=2, init_filters=16, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1])),
    "basicunet": ("BasicUNet", dict(spatial_dims=2, in_channels=1, out_channels=2, features=[16, 32, 64, 128, 256, 16])),
}
#@solution
cfg_small = {**BASE_CONFIG, "name": "unet-small", "arch_kwargs": unet_kwargs((8, 16, 32, 64, 128), 2)}
arch, kwargs = ARCH_ZOO["basicunet"]
cfg_basic = {**BASE_CONFIG, "name": "basicunet", "arch": arch, "arch_kwargs": kwargs}
#@student
#| cfg_small = {**BASE_CONFIG, "name": "unet-small", "arch_kwargs": unet_kwargs(____, 2)}   # TODO: half the channels
#| arch, kwargs = ARCH_ZOO[____]                                                             # TODO: the BasicUNet entry
#| cfg_basic = {**BASE_CONFIG, "name": "basicunet", "arch": arch, "arch_kwargs": kwargs}
#@end
arch_runs = [cfg_small, cfg_basic]
if not FAST:   # 🚀 the rest of the zoo
    arch_runs += [{**BASE_CONFIG, "name": n, "arch": a, "arch_kwargs": k} for n, (a, k) in ARCH_ZOO.items() if n != "basicunet"]
for c in arch_runs:
    run_experiment(c)

# %%
# Check D.1
assert cfg_small["arch_kwargs"]["channels"] == [8, 16, 32, 64, 128] or tuple(cfg_small["arch_kwargs"]["channels"]) == (8, 16, 32, 64, 128)
assert cfg_basic["arch"] == "BasicUNet" and changed_keys(cfg_small) == {"arch_kwargs"}
print("✅ Exercise D.1 passed")


# %%
def scatter_params_time(names):
    """Dice vs number of parameters (log x); marker area = training time."""
    t = load_results().loc[names]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(t["n_params"], t["best_val_dice"], s=60 + 400 * t["train_time_s"] / t["train_time_s"].max(), alpha=0.6)
    for n, row in t.iterrows():
        ax.annotate(n, (row["n_params"], row["best_val_dice"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set(xscale="log", xlabel="parameters (log)", ylabel="best val Dice", title="D: Dice vs size (marker area = training time)")
    ax.grid(alpha=0.3, which="both")
    plt.show()


arch_names = ["baseline-s42"] + [c["name"] for c in arch_runs]
compare(arch_names, "D: architectures")
scatter_params_time(arch_names)

# %% [markdown]
# > 🧠 **Conclusion:** is bigger better here? Which model gives the best Dice **per parameter / per second**? If the web app runs on a
# > machine with 1 CPU core, which of these would you ship, and what extra measurement (not Dice!) would you do before deciding
# > (hint: time per image on CPU, notebook 06)?

# %% [markdown] tags=["solution-only"]
# **Discussion.** *Expected (qualitative).* Most architectures land within the noise band; the smallest U-Net typically loses almost
# nothing with a fraction of the parameters (and runs several times faster on CPU), the large U-Net does not pay back its 4x
# parameters. This is the nnU-Net message: *for a well-posed task, the pipeline and the data matter more than the architecture*.
# Differences, if any, appear on hard cases and boundaries (see F). For deployment the model with the best trade-off is usually
# the small one; measure CPU latency before deciding. `AttentionUnet` and `SegResNet` need image sizes divisible by 8 or 16.

# %% [markdown]
# ## 🚀 Experiment E (additional): optimisation
# **Motivation.** The learning rate is the most important hyper-parameter: too high and training diverges or is noisy, too low and
# 30 epochs are not enough. Others: the **scheduler** (cosine decay lets the network settle), the **batch size**, **AdamW with weight decay**
# and **mixed precision (AMP)** which on a GPU trades a tiny numerical difference for a big speed-up and lower memory.
#
# **Hypothesis.** Which learning rate is best among `1e-2, 1e-3, 1e-4`? What happens at `1e-2`?

# %% [markdown] tags=["student-only"]
# **My hypothesis:** _(write it here)_

# %% [markdown] tags=["solution-only"]
# **My hypothesis (example):** 1e-3 (Adam default) is best or tied; 1e-2 is unstable early (noisy or stuck curve) but may still recover with
# the cosine decay; 1e-4 is stable but under-trained after 30 epochs (still improving at the end).

# %% [markdown]
# ### ✏️ Exercise E.1 - learning-rate sweep
# Run the learning rates `1e-2` and `1e-4` (the baseline is `1e-3`). Use names like `lr-0.01`.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# One loop over a list of learning rates; the config key is `lr`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `for lr in (1e-2, 1e-4): run_experiment({**BASE_CONFIG, "name": f"lr-{lr}", "lr": lr})`
#
# </details>

# %%
LRS = (1e-2,) if FAST else (1e-2, 1e-4)
#@solution
for lr in LRS:
    run_experiment({**BASE_CONFIG, "name": f"lr-{lr}", "lr": lr})
#@student
#| for lr in LRS:
#|     run_experiment({**BASE_CONFIG, "name": f"lr-{lr}", "lr": ____})   # TODO
#@end

# %%
# Check E.1
assert all(f"lr-{lr}" in load_results().index for lr in LRS)
assert all(load_results().loc[f"lr-{lr}", "lr"] == lr for lr in LRS)
print("✅ Exercise E.1 passed")

# %%
compare(["baseline-s42"] + [f"lr-{lr}" for lr in LRS], "E: learning rate");

# %% [markdown]
# ### 🚀 Challenge E.2 (additional): scheduler, batch size, AdamW, mixed precision
# One run each, **one change at a time**. AMP only makes sense on a GPU: the cell skips it on CPU. Look at the **time** and
# **memory** columns for AMP: it is an optimisation that should not change the Dice (beyond noise) but should be faster.

# %%
if not FAST:
    extra = [
        {"name": "sched-none", "scheduler": "none"},
        {"name": "bs-8", "batch_size": 8},
        {"name": "bs-32", "batch_size": 32},
        {"name": "adamw-wd", "optimizer": "adamw", "weight_decay": 1e-2},
    ]
    if device.type == "cuda":
        extra += [{"name": "amp", "amp": True}]
    for e in extra:
        run_experiment({**BASE_CONFIG, **e})
    t = compare(["baseline-s42"] + [e["name"] for e in extra], "E: scheduler, batch size, AdamW, AMP")
    display(load_results().loc[["baseline-s42"] + [e["name"] for e in extra], ["train_time_s", "peak_mem_mb"]])
else:
    print("skipped in FAST mode (scheduler / batch size / AdamW / AMP runs)")

# %% [markdown]
# > 🧠 **Conclusion:** which knob mattered most? A learning-rate sweep on a *single* seed can mislead: what would you do to be sure that
# > 3e-4 is better than 1e-3? Why is a **batch size** change not a "pure" experiment when the learning rate is kept fixed?

# %% [markdown] tags=["solution-only"]
# **Discussion.** *Expected (qualitative).* Adam with 1e-3 and cosine decay is a robust default; 1e-2 is noisier but often still reaches
# a similar Dice thanks to the decay; 1e-4 is slower and a bit worse after the same number of epochs. Removing the scheduler gives
# a noisier last epochs. AMP: same Dice within noise, noticeably faster and less memory on a T4 (tensor cores), little or no gain on small
# images where the GPU is under-used. Batch size: the number of updates per epoch changes with it (a bigger batch = fewer steps), so lr
# and batch size interact: a fair comparison would retune the learning rate (linear-scaling rule as a starting point).

# %% [markdown]
# ## 🚀 Experiment F (additional): input resolution
# **Motivation.** 512x512 is the native resolution, but we train at 256 (4x cheaper). Resizing the label with nearest neighbour and the
# image with bilinear interpolation loses detail at the lung border, and **Dice at low resolution is generous**: a one-pixel error at
# 128x128 is 4 pixels of the original image. So we compare resolutions on equal terms: every model predicts at its own resolution, the
# prediction is **upsampled to the native grid**, and Dice and **HD95 (in mm)** are computed there.
#
# **Hypothesis.** How much do you expect Dice and HD95 to change from 128 to 256 to 512? And time and memory?

# %% [markdown] tags=["student-only"]
# **My hypothesis:** _(write it here)_

# %% [markdown] tags=["solution-only"]
# **My hypothesis (example):** Dice at native resolution barely changes (the lung is a big object) but HD95 improves with resolution as
# the border is placed more precisely; the time grows ~4x per doubling; 512 may need a smaller batch to fit in memory.

# %% [markdown]
# ### ✏️ Exercise F.1 - resolution sweep
# Train at the other resolutions (the baseline is `IMG_SIZE`) and evaluate each model at native resolution on the **validation** set.
# On real data use 128 and 512 (**512 is slow: use `batch_size=8` and say so in your notes - it is then not a one-change experiment!**). In FAST
# mode the sweep is 32 vs 64.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The config key is `img_size`. `evaluate_model_dir(WORK_DIR / "models" / name, val_files)` returns a per-case DataFrame.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `{**BASE_CONFIG, "name": f"img-{s}", "img_size": s}`
#
# </details>

# %%
SIZES = [32] if FAST else [128, 512]
#@solution
for s in SIZES:
    run_experiment({**BASE_CONFIG, "name": f"img-{s}", "img_size": s, **({"batch_size": 8} if s >= 512 else {})})
#@student
#| for s in SIZES:
#|     run_experiment({**BASE_CONFIG, "name": f"img-{s}", "img_size": ____})   # TODO (add batch_size=8 for 512 if memory is short)
#@end

# %%
# Check F.1
assert all(f"img-{s}" in load_results().index for s in SIZES)
print("✅ Exercise F.1 passed")

# %%
rows = []
for name in ["baseline-s42"] + [f"img-{s}" for s in SIZES]:
    ev = evaluate_model_dir(WORK_DIR / "models" / name, val_files)
    r = load_results().loc[name]
    rows.append({"model": name, "img_size": r["img_size"], "val dice (train res)": r["best_val_dice"],
                 "dice native (mean)": ev["dice_lung"].mean(), "dice native (worst case)": ev["dice_lung"].min(),
                 "HD95 mm (mean)": ev["hd95_mm"].mean(), "HD95 mm (95th pct)": ev["hd95_mm"].quantile(0.95),
                 "time s": r["train_time_s"], "peak GPU MB": r["peak_mem_mb"]})
display(pd.DataFrame(rows).set_index("model").round(3))
compare(["baseline-s42"] + [f"img-{s}" for s in SIZES], "F: resolution");

# %% [markdown]
# > 🧠 **Conclusion:** which metric separates the resolutions better, Dice or HD95? What would you do for a clinical measurement (lung
# > volume or area in cm2) that depends on the **exact border**?

# %% [markdown] tags=["solution-only"]
# **Discussion.** *Expected (qualitative).* Mean Dice at native resolution changes very little between 128, 256 and 512; **HD95 and the worst
# case** are more sensitive: low resolution blurs the border and thin structures. Time grows roughly with the number of pixels
# (4x per doubling), and memory too. For this application 256 is the sweet spot. The comparison "Dice at training resolution" in the
# first column is *not* fair across resolutions; that is why we re-evaluate at native resolution. Cost vs benefit for a clinical area
# measurement: a 1-pixel border error at 256 is ~2.7 mm with 0.7 mm native pixels, which is only relevant for fine measurements.

# %% [markdown]
# ## 🚀 Experiment G (additional): multi-class - right lung, left lung, trachea
# **Motivation.** The dataset has four labels: background, right lung, left lung, trachea/main airways. Going from binary to multi-class
# changes **three** things: (1) the label transform (do **not** collapse the labels: `multiclass=True`), (2) the network head
# (`out_channels=4`), (3) the metric (Dice **per class**, `include_background=False` and `reduction="mean_batch"`). The harness already
# switches on `out_channels`; the per-class Dice per epoch is stored in the history.
#
# **Hypothesis.** Which class will be the hardest, and why? Two candidates: the *trachea* (tiny, thin) and *left vs right lung* (two
# structures that look alike and are separated only by position).

# %% [markdown] tags=["student-only"]
# **My hypothesis:** _(write it here)_

# %% [markdown] tags=["solution-only"]
# **My hypothesis (example):** the trachea is the hardest by Dice (tiny: a few hundred pixels, one pixel of error costs a lot), while the two lungs
# are easy except for rare *swap* errors (a lung labelled left in its upper part and right in the lower), because left and right are decided by
# global position, not by local appearance.

# %% [markdown]
# > ⚠️ **Pitfall: the left-right flip.** A horizontal flip is a popular augmentation. With a **binary** lung mask it is harmless
# > (the lung stays a lung). With **multi-class** labels a flip turns the right lung into something on the left side of the image *that is
# > still labelled "right lung"*: the network gets contradictory training signals about what "left" and "right" mean. (And the heart and the liver are on
# > one specific side: a flip creates a *situs inversus* that does not occur in the test data.) Look:
#
# ### ✏️ Exercise G.1 - see the pitfall, then fix it
# Flip a sample along the x axis (axis 0, which is patient left-right in our LPS+ arrays) and plot it. Then **swap the label values 1 and 2**
# after the flip so that the flipped label becomes consistent.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# After the flip, where the mask said 1 it should say 2 and vice versa: a swap needs a temporary mask or a lookup table.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `swapped = torch.where(flipped == 1, 2, torch.where(flipped == 2, 1, flipped))`
#
# </details>

# %%
sample = get_transforms(IMG_SIZE, multiclass=True)(train_files[0])
img, lab = sample["image"][0], sample["label"][0]
flipped_img, flipped_lab = torch.flip(img, dims=[0]), torch.flip(lab, dims=[0])
#@solution
swapped = torch.where(flipped_lab == 1, 2, torch.where(flipped_lab == 2, 1, flipped_lab))
#@student
#| swapped = ____   # TODO: exchange the label values 1 <-> 2 in flipped_lab
#@end
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
for a, (im, lb, t) in zip(ax, [(img, lab, "original"), (flipped_img, flipped_lab, "flipped, labels NOT swapped"), (flipped_img, swapped, "flipped, labels swapped")]):
    a.imshow(im.T, cmap="gray")
    a.imshow(np.ma.masked_equal(lb.T.numpy(), 0), cmap="Set1", alpha=0.5, vmin=1, vmax=9, interpolation="nearest")
    a.set_title(t, fontsize=9)
    a.axis("off")
plt.suptitle("label 1 (red) = right lung should be on the LEFT of the screen (radiological view)")
plt.show()

# %%
# Check G.1
assert swapped.shape == flipped_lab.shape
assert ((swapped == 1) == (flipped_lab == 2)).all() and ((swapped == 2) == (flipped_lab == 1)).all() and ((swapped == 3) == (flipped_lab == 3)).all()
print("✅ Exercise G.1 passed")

# %% [markdown]
# ### Train the 4-class model
# Only two things change in the config: `out_channels=4` in the network. The data pipeline and the metrics adapt automatically
# (`multiclass` is inferred from `out_channels`). The Dice in the table is the **mean over the three foreground classes**: it is *not* comparable to
# the binary runs, so we compare the lungs on equal terms with the native-resolution evaluation (`dice_lung` merges classes 1 and 2).

# %%
cfg_multi = {**BASE_CONFIG, "name": "multiclass", "arch_kwargs": unet_kwargs(out_channels=4)}
r_multi = run_experiment(cfg_multi)
ev_multi = evaluate_model_dir(WORK_DIR / "models" / "multiclass", val_files)
ev_bin = evaluate_model_dir(WORK_DIR / "models" / "baseline-s42", val_files)
per_class = ev_multi[["dice_c1", "dice_c2", "dice_c3"]].rename(columns={"dice_c1": "right lung", "dice_c2": "left lung", "dice_c3": "trachea"})
display(pd.DataFrame({"binary baseline: lung": ev_bin["dice_lung"].describe(), "4-class: lung (1+2)": ev_multi["dice_lung"].describe()}).round(4))
display(per_class.describe().round(4))
per_class.plot.box(figsize=(5, 3.5), title="G: per-class Dice, validation set", ylabel="Dice")
plt.show()

# %% [markdown]
# > 🧠 **Conclusion:** which class is the hardest and why? Look at the minimum of right/left lung: do you see cases where **left and right are mixed up**? Why does
# > separating left from right need *global* context (receptive field!) while "lung vs not lung" can be decided locally? Is the lung Dice of the
# > 4-class model better or worse than the binary one?

# %% [markdown] tags=["solution-only"]
# **Discussion.** *Expected (qualitative).* Right and left lung have a Dice close to the binary one; the **trachea is clearly the lowest** (small
# object of a few hundred pixels: one boundary pixel is a big fraction of the area; also the labels were produced by an algorithm and the main bronchi are at the
# limit of what it segments). The lowest lung scores (outliers) are often left/right swaps in slices where the two lungs touch or one of them is almost
# absent: deciding "left or right" cannot be done from the local texture, the network must "see" the whole thorax; with the receptive field of a small U-Net at 256 px this is
# possible but fragile. Multi-class training can help or slightly hurt the lung Dice (it makes the task harder but adds anatomical supervision): usually within the noise.
# The web app of day 5 can show multi-class models because `config.json` carries the `labels` dictionary.

# %% [markdown]
# ## 🌶️ Experiment H (optional): transfer learning and multi-window input
# **Idea 1: pretrained encoder.** `FlexibleUNet` can use an EfficientNet backbone with weights pretrained on ImageNet (natural images!). Is a
# pretrained encoder useful for CT? It usually helps most with **little data** (combine it with experiment B: train on 10%).
# It needs the internet (the weights are downloaded) and an input of 3 channels, which brings us to idea 2.
#
# **Idea 2: three windows as three channels.** A CT image is a 12-bit signal; we squeezed it into one window `[-1000, 400]`. Give the network three
# views at once: lung window, mediastinal window and bone window, as 3 channels. Add `windows=[...]` to the config (the harness does the rest).
#
# > ⚠️ **Pitfall:** `pretrained=True` makes the constructor *download weights*. The harness writes `pretrained=False` into `config.json` (the trained weights are in `model.pt`):
# > otherwise the web app would try to download ImageNet weights at start-up. Also: the web app of day 5 must apply the same windows (`config.json -> preprocessing -> windows`);
# > if it is a single-channel app, this model is *not* compatible: check with your teachers.
#
# **Idea 3: MONAI Model Zoo.** Pretrained bundles for segmentation exist (https://monai.io/model-zoo.html); mostly 3D, so look at what fits before investing time.

# %%
WINDOWS_3CH = [(-1350, 150), (-160, 240), (-500, 1300)]   # lung, mediastinal, bone: (min HU, max HU)

def try_transfer_learning(frac=1.0):
    """Starter code: FlexibleUNet + EfficientNet-b0 (ImageNet weights), 3 CT windows as input channels."""
    kwargs = dict(spatial_dims=2, in_channels=3, out_channels=2, backbone="efficientnet-b0", pretrained=True)
    try:
        build_model("FlexibleUNet", kwargs)   # downloads the weights; fails offline
    except Exception as err:                  # no internet, missing package, unsupported option...
        print(f"transfer learning skipped ({type(err).__name__}: {str(err)[:100]})")
        return None
    cfg = {**BASE_CONFIG, "name": f"flex-effb0-3win-{frac}", "arch": "FlexibleUNet", "arch_kwargs": kwargs,
           "windows": WINDOWS_3CH, "train_fraction": frac, "img_size": max(BASE_CONFIG["img_size"], 64)}   # input must be divisible by 32
    return run_experiment(cfg)


if FAST:
    print("skipped in FAST mode (needs the internet)")
else:
    r_flex = try_transfer_learning(frac=0.1)   # compare with frac-0.1 of experiment B: does pretraining help with little data?

# %% [markdown]
# ### 🌶️ Going further (optional)
# * Same run with `windows=None` and `in_channels=1`: is it the pretraining or the extra windows that helps?
# * Compare with `frac-0.05` and `frac-0.1` from B: the *benefit of pretraining shrinks as data grows*? Draw both curves on the same axes.
# * Freeze the encoder for the first epochs (look at `model.encoder`, `requires_grad_`).

# %% [markdown] tags=["solution-only"]
# **Discussion.** *Expected (qualitative, not verified here: needs internet and the pretrained weights).* With 10% of the patients the pretrained
# encoder typically converges faster and ends slightly higher than the same architecture from scratch; with 100% the difference vanishes within the noise.
# ImageNet features are not CT features, but low-level edge/texture filters transfer; on a task as easy as the lungs expect a small benefit.
# The multi-window input is cheap and sometimes helps for structures whose contrast lies outside the lung window (e.g. mediastinum, bone).

# %% [markdown]
# ## 🌶️ Experiment I (optional): Optuna, cross-validation, ensembles
# **I.1 Hyper-parameter search with Optuna** (starter code, not executed): a search is the "automatic" version of experiments C-E. It
# explores several knobs *together* (which breaks the one-change-at-a-time rule on purpose, for efficiency), so you must still confirm the winner with
# seeds. The objective function simply calls `run_experiment(..., record=False)`.

# %%
def optuna_objective(trial):
    """Starter code: one trial = one short run with sampled hyper-parameters."""
    cfg = {**BASE_CONFIG, "name": f"optuna-{trial.number}",
           "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
           "loss": trial.suggest_categorical("loss", ["dice", "dice_ce", "dice_focal"]),
           "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
           "epochs": max(2, BASE_CONFIG["epochs"] // 3)}          # short runs: relative ranking is what matters
    return run_experiment(cfg, record=False, verbose=False)["best_val_dice"]


if FAST:
    print("skipped in FAST mode")
else:
    try:
        import optuna
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(optuna_objective, n_trials=8)
        print(study.best_params, study.best_value)
    except ImportError:
        print("pip install optuna to try this (https://optuna.readthedocs.io)")

# %% [markdown]
# **I.2 Cross-validation with the 10 official LUNA16 subsets.** One validation set gives one noisy number. With `fold=k` the harness validates on
# subset `k` and trains on the other development subsets (0-8; subset 9 stays the closed test set). The mean and spread over folds is a better estimate
# than the seed trick. It costs one run per fold (use 3 folds).

# %%
if FAST:
    print("skipped in FAST mode")
else:
    cv_rows = []
    for k in (0, 3, 6):
        r = run_experiment({**BASE_CONFIG, "name": f"cv-fold{k}", "fold": k}, verbose=False)
        cv_rows.append(r["best_val_dice"])
    print(f"3-fold CV: mean {np.mean(cv_rows):.4f}  std {np.std(cv_rows, ddof=1):.4f}   (per fold {np.round(cv_rows, 4)})")

# %% [markdown]
# **I.3 Ensemble of models** (executed: it only needs two trained models). Averaging the softmax of several models reduces the variance of their errors.
# Do the two baseline seeds together beat each of them alone, at native resolution?

# %%
members = [load_model(WORK_DIR / "models" / n, device)[0] for n in ("baseline-s42", "baseline-s43")]
ev_ens = evaluate_cases(SoftmaxEnsemble(members), val_files, BASE_CONFIG["img_size"])
ev_one = {n: evaluate_model_dir(WORK_DIR / "models" / n, val_files) for n in ("baseline-s42", "baseline-s43")}
print({**{n: round(e["dice_lung"].mean(), 4) for n, e in ev_one.items()}, "ensemble": round(ev_ens["dice_lung"].mean(), 4)})
print("HD95 mm:", {**{n: round(e["hd95_mm"].mean(), 3) for n, e in ev_one.items()}, "ensemble": round(ev_ens["hd95_mm"].mean(), 3)})

# %% [markdown]
# > 🧠 **Conclusion:** does the ensemble beat its members *beyond the noise*? What is the price (inference time and file size x2), and can the web app of
# > day 5 load such a model through the contract `arch` + `arch_kwargs`? (No: an ensemble is not a single class of `monai.networks.nets`.)

# %% [markdown] tags=["solution-only"]
# **Discussion.** *Expected (qualitative).* A 2-model ensemble gains a few thousandths of Dice at best, mostly on the hard cases and on HD95; for the mean
# on an easy task it is within noise. It doubles inference cost, so for a small CPU web app it is rarely worth it; it is an excellent way to get a
# *free uncertainty estimate*: where the members disagree, the segmentation is unreliable (a nice extension for the web app).
# Optuna: the winner is usually near lr 1e-3 with a Dice-based loss - i.e. the defaults were already good: a successful negative result.
# Cross-validation: std over folds is typically larger than the seed noise because it includes the difference between subsets (different scanners and sites).

# %% [markdown]
# ## 🌶️ Experiment J (optional): MLflow instead of the CSV
# Our harness logs to a CSV and TensorBoard. Real teams use an **experiment tracker** (MLflow, Weights & Biases): runs, parameters,
# metrics, artefacts and a web UI to compare them. Starter code for MLflow (guarded: it does nothing if MLflow is not installed). It re-logs
# the runs you already did in this session (`RESULTS`). For Weights & Biases the idea is the same: `wandb.init(config=cfg)`, `wandb.log({...})`
# (needs an account).

# %%
def log_to_mlflow(result, experiment="lung-segmentation"):
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{WORK_DIR / 'mlflow.db'}")   # recent MLflow versions want a database backend, not plain folders
    mlflow.set_experiment(experiment)
    cfg = result["config"]
    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.log_params({k: str(v) for k, v in cfg.items()})
        for epoch, dice in enumerate(result["history"]["val_dice"], 1):
            mlflow.log_metric("val_dice", dice, step=epoch)
        mlflow.log_metrics({"best_val_dice": result["best_val_dice"], "train_time_s": result["train_time_s"],
                            "n_params": result["n_params"]})


if FAST:
    print("skipped in FAST mode")
else:
    try:
        for r in RESULTS.values():
            log_to_mlflow(r)
        print(f"logged {len(RESULTS)} runs. Open the UI with:  mlflow ui --backend-store-uri sqlite:///{WORK_DIR / 'mlflow.db'}")
    except ImportError:
        print("pip install mlflow to try this (https://mlflow.org/docs/latest/)")

# %% [markdown] tags=["solution-only"]
# **Teacher note.** MLflow is optional and not needed for the course. On Kaggle the UI cannot be opened directly (no inbound ports): download `mlflow.db`
# and run `mlflow ui --backend-store-uri sqlite:///mlflow.db` locally. Everything the students need is already in `experiments.csv`.

# %% [markdown]
# ## 5. Comparing fairly
# Now put all your runs in one table. Rules for reading it:
# * compare **full-data, binary** runs with each other (the learning-curve points, and the 4-class run, use other settings);
# * differences **smaller than the seed noise** are not differences;
# * Dice is not the only column: the web app runs on a small CPU machine, so look at **parameters and time** too;
# * you ran many experiments: the best number is probably a bit **lucky** (winner's curse).

# %%
table = load_results()
table = table[(table["train_fraction"] == 1.0) & (table["n_classes"] == 2) & table["fold"].isna() & ~table.index.str.startswith("_")]
table = table.sort_values("best_val_dice", ascending=False)
display(table[["arch", "loss", "lr", "img_size", "augment", "n_params", "train_time_s", "best_val_dice"]].round(4))

fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(table)), 4))
ax.bar(range(len(table)), table["best_val_dice"], color="tab:blue")
if "baseline-s42" in table.index and NOISE is not None:
    b = table.loc["baseline-s42", "best_val_dice"]
    ax.axhspan(b - NOISE, b + NOISE, color="tab:orange", alpha=0.35, label=f"baseline +/- seed noise ({NOISE:.4f})")
    ax.legend()
ax.set_xticks(range(len(table)), table.index, rotation=45, ha="right", fontsize=8)
ax.set(ylim=(max(0, table["best_val_dice"].min() - 0.01), min(1.0, table["best_val_dice"].max() + 0.005)), ylabel="best val Dice",
       title="All full-data runs (sorted)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### ✏️ Exercise 5.1 - choose your model
# Pick **one** run (its `name` from the table) as the configuration to ship, and justify in two lines (accuracy *within noise*? speed? size? the app runs on a small CPU).
# A good rule: *among the runs within the noise of the best one, take the cheapest.*
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Filter the table for `best_val_dice >= best - NOISE` and then look at `n_params` or `train_time_s`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `ok = table[table.best_val_dice >= table.best_val_dice.max() - NOISE]` and `ok.n_params.idxmin()`
#
# </details>

# %% [markdown] tags=["student-only"]
# **My choice and justification (2 lines):** _(write it here)_

# %%
#@solution
near_best = table[table["best_val_dice"] >= table["best_val_dice"].max() - max(NOISE, 1e-4)]
CHOSEN = near_best["n_params"].idxmin()
#@student
#| CHOSEN = "____"   # TODO: the name of the run you ship (a name from the table above)
#@end
print(f"chosen run: {CHOSEN}")

# %%
# Check 5.1
assert CHOSEN in table.index, "CHOSEN must be one of the run names in the table above"
print("✅ Exercise 5.1 passed")

# %% [markdown] tags=["solution-only"]
# **Answer (example).** "Within the seed noise several runs tie for the best Dice; we ship the one with the fewest parameters, which also trains and
# runs fastest. The numbers are validation numbers from the same set we selected on, so they are optimistic: the test set (next) will tell." Teachers: check that students
# resist reading a 0.001 gap as a winner, and that they consider CPU latency (notebook 06) before shipping the biggest model.

# %% [markdown]
# > 🧠 **Question:** why is the "best of N" validation Dice optimistic? (Think of N noisy measurements of identical models: the maximum of N
# > noisy numbers has a positive bias.) How could you reduce the bias without a new data set?

# %% [markdown] tags=["solution-only"]
# **Answer.** If all configurations were equally good, the best validation Dice among N would still be above the true value, because we select the noisy maximum
# (winner's curse / selection bias on the validation set). The more experiments, the larger the optimism. Remedies: a fresh, untouched test set (what we do),
# cross-validation (more validation data, less noise), nested cross-validation for honest model selection, and limiting the number of comparisons (hypotheses first!).

# %% [markdown]
# ## 6. The final model: retrain, then test **once**
# We retrain the chosen configuration under your personal name with a **new seed** (a second look at the "lucky seed" question: the number
# we report must not depend on the seed that made it win). Default: train on the **training set only**. Training on train+val would give a
# slightly better model, but then there is no validation set left to confirm that the run is healthy or to choose the epoch: in a real project you do that **after** the model selection is
# closed, with a fixed number of epochs, and you must say so in the model card.
#
# ### ✏️ Exercise 6.1 - name your model
# Choose a distinctive `MY_NAME` (letters, digits, `-`, e.g. `unet-small-giulia`). The web app lists models by this name; do not use `baseline`!
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Add your name or your pair's name and a hint of what is special (architecture, loss): the whole class will see this name in the app.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `MY_NAME = "unet-small-giulia"`
#
# </details>

# %%
#@solution
MY_NAME = "unet-solution-demo"
STUDENT = "teachers (solution notebook)"
#@student
#| MY_NAME = "____"   # TODO: your personal model name, lowercase letters, digits and '-'
#| STUDENT = "____"   # TODO: your name (or both names of the pair)
#@end

# %%
# Check 6.1
assert re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,}", MY_NAME) and not MY_NAME.startswith("baseline"), "use a personal name like unet-small-giulia"
assert "____" not in STUDENT, "write your name"
print(f"✅ Exercise 6.1 passed: {MY_NAME} by {STUDENT}")

# %%
FINAL_CFG = {**config_of(CHOSEN), "name": MY_NAME, "seed": 2024}      # same configuration, a NEW seed
FINAL_CFG["epochs"] = BASE_CONFIG["epochs"]
final = run_experiment(FINAL_CFG)
print(f"validation Dice of the retrained model: {final['best_val_dice']:.4f}  (selected run {CHOSEN}: {load_results().loc[CHOSEN, 'best_val_dice']:.4f})")

# %% [markdown]
# **Now the test set - once.** `evaluate_model_dir` reads everything from the model folder (the *contract*): that is also the proof that
# your folder is self-contained. Per-case Dice and HD95 at native resolution, then the summary.
#
# > ⚠️ **Pitfall:** if the number is disappointing you may **not** go back and change something and test again: that turns the test set into
# > a second validation set. Report what you got, and discuss it.

# %%
test_cases = evaluate_model_dir(WORK_DIR / "models" / MY_NAME, test_files)
test_dice, test_hd95 = test_cases["dice_lung"].mean(), test_cases["hd95_mm"].mean()
display(test_cases[["dice_lung", "hd95_mm"]].describe().round(4))
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].hist(test_cases["dice_lung"].dropna(), bins=20)
ax[0].set(title="test: per-case Dice", xlabel="Dice")
ax[1].hist(test_cases["hd95_mm"].dropna(), bins=20, color="tab:red")
ax[1].set(title="test: per-case HD95 (mm)", xlabel="mm")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### The class leaderboard
# Paste this one line into the shared class sheet (your teachers will tell you where). It is the *only* test number you report.

# %%
n_params_final = int(load_results().loc[MY_NAME, "n_params"])
leaderboard_line = f"{MY_NAME} | {FINAL_CFG['arch']} | {n_params_final:,} params | val Dice {final['best_val_dice']:.4f} | test Dice {test_dice:.4f} | test HD95 {test_hd95:.2f} mm"
print(leaderboard_line)

# %% [markdown]
# > 🧠 **Question:** the class sheet compares ~15 test Dice values. Why would it be a mistake to **choose the best model of the class for the app by that test Dice**,
# > or worse, to let every group re-train until its test Dice goes up? What would you need to compare the models of the class honestly?

# %% [markdown] tags=["solution-only"]
# **Answer.** The test set is a finite sample: differences between the models in the class are of the order of the noise (a few thousandths).
# Picking the maximum selects the lucky model *and* the models that fit the idiosyncrasies of those test patients; each re-training with a look
# at the test score leaks information from the test set into the model (indirect overfitting), so the leaderboard number stops being an unbiased estimate: the test set
# has become a validation set. The same holds for public leaderboards (cf. Kaggle's public vs private leaderboard "shake-up"). Honest comparison: a fresh
# held-out set nobody has seen (e.g. another hospital: external validation), confidence intervals / paired tests on per-case differences, and
# criteria beyond Dice (speed, size, robustness - notebook 04).
# **Teacher note.** In the real-data class, expect almost all test Dice values in a narrow band (e.g. 0.97-0.99); use the sheet to show that the *ranking inside the band is not meaningful*.

# %% [markdown]
# ## 7. Save and hand over your model (for days 4-5)
# `run_experiment` already wrote your model folder `WORK_DIR/models/<MY_NAME>/` (`model.pt` + `config.json`). We add the test numbers to the
# `config.json` (a model card in miniature), prove that the folder loads with the **course loader**, and zip it.

# %%
model_dir = WORK_DIR / "models" / MY_NAME
cfg_json = json.loads((model_dir / "config.json").read_text())
cfg_json["metrics"].update(test_dice=round(float(test_dice), 4), test_hd95_mm=round(float(test_hd95), 3))
cfg_json["trained_by"] = STUDENT
(model_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))

# the course contract check: rebuild from config.json only and predict a validation case
net, loaded_cfg = load_model(model_dir, device)
size = loaded_cfg["preprocessing"]["img_size"][0]
x = get_transforms(size, windows=loaded_cfg["preprocessing"].get("windows"))(val_files[0])["image"].unsqueeze(0).to(device)
with torch.no_grad():
    out = net(x)
assert out.shape[1] == len(loaded_cfg["labels"]) and out.shape[-2:] == (size, size)
zip_path = shutil.make_archive(str(WORK_DIR / MY_NAME), "zip", model_dir)
print(json.dumps(cfg_json, indent=2))
print(f"\n✅ contract OK. Zip: {zip_path} ({Path(zip_path).stat().st_size / 1e6:.1f} MB)")

# %% [markdown]
# ### Getting your model out of Kaggle / Colab
# * **Kaggle**: the zip is in `/kaggle/working`. After *Save Version -> Save & Run All (Commit)* it appears under the notebook's **Output** tab
#   (and it can be downloaded or added as input of another notebook). Without a commit, use the file browser on the right (*Output*) and *Download*.
#   Kaggle sessions delete `/kaggle/working` when they end unless you commit: **do not close the tab before downloading**.
# * **Colab**: `from google.colab import files; files.download("<name>.zip")` (or mount Google Drive). The runtime is erased when it disconnects.
# * **Locally / VS Code**: it is simply the folder `outputs/models/<name>`.
#
# On day 4 (notebook 06) you load this folder in an inference pipeline; on day 5 the app will list all the model folders of the class.
# **Keep the zip, and tell your teachers your model name for the class sheet.**

# %%
# Optional convenience: on Colab, offer the download right away
if not FAST:
    try:
        from google.colab import files
        files.download(str(zip_path))
    except ImportError:
        pass

# %% [markdown]
# ## 8. Recap
# * An experiment = **hypothesis -> one change -> fixed seed and validation set -> logged run -> comparison beyond the noise -> conclusion**.
# * The harness (`BASE_CONFIG` + `run_experiment`) made every run cheap, reproducible and comparable; the `experiments.csv` table was the deliverable.
# * **Seed noise** calibrates decisions: a +0.002 Dice difference is nothing.
# * For an easy task like lung segmentation, **most variants saturate around 0.97-0.99 Dice**. That *is* the lesson: the differences show up in
#   *small-data regimes* (B), at the **tails and boundaries** (F, HD95, worst cases), and in **harder tasks** (G, the trachea), not in the mean.
# * **Data efficiency** (B) answers a practical question that hospitals ask: how many annotations do we need?
# * Select with the validation set, report **once** on the test set; many looks at the same set make it optimistic (winner's curse, leaderboard overfitting).
# * Choose for deployment by **accuracy within noise, then speed and size**.
#
# **What is next.** Notebook 06: inference on raw files, training-serving skew, exporting and the model card. Day 5: the web app.
#
# **Where to look next**
# * MONAI networks: https://docs.monai.io/en/stable/networks.html - losses: https://docs.monai.io/en/stable/losses.html - metrics: https://docs.monai.io/en/stable/metrics.html
# * MONAI tutorials (experiment management, `spleen_segmentation_3d`, ...): https://github.com/Project-MONAI/tutorials
# * nnU-Net: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation": https://arxiv.org/abs/1809.10486
# * Loss odyssey in medical image segmentation: https://doi.org/10.1016/j.media.2021.102035
# * Metrics Reloaded (choosing metrics): https://arxiv.org/abs/2206.01653
# * Attention U-Net: https://arxiv.org/abs/1804.03999
# * TensorBoard: https://www.tensorflow.org/tensorboard - MLflow: https://mlflow.org/docs/latest/ - Optuna: https://optuna.readthedocs.io
# * MONAI Model Zoo: https://monai.io/model-zoo.html
