# Authoring guide (for teachers and contributors)

Every notebook is written **once** in `src/NN_name.py` (Python "percent" format) and
built into two notebooks:

| Output | Audience |
|---|---|
| `notebooks/NN_name.ipynb` | students: exercises with blanks, hints hidden in collapsible blocks |
| `solutions/NN_name_solution.ipynb` | teachers: complete walk-through, answers to the questions |

```bash
python tools/build_notebooks.py                       # rebuild all
python data_prep/make_synthetic_dataset.py --out data/synthetic --n 60
HI_DATA_DIR=$PWD/data/synthetic HI_FAST=1 python tools/run_notebook.py solutions/*.ipynb
```

Never edit the `.ipynb` files by hand: edit `src/*.py` and rebuild.

## Source syntax

```python
# %% [markdown]
# ## A markdown cell
# Every line starts with "# ".

# %%
x = 1  # a code cell shared by students and teachers

# %%
#@solution
loader = DataLoader(train_ds, batch_size=16, shuffle=True)
#@student
#| loader = ____  # TODO: build the training DataLoader (batch_size=16, shuffle!)
#@end

# %% [markdown] tags=["solution-only"]
# **Answer.** Text that only teachers see.

# %% tags=["student-only"]
# a cell that only students see
```

## Pedagogical conventions (keep them identical in every notebook)

**Language**: English. Short sentences. Explain *why* before *how*.

**Header cell** of each notebook: title, "Day N - part M", a 3-5 line summary, a
*Learning objectives* list, an estimated time, and a legend of the icons below.

**Icons / block types** (always these, so students learn to scan for them):

| Marker | Meaning |
|---|---|
| `### ✏️ Exercise N.M - title` | **Mandatory** task. Has blanks (`____`) to fill. Followed by a check cell. |
| `### 🔍 Explore` | Small "try this / change that and re-run" experiments (2-5 bullets). No blanks. For everybody, but skippable. |
| `### 🚀 Challenge (additional)` | Additional task for students who finish early. Solution is provided in the solution notebook. |
| `### 🌶️ Going further (optional)` | Open-ended, optional. Only pointers / starter code, may have no full solution. |
| `> 🧠 **Question:**` | A concept question to answer in words (answer hidden in `<details>` in the student notebook *only if* it is not an assessment; full answer in a solution-only cell). |
| `> ⚠️ **Pitfall:**` | Classic mistake. |
| `> 📚 **Read more:**` | 1-3 links for further study (MONAI tutorial/doc, paper). |

**Hints** are hidden, progressive and in markdown, right after the exercise text:

```markdown
<details>
<summary>💡 Hint 1</summary>

Look at the `keys` argument: both the image and the label must be loaded.

</details>

<details>
<summary>💡 Hint 2 (almost the solution)</summary>

`LoadImaged(keys=["image", "label"])`

</details>
```

Leave an empty line after `<summary>` and before `</details>` (needed for the markdown
inside to render on Kaggle/Colab/VS Code). Give 1-3 hints per exercise: the first one
conceptual, the last one nearly the answer. The actual solution is NOT in the student
notebook.

**Check cells**: after each exercise put a tiny cell with `assert`s and a friendly
message (`print("✅ Exercise 2.1 passed")`) so students get immediate feedback and
weaker students know when they are stuck. Asserts must test behaviour (shape, dtype,
value range), not the exact implementation.

**Blanks**: use `____` (four underscores) for each thing to fill, plus a `# TODO:`
comment. Give scaffolding generously: the mandatory path must be doable by the
weakest student with the hints; strong students are kept busy by 🚀 and 🌶️.

**Exercise density**: roughly one ✏️ every 10-15 minutes of class time; one 🔍 after
each major concept; 1-2 🚀 and 1 🌶️ per notebook.

## Technical conventions

**Self-contained notebooks**: they run on Kaggle (primary), Colab and locally. No
imports from this repo. The first code cells are always:

1. *Setup*: `pip install` of missing packages, guarded by an import check, quiet.
2. *Configuration*: the cell below, identical everywhere (copy verbatim).

```python
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
```

**FAST mode**: every expensive setting must have a FAST alternative so that the solution
notebook executes on a 2-core CPU in < 3 minutes on the synthetic dataset, e.g.
`IMG_SIZE = 64 if FAST else 256`, `MAX_EPOCHS = 2 if FAST else 30`,
`files = files[:24] if FAST else files`. Asserts on *quality* (e.g. Dice > 0.9) must be
skipped in FAST mode; asserts on shapes/types must hold in both.

**Device**: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`,
never hard-code `cuda:0`.

**Reproducibility**: `monai.utils.set_determinism(seed=42)` near the top.

**DataLoader workers**: `NUM_WORKERS = 0 if FAST else 2`.

**Libraries**: PyTorch + MONAI (current stable API: no `AddChanneld`/`AsChannelFirstd`;
use `EnsureChannelFirstd`, `EnsureTyped`; `torch.load(..., weights_only=True)` for
state dicts), nibabel, pydicom, SimpleITK, matplotlib, pandas, scikit-learn. Avoid
other dependencies unless a section is explicitly optional.

## Dataset layout and conventions (v1) - `luna16-lung-slices`

```
images/case_0001.nii.gz   2D NIfTI, int16, Hounsfield units (HU). Typically 512x512.
labels/case_0001.nii.gz   2D NIfTI, uint8: 0 background, 1 right lung, 2 left lung, 3 trachea/main airways
metadata.csv              one row per case (= one patient = one slice)
demo/volume/<case>_image.nii.gz, <case>_label.nii.gz   one full 3D scan (for day 1)
demo/dicom/<case>/IM_0001.dcm ...                      the same scan (or a slab of it) as a DICOM series
demo/mhd/<case>.mhd + .raw                             the same in MetaImage (LUNA16 native format)
volumes_lowres/ (optional)  images/, labels/, metadata.csv: low-resolution 3D volumes for the 3D side track
```

* **Orientation**: arrays are stored in **LPS+** index order (DICOM/ITK convention, the
  native LUNA16 one): axis 0 = x, growing towards the patient's **L**eft; axis 1 = y,
  growing towards **P**osterior. Therefore **`plt.imshow(img.T, cmap="gray")` shows the
  standard radiological view** (anterior on top, patient's right on the left of the
  screen, spine at the bottom). The right lung (label 1) has low x. The NIfTI affine
  encodes this correctly (negative x/y diagonal in RAS world coordinates) together
  with the pixel spacing (mm) and slice thickness.
* **Intensities**: HU. Air is about -1000, lung parenchyma -900..-500, soft tissue
  0..80, bone > 300. Some scans contain a padding value outside the circular field of
  view (-2000, -3024...): always clip before normalising.
* **Binary task (mandatory path)**: lungs = `label in {1, 2}`; trachea counts as
  background. Multi-class (right/left/trachea) is an additional task.
* **`metadata.csv` columns**: `case_id, seriesuid, subset` (0-9: the official LUNA16
  10-fold subsets), `slice_index, n_slices, z_rel` (0 = most inferior slice of the scan,
  1 = most superior), `spacing_x, spacing_y, slice_thickness` (mm), `size_x, size_y`,
  `right_lung_px, left_lung_px, trachea_px, lung_area_mm2`, `has_nodule` (1 if the
  patient has at least one annotated nodule in LUNA16 `annotations.csv`).
* **Splits used in the course**: `subset` 0-6 -> train, 7-8 -> validation, 9 -> test
  (held out until day 3). One slice per patient, so a patient can never leak across
  splits - but students must still understand why splitting by patient matters.
* **Label quality caveat**: LUNA16 lung masks were produced by an automatic algorithm,
  they are "silver standard" labels. This is a teaching point, not a bug.

## Model hand-over contract (training notebooks -> web app)

A trained model is saved as a folder (zip it to move it around):

```
model/
  model.pt          state_dict (torch.save(model.state_dict(), ...))
  config.json       everything needed to rebuild the network and the preprocessing
```

```json
{
  "name": "unet-baseline",
  "arch": "UNet",
  "arch_kwargs": {"spatial_dims": 2, "in_channels": 1, "out_channels": 2,
                  "channels": [16, 32, 64, 128, 256], "strides": [2, 2, 2, 2], "num_res_units": 2},
  "preprocessing": {"hu_min": -1000, "hu_max": 400, "img_size": [256, 256]},
  "labels": {"0": "background", "1": "lung"},
  "metrics": {"val_dice": 0.97},
  "trained_by": "student name", "date": "2026-..-.."
}
```

`arch` is the name of a class in `monai.networks.nets`. The web app (and notebook 07)
rebuild the model with `getattr(monai.networks.nets, arch)(**arch_kwargs)`.

## Canonical code (copy verbatim when a notebook needs the "given" pipeline)

`data_prep/train_baseline.py` is the condensed reference implementation of the whole
training pipeline. Notebooks that need to *recap* earlier steps (04, 05, 06, optional)
must reuse its `lungs_only`, `get_transforms`, `make_datalists` verbatim, in a single
"recap" cell, so that every notebook and the app agree on preprocessing:

* HU window `[-1000, 400]` -> `[0, 1]`, clip; resize to `IMG_SIZE x IMG_SIZE`
  (`256`, or `64` in FAST mode) with `bilinear` for the image and `nearest` for the label.
* Binary lungs: labels 1 and 2 -> 1; network with `out_channels=2` + softmax/argmax.
* Splits from `metadata.csv`: `subset` 0-6 train, 7-8 val, 9 test.

Finding and loading a trained model (verbatim in notebooks 04-06 and optional ones):

```python
import json, monai, torch

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
```

Notebook 03 saves the student's model to `WORK_DIR / "models" / "<name>"`. The course
dataset ships a teachers' model in `DATA_DIR / "pretrained" / "unet-baseline"` (made with
`data_prep/train_baseline.py`) as a safety net: notebooks 04-06 use the student's model
if `find_model_dirs()` finds one, otherwise the teachers' one. The preprocessing
`img_size` to use at inference is always read from the model's `config.json`.

## Optional 3D side-track data: `volumes_lowres/`

```
volumes_lowres/images/case_0001.nii.gz   3D int16 HU, LPS+, resampled to ~2.5 mm isotropic (about 140x140x120)
volumes_lowres/labels/case_0001.nii.gz   3D uint8, same label values as the 2D labels
volumes_lowres/metadata.csv              case_id, seriesuid, subset, size_x, size_y, size_z, spacing_x, spacing_y, spacing_z
```

## Testing in this repo's dev environment

Dev box: 2 CPU cores, no GPU, MONAI 1.6.1, torch 2.5.1 (CPU), Python 3.13. A synthetic
dataset (60 cases, with `pretrained/unet-baseline` trained at 64x64) lives in
`/home/claude/data/synthetic`. Run tests with
`HI_DATA_DIR=/home/claude/data/synthetic HI_FAST=1 HI_WORK_DIR=/tmp/hi_<yourname> OMP_NUM_THREADS=1 python tools/run_notebook.py solutions/<nb>_solution.ipynb`.
Other people are running tests on the same machine at the same time: keep FAST runs tiny.
