# Teacher guide

Run-of-show, what you need to know before each day and where to read it, and the
preparation checklist. Student-facing topic list with references: [SYLLABUS.md](SYLLABUS.md).

Levels: **M** mandatory - **A** additional - **O** optional (same as ✏️ / 🚀 / 🌶️ in the notebooks).

## 1. Design in one page

* **One thread**: a lung-segmentation model on one CT slice per patient, from raw files
  to a public web app. Every day ends with an artefact that the next day needs.
* **Three levels in the same material**: the ✏️ path (blanks + hidden progressive hints
  + check cells) is sized so that everybody finishes it; 🚀 keeps fast students busy and
  has solutions; 🌶️ is open-ended. Side-track notebooks O1/O2 exist for a fast class.
* **Safety nets** so that nobody is blocked by yesterday:
  * the dataset ships a teachers' model (`pretrained/unet-baseline`): notebooks 04-06
    and the app use the student's model if they find one, otherwise this one;
  * every notebook has a `FAST` switch (tiny images, 1-2 epochs) that runs on CPU: the
    fallback if the free GPUs are not available;
  * the day-5 starter app runs at every step, even with all TODOs unfilled.
* **Contracts** (do not break them): dataset layout, the canonical pre-processing
  (HU window [-1000, 400] -> [0, 1], resize to 256, bilinear for images / nearest for labels), and the
  model folder (`model.pt` + `config.json`). They are written down in
  [tools/AUTHORING.md](tools/AUTHORING.md).

Daily artefacts: D1 a correct batch out of a DataLoader -> D2 a trained model folder ->
D3 `test_results.csv` + an evaluation summary, first experiments -> D4 the chosen model,
its model card, `app_starter_kit.zip` -> D5 a public URL.

## 2. Schedule

Assumption: 24 h = 5 + 5 + 5 + 5 + 4, one 15-20' break per day included. The opening
of day 1 is the full introduction; days 2-5 open with a 45' seminar. Adjust freely:
the cut list and the extension list are in section 5.

### Day 1 - images, dataset, transforms (300')

| Min | What | Level |
|---|---|---|
| 60 | **Introduction**: course map and final demo of the app (show the goal first!); modalities; CT physics in 5 slides; HU; DICOM, PACS, NIfTI; the dataset and its caveats | - |
| 40 | **Technical setup** ([docs/SETUP.md](docs/SETUP.md)): Kaggle (phone verification!), attach dataset, run the first cells; check GitHub accounts | M |
| 90 | Notebook **01** - Medical images and formats (8 ✏️) | M; 🚀 improved classical baseline |
| 15 | break | |
| 85 | Notebook **02** - Dataset and MONAI transforms (8 ✏️) | M; 🚀 own `MapTransform`, 3-class labels |
| 10 | Wrap-up: what a batch is; homework = finish 02, install VS Code/Python/Git | |

### Day 2 - training (300')

| Min | What | Level |
|---|---|---|
| 45 | **Seminar**: segmentation as per-pixel classification; U-Net (encoder, decoder, skips); Dice vs cross-entropy and imbalance; overlap metrics; "neural networks fail silently" | - |
| 110 | Notebook **03** parts 0-5: split, pipelines, model, loss/metric, sanity checks (9 ✏️) | M |
| 15 | break | |
| 100 | Notebook **03** parts 6-9: training loop, curves, predictions, save the model (5 ✏️). While it trains: 🧠 questions and 📚 links | M |
| 20 | Part 10 explore runs / 🚀 scheduler + AMP, early stopping, TensorBoard, 4-class | A |
| 10 | Wrap-up: everybody downloads `unet-baseline.zip`. Who has no model? -> safety net | |

### Day 3 - evaluation and experiments (300')

| Min | What | Level |
|---|---|---|
| 45 | **Seminar**: what a metric hides; Dice vs boundary metrics; subgroups and shift; silver-standard labels; experimental method (one change, seeds, noise floor) | - |
| 130 | Notebook **04** - Evaluation and error analysis (11 ✏️) | M; 🚀 bootstrap CI, TTA, uncertainty QC |
| 15 | break | |
| 100 | Notebook **05** sections 0-3 + experiment A: harness, baseline, seed noise, augmentation (about 7 ✏️) | M |
| 10 | Wrap-up: 3 students show their worst case. Fast students: side track O1 or O2 | |

### Day 4 - choosing a model and packaging it (300')

| Min | What | Level |
|---|---|---|
| 45 | **Seminar**: notebook vs product; training-serving skew (last year's app as the example); model formats; MLOps loop; model cards; MDR / EU AI Act in 3 slides | - |
| 100 | Notebook **05** experiment B (data efficiency), one of C-E, compare, final model, class leaderboard | M (B, one of C-E, 5-7); A (D-G); O (H-J) |
| 15 | break | |
| 125 | Notebook **06** - Inference and packaging (11 ✏️): contract, skew demo, readers, speed, export, model card, tests, starter kit | M; 🚀 ONNX, 3D volume slice by slice |
| 15 | Wrap-up: everybody has `app_starter_kit.zip` on the laptop; check VS Code + Python + Git work | |

### Day 5 - the web app (240')

| Min | What | Level |
|---|---|---|
| 45 | **Seminar**: client/server and Streamlit's rerun model; environments; containers; deployment options; responsible use (no patient data in a public demo, PHI in DICOM headers) | - |
| 25 | VS Code, venv, install, unzip the kit, `streamlit run app.py` ([app/README.md](app/README.md) §1-2) | M |
| 60 | App steps 1-5: upload, display, cached model, predict + overlay, area + Dice | M |
| 15 | break | |
| 35 | Git, GitHub, deploy on Streamlit Community Cloud | M |
| 25 | Docker: read the Dockerfile, build, run (live demo by the teacher if laptops struggle) | M (read + demo), A (build yourself) |
| - | Steps 6-8: window controls, model selector, download, model comparison | A / O, in parallel |
| 35 | **Show & tell**: each group shows the URL, one failure case, one thing they would do next | M |

## 3. What to know, where to find it

Short on time? Read at least the lines in bold.

### Day 1
* **LPS vs RAS.** Our files are stored in LPS index order (DICOM/ITK convention), so
  `nib.aff2axcodes` prints `('L','P','S')` and `plt.imshow(img.T, cmap="gray")` gives
  the radiological view. Students expect "NIfTI = RAS". -> [NiBabel coordinate systems](https://nipy.org/nibabel/coordinate_systems.html), [Slicer coordinate systems](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html)
* **Axis-order traps**: DICOM `pixel_array` is `[row, col] = [y, x]` and `PixelSpacing`
  is `[row spacing, column spacing]`; SimpleITK arrays are `[z, y, x]` while
  `GetSize/GetSpacing` are `(x, y, z)`; slice order comes from
  `ImagePositionPatient`, not from file names. -> [pydicom](https://pydicom.github.io/pydicom/stable/)
* HU: air -1000, lung about -900..-500, water 0, soft tissue 0..80, bone > 300; scans
  contain padding values (-2000, -3024) outside the field of view: clip before scaling.
* **Label interpolation**: `nearest` for labels, `bilinear` for images; bilinear on a
  label invents classes. (Known detail: PyTorch's `nearest` and `bilinear` sample with a
  sub-pixel offset from each other - about a quarter of a pixel at 256, clearly visible
  as a thin rim in the 64-pixel `FAST` overlays. Harmless here; `nearest-exact` removes
  it but must then be changed everywhere, app included.) Left-right flip is harmless for the
  binary task and a bug for right/left classes.
* MONAI dictionary transforms keep image and label in sync; `CacheDataset` caches the
  deterministic part only. -> [transforms docs](https://docs.monai.io/en/stable/transforms.html), tutorial `modules/transforms_demo_2d.ipynb`
* **The masks are silver standard** (automatic algorithm, LUNA16 says not a reference
  standard). Say it on day 1 and come back to it on day 3.
* Typical mistakes: forgetting `.T` when plotting; `float()` on DICOM tags; `shuffle`
  forgotten; `translate_range` is in pixels while `rotate_range` is in radians.

### Day 2
* **The five lines**: `zero_grad -> forward -> loss -> backward -> step`. Students must
  be able to write them without looking. -> tutorial `3d_segmentation/spleen_segmentation_3d.ipynb`
* Why 2 output channels + softmax (same code works for multi-class) and what
  `DiceLoss(to_onehot_y=True, softmax=True)` does; `DiceCELoss` as default.
  -> [losses docs](https://docs.monai.io/en/stable/losses.html), tutorial `modules/dice_loss_metric_notes.ipynb`
* **`decollate_batch` + `AsDiscrete`**: post-transforms work on single items, not on
  batches. This is where most students get lost. -> tutorial `modules/decollate_batch.ipynb`
* Input size must be divisible by 2^4 = 16 with 4 strides. -> tutorial `modules/UNet_input_size_constraints.ipynb`
* **Sanity checks decide**: if "overfit one batch" (ex. 5.1) fails, do not start the long run.
  -> [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/)
* Typical mistakes: missing `.to(device)`; missing `dice_metric.reset()` (Dice drifts
  across epochs); saving `model` instead of `state_dict()`; GPU not enabled (the
  notebook warns); a custom image size not divisible by 16.
* The best epoch is chosen on validation, so validation Dice is optimistic: the hook for day 3.

### Day 3
* Dice vs IoU (`IoU = D / (2 - D)`), mean vs median vs worst case; HD95 and average
  surface distance in mm need the pixel size **at network resolution**
  (`spacing * original_size / img_size`). -> [metrics docs](https://docs.monai.io/en/stable/metrics.html), [Metrics Reloaded](https://arxiv.org/abs/2206.01653), [metric pitfalls](https://arxiv.org/abs/2302.01790)
* Bland-Altman: bias and limits of agreement (mean difference ± 1.96 SD).
* **Expect saturation**: lung segmentation is easy, most models reach about 0.97-0.99 Dice.
  The lessons are in the tails (apex/base slices, trachea), in boundary metrics, in
  small-data regimes and in the stress tests. Do not promise big jumps on day 3-4.
* The threshold baseline is flattered: the reference masks come from an intensity-based
  algorithm. Good discussion on circular evaluation.
* Post-processing and TTA are tuned on the test set here *for teaching*; say out loud
  that real decisions belong to the validation set.
* MONAI post-transforms modify their input in place; `torch.as_tensor(numpy)` shares
  memory (the notebook warns).
* Experiments: **measure the seed-to-seed noise first** (experiment 0); anything inside
  that band is not an improvement. Measure seconds/epoch on the real GPU before class
  to set the compute budget. -> [nnU-Net](https://arxiv.org/abs/1809.10486), [Loss odyssey](https://doi.org/10.1016/j.media.2021.102035)

### Day 4
* **Training-serving skew** is the centre of the day: last year's app re-implemented
  pre-processing with OpenCV (nearest-neighbour resize, no HU window): show it.
  Notebook 06 measures the Dice cost of several "slightly wrong" pipelines.
  -> [Hidden technical debt](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
* `inference.py` is the single source of truth: the notebook writes it, the app imports
  it, and a test fails if the copies differ (`app/solution/tests`). If you edit it,
  update the embedded copy in `src/06_inference_and_packaging.py` and rebuild.
* `torch.load(..., weights_only=True)` and why pickled models are a security risk.
  -> [serialization notes](https://pytorch.org/docs/stable/notes/serialization.html)
* Model card: intended use / not intended use, data and licence, metrics with n,
  limitations. -> [Model cards](https://arxiv.org/abs/1810.03993), [CLAIM 2024](https://doi.org/10.1148/ryai.240300)
* Models that the app cannot serve: the multi-class one (4 labels), the multi-window
  one (3 input channels), ensembles. Tell students to hand in a 1-channel binary model.
* Regulation at awareness level: software with a medical purpose is a medical device
  (MDR, MDCG 2019-11); medical-device AI falls under the EU AI Act high-risk regime
  through product legislation; FDA/HC/MHRA Good ML Practice. Check current texts
  before the seminar.

### Day 5
* Streamlit reruns the whole script at every interaction: hence `@st.cache_resource`
  for the model. -> [cache_resource](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource)
* venv activation differs per OS (PowerShell execution policy on Windows); selecting
  the interpreter in VS Code is the most common blocker. -> [venv docs](https://docs.python.org/3/library/venv.html)
* Git minimum: `init / add / commit / push`, `.gitignore` before the first commit
  (never `.venv`, never data). GitHub rejects files > 100 MB: the U-Net is about 6.5 MB.
* **Streamlit Community Cloud**: about 2.7 GB RAM / 2 cores, sleeps after 12 h idle,
  pick Python 3.12 in *Advanced settings* at creation (cannot be changed later).
  -> [docs](https://docs.streamlit.io/deploy/streamlit-community-cloud)
* Docker: image vs container, layers, `build`, `run -p 8501:8501`.
  -> [Docker get started](https://docs.docker.com/get-started/)

## 4. Preparation checklist

**Decisions already taken (change them if you disagree)**
* GPU platform: **Kaggle** primary (per-student quota of about 30 h/week, dataset
  attached with one click, 12 h sessions), **Colab** fallback, `FAST` CPU mode as the
  last resort (measured: 63 patients, 128 px, 25 epochs = 33 s on 2 CPU cores, validation
  Dice 0.945). SageMaker Studio Lab is closed to new users; Lightning AI free credits
  are unclear. A full run of the course needs well under 5 GPU-hours per student.
* Deployment: **Streamlit Community Cloud**. Docker is taught locally. Hugging Face
  Spaces (Docker) may no longer be free for new accounts: optional only.
* Notebooks and UI in English.

**Two weeks before**
1. Build the dataset: [data_prep/README.md](data_prep/README.md)
   (`prepare_luna16_slices.py`, about 67 GB of downloads for all 10 subsets; a subset
   of the folds is enough for a rehearsal: one subset = 89 patients, 7 GB, 2 minutes of
   processing). Run `--check`. Decide what to do with failed reference masks (in subset0
   one patient has outside air labelled as lung): `--drop-bad-masks` removes them; keeping
   one is a good classroom example of silver-standard labels (`mask_outside_body` column).
2. Train the safety-net model on a GPU: `python data_prep/train_baseline.py --data <dataset>`
   (writes `<dataset>/pretrained/unet-baseline/`). Note its validation Dice.
3. Publish the dataset on Kaggle (public is simplest; CC BY 4.0, keep the attribution
   file) and keep the zip for Colab/local use.
4. **Run every solution notebook once on Kaggle with the full dataset and a GPU.**
   What has been tested so far: all eight solution notebooks run end to end in `FAST`
   mode on synthetic data and on real LUNA16 slices (subset0, 89 patients); notebooks
   03, 04 and 06 also ran in full mode on those 89 patients **on CPU** (13 min for 03,
   test Dice 0.97; rendered copies in `solutions/rendered/`). Never tested: a GPU
   (so the mixed-precision path), the full 888 patients, Kaggle/Colab themselves, and
   the full-mode branches of 05, O1 and O2. To calibrate after your run: the
   `best_dice > 0.85` assert in notebook 03, the time estimates in 03 and 05, and the
   "expected results" solution cells in 04-06 (written qualitatively).
   Note from the small real run: the classical threshold baseline scored *higher* than
   the U-Net (0.980 vs 0.971 Dice) - the reference masks come from an intensity-based
   algorithm. With 888 patients the U-Net should close the gap; either way it is the
   best discussion of the week.
5. `python tools/make_student_release.py` -> push `dist/student` to a **public**
   repository (needed for Colab links and for Streamlit Cloud forks) and/or publish the
   six notebooks on Kaggle for "Copy & Edit". This repository must stay private.
6. Do one throw-away deploy of `app/solution` on Streamlit Community Cloud with the real
   model: it verifies that the CPU-only PyTorch line in `requirements.txt`
   (`--extra-index-url https://download.pytorch.org/whl/cpu`) is honoured there.
7. Build the Docker image once on your laptop (`docker build` was never run during authoring).
8. Verify the UI labels in [docs/SETUP.md](docs/SETUP.md) and in `app/README.md`
   against the current Kaggle / Colab / Streamlit screens (written from documentation).
9. Tell students to create Kaggle (with phone verification) and GitHub accounts.

**Each morning**: open the day's solution notebook; have the safety-net model and the
dataset zip on a USB stick / shared drive; a shared sheet for the class leaderboard (day 4).

## 5. If time is short / if the class is fast

**Cut list (in this order)**: 03 part 10 explore runs and receptive-field cell -> 04
sections 9-10 as teacher demo -> 05 experiment C-E (keep A, B) -> 06 sections 4-5 as
demo -> day-5 Docker as teacher demo only. Never cut: 03 sanity checks, 04 worst-case
gallery, 06 skew demo, the deploy.

**Extensions**: side-track notebooks O1 (3D; needs `volumes_lowres/` in the dataset:
`prepare_luna16_slices.py --volumes N`) and O2 (classification, embeddings, fusion);
experiments F-J in 05; app step 8; compare with `lungmask` / TotalSegmentator (O1).
Further ideas not developed: nodule-centred slices from `annotations.csv` for a
detection/classification task; MONAI Label for interactive annotation; a FastAPI
service behind the Streamlit front-end (see MONAI tutorials `deployment/fastapi_inference`).

## 6. Maintaining the material

* Edit `src/*.py`, never the `.ipynb`; rebuild with `python tools/build_notebooks.py`;
  syntax and conventions in [tools/AUTHORING.md](tools/AUTHORING.md).
* Smoke test: `HI_DATA_DIR=$PWD/data/synthetic HI_FAST=1 python tools/run_notebook.py solutions/*_solution.ipynb`
  (about 5 minutes on 2 CPU cores) and `cd app/solution && HI_MODEL_DIR=<model folder> python -m pytest -q`.
* Notebooks are committed without outputs. To give teachers rendered solutions, run
  them on Kaggle with the real data and save those copies.
* Versions used while authoring: Python 3.13, PyTorch 2.5.1 (CPU), MONAI 1.6.1. The app
  pins `torch==2.5.1`, `monai==1.6.1`; Kaggle/Colab install the current MONAI on top of
  their own PyTorch - re-run step 4 of the checklist close to the course dates.
