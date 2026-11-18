# Health Informatics - Advanced Topics in Machine Learning: hands-on

**From a CT scan to a deployed web app in 5 days.**
A 24-hour hands-on for the MSc in Health Informatics: students train a deep-learning
model that segments the lungs on a chest-CT slice (LUNA16, one slice per patient),
evaluate it like a clinical-AI team would, and ship it as a public Streamlit web app.

> This repository is the **teachers' repository**: it contains the solutions.
> Students receive only the output of `python tools/make_student_release.py`
> (see [TEACHER_GUIDE.md](TEACHER_GUIDE.md)).

## The path

| Day | Seminar (45') | Hands-on | Material |
|---|---|---|---|
| 1 | Course intro: imaging modalities, CT, formats, the project | setup, images & formats, dataset, first MONAI transforms | [`01`](notebooks/01_medical_images_and_formats.ipynb), [`02`](notebooks/02_dataset_and_monai_transforms.ipynb), [setup](docs/SETUP.md) |
| 2 | Segmentation: U-Net, losses, metrics | train the baseline U-Net end to end | [`03`](notebooks/03_train_baseline_unet.ipynb) |
| 3 | Evaluation, generalisation, experimental method | test set, error analysis, first experiments | [`04`](notebooks/04_evaluation_and_error_analysis.ipynb), [`05`](notebooks/05_experiments_improving_the_model.ipynb) |
| 4 | From model to product: inference, packaging, MLOps, regulation | experiments & final model, inference pipeline, model card | [`05`](notebooks/05_experiments_improving_the_model.ipynb), [`06`](notebooks/06_inference_and_packaging.ipynb) |
| 5 | Web apps, containers, deployment, responsible use | VS Code, venv, Streamlit app, GitHub, cloud deploy, Docker, show & tell | [`app/`](app/README.md) |
| side tracks | - | 3D segmentation; classification, embeddings, fusion | [`O1`](notebooks/O1_3d_segmentation.ipynb), [`O2`](notebooks/O2_classification_embeddings_fusion.ipynb) |

Every notebook serves three levels at once:

| Marker | Level | Meaning |
|---|---|---|
| ✏️ Exercise | **mandatory** | fill-in-the-blank task with hidden progressive hints and a check cell |
| 🔍 Explore | mandatory-light | "try this, change that" experiments, no blanks |
| 🚀 Challenge | **additional** | for those who finish early; solved in the solution notebook |
| 🌶️ Going further | **optional** | open-ended, pointers and starter code |

## Where things are

```
SYLLABUS.md          topics (mandatory / additional / optional) with references to study further
TEACHER_GUIDE.md     run-of-show, what to know and where to find it, preparation checklist
docs/SETUP.md        accounts and software students need (day 1 technical setup)
src/                 SINGLE SOURCE of every notebook (edit here, then build)
notebooks/           student notebooks (generated)
solutions/           solution notebooks (generated)
app/                 day 5: student guide, starter app, reference solution (Streamlit, Docker, tests)
data_prep/           build the course dataset from LUNA16, synthetic dataset, teachers' baseline model
tools/               build_notebooks.py, run_notebook.py, make_student_release.py, AUTHORING.md
```

## Quick start for teachers

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU is enough for a dry run
pip install monai nibabel pydicom SimpleITK pandas matplotlib scikit-learn scikit-image scipy \
            nbclient nbformat ipykernel streamlit pytest tensorboard

# 1. dry run of everything on fake data (no download, no GPU, a few minutes)
python data_prep/make_synthetic_dataset.py --out data/synthetic --n 60
python data_prep/train_baseline.py --data data/synthetic --epochs 40 --img-size 64 --workers 0
python tools/build_notebooks.py
HI_DATA_DIR=$PWD/data/synthetic HI_FAST=1 python tools/run_notebook.py solutions/*_solution.ipynb

# 2. the real dataset (see data_prep/README.md), the safety-net model, the student package
python data_prep/prepare_luna16_slices.py --help
python tools/make_student_release.py
```

Then read [TEACHER_GUIDE.md](TEACHER_GUIDE.md).

## Data and licence notes

The course dataset is derived from **LUNA16** (CC BY 4.0), itself derived from
**LIDC-IDRI** (CC BY 3.0). Cite Setio et al., *Medical Image Analysis* 2017
(doi:10.1016/j.media.2017.06.015) and Armato et al., *Medical Physics* 2011
(doi:10.1118/1.3528204). The lung masks are the output of an automatic algorithm:
LUNA16 states they are "not intended to be used as the reference standard for any
segmentation study". We use them knowingly as *silver-standard* labels - and teach it.

Nothing produced in this course is a medical device. Education and research only.
