# Syllabus

**Project**: lung segmentation on chest-CT slices, from raw files to a deployed web app.
**Format**: 5 days, 24 hours. Each day opens with a 45-minute seminar, the rest is hands-on.

Levels: **M** = mandatory (everybody gets there, with hints) - **A** = additional (if you
finish early; solutions exist) - **O** = optional (open-ended, for the curious).
"Where" points to the notebook section; "Study further" lists the best one or two sources.

## Learning outcomes

At the end you can:

1. open CT images in DICOM, MetaImage and NIfTI, and reason about Hounsfield units, spacing and orientation;
2. build a MONAI pre-processing and augmentation pipeline and explain every step;
3. train a U-Net with PyTorch/MONAI, with the sanity checks that catch bugs early;
4. evaluate a segmentation model per case, in physical units, by subgroup and under perturbation, and state its limitations;
5. design and track fair experiments, and choose a model with arguments;
6. package inference so that serving equals training, write a model card;
7. build, version, containerise and deploy a small web app around the model.

---

## Day 1 - Medical images, the dataset, MONAI transforms

Seminar: course overview; imaging modalities (X-ray, CT, MRI, PET, US), what a voxel
measures; DICOM and the hospital data flow (PACS); the project and the dataset.

| Topic | Level | Where | Study further |
|---|---|---|---|
| Technical setup: Kaggle GPU notebook, GitHub account, VS Code + Python | M | [docs/SETUP.md](docs/SETUP.md) | [VS Code Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial) |
| CT and Hounsfield units, windowing | M | 01 §2, §5 | Radiopaedia: [Hounsfield unit](https://radiopaedia.org/articles/hounsfield-unit), [CT windowing](https://radiopaedia.org/articles/windowing-ct) |
| DICOM: tags, stored values -> HU, series -> volume, PHI in headers | M | 01 §3-4 | [pydicom docs](https://pydicom.github.io/pydicom/stable/), [DICOM standard browser](https://dicom.innolitics.com/ciods) |
| MetaImage (.mhd) with SimpleITK; the `[z, y, x]` trap | M | 01 §6 | [SimpleITK](https://simpleitk.org/) |
| NIfTI, the affine, voxel vs world coordinates, LPS vs RAS | M | 01 §7 | [NiBabel: coordinate systems and affines](https://nipy.org/nibabel/coordinate_systems.html), [3D Slicer: coordinate systems](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html) |
| Three orthogonal views, anisotropic spacing | M | 01 §8 | - |
| Classical baseline: thresholding + Dice by hand | M | 01 §9 | - |
| Connected components / morphology to improve the baseline | A | 01 🚀 | [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html) |
| De-identifying DICOM, DICOM -> NIfTI conversion, 3D Slicer / ITK-SNAP | O | 01 🌶️ | [ITK-SNAP](https://www.itksnap.org/), [dcm2niix](https://github.com/rordenlab/dcm2niix) |
| The dataset: LUNA16, one slice per patient, silver-standard masks, licence | M | 02 §2 | [LUNA16 data page](https://luna16.grand-challenge.org/Data/), Setio et al. 2017 ([doi](https://doi.org/10.1016/j.media.2017.06.015)) |
| Exploratory analysis: spacing, thickness, slice level, lung area | M | 02 §3 | - |
| MONAI basics: array vs dictionary transforms, `MetaTensor`, `Compose` | M | 02 §4 | [MONAI transforms docs](https://docs.monai.io/en/stable/transforms.html), tutorial [`modules/load_medical_images.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/load_medical_images.ipynb) |
| Pre-processing: channel first, HU window scaling, resize vs resample, label interpolation | M | 02 §5 | tutorial [`modules/transforms_demo_2d.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/transforms_demo_2d.ipynb) |
| Augmentation: which transforms make anatomical sense | M | 02 §6 | tutorial [`modules/transform_visualization.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/transform_visualization.ipynb) |
| `Dataset`, `CacheDataset`, `DataLoader` | M | 02 §7 | [MONAI data docs](https://docs.monai.io/en/stable/data.html) |
| Writing your own `MapTransform`; `CropForegroundd`; 3-class labels | A | 02 🚀 | tutorial [`modules/transforms_metatensor.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/transforms_metatensor.ipynb) |
| Inverse transforms, elastic deformation, 3D transforms | O | 02 🌶️ | tutorial [`modules/3d_image_transforms.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/3d_image_transforms.ipynb) |

## Day 2 - Training a segmentation model

Seminar: the segmentation task; encoder-decoder networks and U-Net; Dice and
cross-entropy losses; overlap metrics; how training can silently fail.

| Topic | Level | Where | Study further |
|---|---|---|---|
| Train / validation / test split by patient | M | 03 §1 | - |
| U-Net architecture; MONAI `UNet` arguments; input-size constraint | M | 03 §3 | Ronneberger et al. 2015 ([arXiv](https://arxiv.org/abs/1505.04597)), tutorial [`modules/UNet_input_size_constraints.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/UNet_input_size_constraints.ipynb) |
| Class imbalance; soft Dice loss by hand; Dice + CE | M | 03 §4 | Milletari et al. 2016 ([arXiv](https://arxiv.org/abs/1606.04797)), [MONAI losses](https://docs.monai.io/en/stable/losses.html), tutorial [`modules/dice_loss_metric_notes.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/dice_loss_metric_notes.ipynb) |
| `DiceMetric`, `AsDiscrete`, `decollate_batch` | M | 03 §4 | tutorial [`modules/decollate_batch.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/decollate_batch.ipynb) |
| Sanity checks: loss at init, overfit one batch | M | 03 §5 | Karpathy, [A recipe for training neural networks](https://karpathy.github.io/2019/04/25/recipe/) |
| The training and validation loops, best checkpoint | M | 03 §6 | tutorials [`3d_segmentation/spleen_segmentation_3d.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb), [`2d_segmentation/torch/unet_training_dict.py`](https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_dict.py) |
| Reading learning curves; looking at predictions and worst cases | M | 03 §7-8 | - |
| Saving a model others can load (weights + config) | M | 03 §9 | [PyTorch serialization notes](https://pytorch.org/docs/stable/notes/serialization.html) |
| LR scheduler, mixed precision, early stopping, TensorBoard | A | 03 🚀 | [TensorBoard with PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) |
| Multi-class (right / left lung / trachea) and the flip pitfall | A | 03 🚀, 05 G | - |
| Ignite `SupervisedTrainer`, sliding-window inference at full resolution | O | 03 🌶️ | tutorial [`3d_segmentation/unet_segmentation_3d_ignite.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unet_segmentation_3d_ignite.ipynb) |

## Day 3 - Evaluating, then improving

Seminar: why one average Dice is not an evaluation; overlap vs boundary metrics;
generalisation, subgroups and distribution shift; label quality; experimental method.

| Topic | Level | Where | Study further |
|---|---|---|---|
| Test set, per-case Dice and IoU, distributions instead of means | M | 04 §1-2 | Maier-Hein et al. 2024, Metrics Reloaded ([arXiv](https://arxiv.org/abs/2206.01653)) |
| Boundary metrics in mm: Hausdorff 95, average surface distance | M | 04 §3 | [MONAI metrics](https://docs.monai.io/en/stable/metrics.html), Reinke et al., metric pitfalls ([arXiv](https://arxiv.org/abs/2302.01790)) |
| A clinical quantity: lung area in cm², Bland-Altman agreement | M | 04 §4 | Bland & Altman, *Lancet* 1986 |
| Error analysis: worst cases, failure modes, label errors | M | 04 §6 | Reinke et al., a picture story ([arXiv](https://arxiv.org/abs/2104.05642)) |
| Stratified performance (slice level, thickness, nodules) | M | 04 §7 | Hofmanninger et al. 2020, lung segmentation is a data-diversity problem ([doi](https://doi.org/10.1186/s41747-020-00173-2)) |
| Post-processing: largest components, fill holes | M | 04 §8 | tutorial [`modules/postprocessing_transforms.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/postprocessing_transforms.ipynb) |
| Robustness stress tests (window, noise, flips, rotation) | M | 04 §9 | - |
| Deep learning vs the classical threshold baseline | M | 04 §10 | - |
| Bootstrap confidence intervals; test-time augmentation; uncertainty as quality control | A | 04 🚀 | tutorial [`modules/inverse_transforms_and_test_time_augmentations.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/inverse_transforms_and_test_time_augmentations.ipynb) |
| Calibration; evaluation at full resolution | O | 04 🌶️ | - |
| Experimental method: one change, seeds, noise floor, compute budget | M | 05 §1-3 | Isensee et al. 2021, nnU-Net ([arXiv](https://arxiv.org/abs/1809.10486)) |
| Experiment A: augmentation; B: how many patients do we need? | M | 05 §4 A-B | - |
| Experiment C: loss functions | M (one of C-E) | 05 §4 C | Ma et al. 2021, Loss odyssey ([doi](https://doi.org/10.1016/j.media.2021.102035)) |
| Experiments D: architecture and capacity; E: optimisation | A | 05 §4 D-E | [MONAI networks](https://docs.monai.io/en/stable/networks.html), Oktay et al. 2018, Attention U-Net ([arXiv](https://arxiv.org/abs/1804.03999)) |

## Day 4 - From a model to something others can use

Seminar: the gap between a notebook and a product; training-serving skew; packaging
and model formats; MLOps in one picture; model cards; what regulation asks (MDR, EU AI Act).

| Topic | Level | Where | Study further |
|---|---|---|---|
| Comparing runs fairly, choosing a model, one final test evaluation | M | 05 §5-7 | - |
| Experiments F: resolution; G: multi-class | A | 05 §4 F-G | - |
| Transfer learning, hyper-parameter search, cross-validation, ensembles, MLflow | O | 05 §4 H-J | [MLflow getting started](https://mlflow.org/docs/latest/ml/getting-started/), tutorial [`experiment_management/spleen_segmentation_mlflow.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/spleen_segmentation_mlflow.ipynb) |
| The inference contract: raw file in, mask and measurements out, on CPU | M | 06 §1 | - |
| Training-serving skew: measure what a "slightly wrong" pre-processing costs | M | 06 §2 | Sculley et al. 2015, Hidden technical debt in ML systems ([NeurIPS](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)) |
| Readers for NIfTI and DICOM | M | 06 §3 | [pydicom docs](https://pydicom.github.io/pydicom/stable/) |
| Latency and size on CPU | M | 06 §4 | - |
| Model formats: state dict (and why not pickle), TorchScript | M | 06 §5 | [TorchScript](https://docs.pytorch.org/docs/stable/jit.html) |
| Model card | M | 06 §6 | Mitchell et al. 2019, Model cards ([arXiv](https://arxiv.org/abs/1810.03993)), CLAIM 2024 ([doi](https://doi.org/10.1148/ryai.240300)) |
| Tests for ML code; the hand-over kit | M | 06 §7-8 | - |
| ONNX export, slice-by-slice inference on a 3D volume, quantisation | A | 06 🚀 | [ONNX export tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/intro_onnx.html), tutorial [`modules/2d_inference_3d_volume.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/2d_inference_3d_volume.ipynb) |
| MONAI Bundle, Model Zoo | O | 06 🌶️ | [MONAI bundle docs](https://docs.monai.io/en/stable/bundle.html), tutorials [`bundle/`](https://github.com/Project-MONAI/tutorials/tree/main/bundle) |
| Regulation and reporting (awareness) | O | seminar | [FDA/HC/MHRA Good ML Practice](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles), [EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), [TRIPOD+AI](https://www.tripod-statement.org/) |

## Day 5 - The web app

Seminar: how a web app works (and how Streamlit simplifies it); environments and
reproducibility; containers; deployment options; responsible use of a public demo.

| Topic | Level | Where | Study further |
|---|---|---|---|
| VS Code, terminal, virtual environment, `requirements.txt` | M | app guide §1 | [Python venv](https://docs.python.org/3/library/venv.html), [uv](https://docs.astral.sh/uv/) |
| Streamlit: execution model, widgets, caching the model | M | app guide §2, steps 1-5 | [Streamlit docs](https://docs.streamlit.io/), [`st.cache_resource`](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource), [`st.file_uploader`](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) |
| Upload a slice, predict, overlay, lung area, Dice vs a reference mask | M | app steps 1-5 | - |
| Git and GitHub essentials | M | app guide §4 | [GitHub Hello World](https://docs.github.com/en/get-started/start-your-journey/hello-world), [Git cheat sheet](https://education.github.com/git-cheat-sheet-education.pdf) |
| Deploy on Streamlit Community Cloud | M | app guide §5 | [Community Cloud docs](https://docs.streamlit.io/deploy/streamlit-community-cloud) |
| Docker in practice: read a Dockerfile, build, run | M | app guide §6 | [Docker get started](https://docs.docker.com/get-started/), [Dockerfile best practices](https://docs.docker.com/build/building/best-practices/) |
| Window controls, model selector, mask download, tests with pytest | A | app steps 6-7, guide §3 | - |
| Compare two models, 3D volume slider, deploy the container elsewhere | O | app step 8 | tutorials [`deployment/`](https://github.com/Project-MONAI/tutorials/tree/main/deployment) (FastAPI, BentoML, Triton) |

## Side tracks (optional notebooks, for days 3-4)

| Topic | Level | Where | Study further |
|---|---|---|---|
| 3D segmentation: patches, 3D U-Net, sliding-window inference, lung volume in litres, 2D-stack vs 3D | O | O1 | tutorial [`spleen_segmentation_3d.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb), [`modules/2d_slices_from_3d_training.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/modules/2d_slices_from_3d_training.ipynb), [`auto3dseg/`](https://github.com/Project-MONAI/tutorials/tree/main/auto3dseg) |
| Strong pretrained baselines: lungmask, TotalSegmentator | O | O1 🌶️ | [lungmask](https://pypi.org/project/lungmask/), Wasserthal et al. 2023 ([arXiv](https://arxiv.org/abs/2208.05868)) |
| Classification / regression of the slice level; a badly defined target (`has_nodule`) | O | O2 A | tutorial [`2d_classification/mednist_tutorial.ipynb`](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb) |
| Embeddings: hooks, PCA / t-SNE, retrieval, linear probes, acquisition shortcuts | O | O2 B | - |
| Multimodal fusion (image + tabular), early vs late, leakage | O | O2 C | - |
| Foundation models for segmentation (discussion) | O | seminar day 4 | Ma et al. 2024, MedSAM ([Nat Commun](https://www.nature.com/articles/s41467-024-44824-z)) |

## General references

* MONAI: [docs](https://docs.monai.io/en/stable/), [tutorials](https://github.com/Project-MONAI/tutorials) (start with `2d_classification/monai_101.ipynb`), paper: Cardoso et al. 2022 ([arXiv](https://arxiv.org/abs/2211.02701)), [YouTube channel](https://www.youtube.com/@ProjectMONAI)
* PyTorch: [docs](https://pytorch.org/docs/stable/)
* Courses and books: [fast.ai Practical Deep Learning](https://course.fast.ai/), [Dive into Deep Learning](https://d2l.ai/), [AI for Medicine (Coursera)](https://www.coursera.org/specializations/ai-for-medicine)
* Data: LUNA16 - Setio et al. 2017 ([doi](https://doi.org/10.1016/j.media.2017.06.015)); LIDC-IDRI - Armato et al. 2011 ([doi](https://doi.org/10.1118/1.3528204)), [TCIA collection](https://www.cancerimagingarchive.net/collection/lidc-idri/)
