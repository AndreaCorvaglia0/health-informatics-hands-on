# Syllabus

For each day: what everybody does (**core**), what is there for those who are ahead (**advanced** and
**extra**), and where to study further. The theory of every part is in `theory/theory.pdf`, in the same order.

## Day 1 - Medical images and how to prepare them

Course presentation, then two seminars: deep learning in medicine (an overview); medical images and their files.

| Topic | Level | Where |
|---|---|---|
| A CT slice as a matrix; Hounsfield units; windowing | core | 01, sections 2-4 |
| Segmentation masks; pixel spacing and real measurements | core | 01, sections 5-6 |
| DICOM and NIfTI: what is in the files | core | 01, sections 7-8 |
| A scan as a 3D volume | core | 01, section 9 |
| The dataset; dictionaries of files | core | 02, section 2 |
| MONAI transforms: load, channel, intensity, resize, Compose | core | 02, sections 3-9 |
| Data augmentation | core | 02, section 10 |
| Threshold segmentation; flips; wrong interpolation | extra | 01 and 02, "Extra" |

Study further
- Hounsfield units and windowing: https://radiopaedia.org/articles/hounsfield-unit , https://radiopaedia.org/articles/windowing-ct
- DICOM with pydicom: https://pydicom.github.io/pydicom/stable/
- NIfTI and coordinate systems: https://nipy.org/nibabel/coordinate_systems.html
- MONAI transforms: https://docs.monai.io/en/stable/transforms.html
- MONAI tutorial, 2D transforms: https://github.com/Project-MONAI/tutorials/blob/main/modules/transforms_demo_2d.ipynb

## Day 2 - Project: data, model, loss

Seminar: segmentation as a task, the U-Net, the Dice loss.

| Topic | Level | Where |
|---|---|---|
| Training, validation and test sets | core | 03, STEP 3 |
| Preprocessing and augmentation pipelines | core | 03, STEP 4 |
| Dataset and DataLoader, batches | core | 03, STEP 5 |
| The U-Net and its output | core | 03, STEP 6 |
| Loss, optimizer, metric | core | 03, STEP 7 |

Study further
- U-Net: https://arxiv.org/abs/1505.04597
- Dice loss (V-Net): https://arxiv.org/abs/1606.04797
- PyTorch, datasets and dataloaders: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- MONAI in one notebook: https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/monai_101.ipynb

## Day 3 - Project: training and results

Seminar: how a network learns, reading the learning curves, overfitting.

| Topic | Level | Where |
|---|---|---|
| The five lines of a training step | core | 03, STEP 8 |
| The training loop, validation, saving the best model | core | 03, STEP 9 |
| Learning curves | core | 03, STEP 10 |
| Looking at the predictions; downloading the model | core | 03, STEP 11 |

Study further
- PyTorch, the optimisation loop: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- A recipe for training neural networks: https://karpathy.github.io/2019/04/25/recipe/
- MONAI tutorial, the same pipeline on the spleen: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb

## Day 4 - Project: is it good? Then lung nodules

No seminar. Everybody finishes the project. Who has finished works on the nodules.

| Topic | Level | Where |
|---|---|---|
| Performance on the test set, patient by patient; worst cases | core | 03, STEP 12 |
| Using the model on a new file; same preprocessing as in training | core | 03, STEP 13 |
| Lung nodules: patches, split by patient, a baseline and its mistakes | advanced | 04, STEP 1 to 7 |
| Experiments: noise between runs, loss, augmentation, learning rate, network size | advanced | 04, STEP 8 to 10 |
| Threshold baseline; right and left lung; find the bugs | extra | 03, "If you have finished" |
| 3D segmentation | extra | 05 |

Study further
- Choosing metrics, Metrics Reloaded: https://arxiv.org/abs/2206.01653
- MONAI metrics and losses: https://docs.monai.io/en/stable/metrics.html , https://docs.monai.io/en/stable/losses.html
- Lung segmentation is a data problem (Hofmanninger et al. 2020): https://doi.org/10.1186/s41747-020-00173-2
- LUNA16, the nodule challenge: https://luna16.grand-challenge.org/
- Losses for segmentation, a comparison: https://arxiv.org/abs/2005.13449
- nnU-Net: https://arxiv.org/abs/1809.10486

## Day 5 - The web app

Short seminar: from a model to a product (MLOps), containers, the last part of the project.

| Topic | Level | Where |
|---|---|---|
| VS Code, virtual environment, packages | core | webapp guide, parts 1-3 |
| A Streamlit app around the model | core | webapp guide, part 4 |
| GitHub | core | webapp guide, part 5 |
| Publishing the app | core | webapp guide, part 6 |
| Docker | core (demo), extra (do it yourself) | webapp guide, part 7 |
| Dice with the true mask, sliders, download, choice of the model | extra | webapp guide, "If you have finished" |

Study further
- Streamlit: https://docs.streamlit.io/
- Virtual environments: https://docs.python.org/3/library/venv.html
- GitHub, first steps: https://docs.github.com/en/get-started/start-your-journey/hello-world
- Docker, getting started: https://docs.docker.com/get-started/
- Model cards, how to describe a model honestly: https://arxiv.org/abs/1810.03993
- Hidden technical debt in machine learning systems: https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html

## Extra module - Multimodal fusion (optional, outside the five days)

Only for who has finished the project, the nodules and wants more. Separate folder: `extra_multimodal/`.
The clinical data of this module are invented; the images are the nodule patches of notebook 04.

| Topic | Level | Where |
|---|---|---|
| A clinical table and a model on it | extra | multimodal_fusion, STEP 1-4 |
| Measurements from a mask (radiomics) | extra | multimodal_fusion, STEP 5 |
| Early and late fusion | extra | multimodal_fusion, STEP 6-7 |
| Embeddings from a pretrained network | extra | multimodal_fusion, STEP 8 |
| Intermediate fusion | extra | multimodal_fusion, STEP 9 |

Theory: `extra_multimodal/handbook.pdf`.

Study further
- Fusion of imaging and health records (early, joint, late): https://pmc.ncbi.nlm.nih.gov/articles/PMC7567861
- MONAI tutorials, multimodal: https://github.com/Project-MONAI/tutorials/tree/main/multimodal
- CMU course on multimodal machine learning: https://cmu-mmml.github.io/
- PyRadiomics: https://pyradiomics.readthedocs.io
- Survival analysis, basic concepts: https://pmc.ncbi.nlm.nih.gov/articles/PMC2394262/
