# Teacher guide

## The idea

Students have no practical experience of medical images, PyTorch, Git or web apps. The course makes them do
one small project from start to finish, slowly, so that the basics stay. It follows last year's notebook and
web app step by step, moved from chest X-rays to CT.

- Little material, on purpose. Everything a student must do is written as numbered instructions, with the
  expected result after each step ("You should see").
- Most of the class time is the students working on the project (notebook 03) while we walk around.
  Who is behind gets help. Who is ahead finds short "Extra" items inside the steps and a list of experiments
  at the end of the notebook.
- If the whole class is fast, there is one optional block for everybody: notebook 04 (3D).

## Material

| File | What it is | Size |
|---|---|---|
| `SETUP.md` | accounts, Kaggle, laptop: three parts with checks | 2 pages |
| `00_environment_check.ipynb` | libraries, GPU, dataset: prints OK or what to fix | 5 minutes |
| `01_medical_images.ipynb` | lesson: CT, HU, windowing, masks, DICOM, NIfTI, 3D. 4 exercises | 60-75 min |
| `02_monai_transforms.ipynb` | lesson: dataset, MONAI transforms one by one, augmentation. 3 exercises | 60-75 min |
| `03_project_lung_segmentation.ipynb` | the project, 13 steps | days 2 to 4 |
| `04_optional_3d_segmentation.ipynb` | optional class block: the same pipeline in 3D, 8 steps | 2 hours |
| `webapp/` | day 5: guide (`README.md`), `app.py` to complete in 6 steps, Dockerfile, 3 sample slices | day 5 |
| `solutions/` | the same notebooks completed, with the answers to the "Check yourself" questions; `app.py` completed | |
| `teachers/` | this guide, `find_the_bugs.md`, the script that builds the dataset | |

Give the students everything except `solutions/` and `teachers/`.

## Schedule (24 hours)

| Day | Hours | Opening | Hands-on |
|---|---|---|---|
| 1 | 5 | Full introduction (about 1 h) | Setup part B and notebook 00 (30 min). Notebook 01. Notebook 02. |
| 2 | 5 | Seminar 45 min: segmentation, U-Net, Dice | Project STEP 1 to 7 |
| 3 | 5 | Seminar 45 min: how a network learns, curves, overfitting | Project STEP 8 to 11. Everybody downloads `lung_segmenter.pth`. |
| 4 | 5 | No seminar | Project STEP 12 and 13. Then: extra experiments, or notebook 04 with the class, or parts 1-3 of the web app guide (laptop, venv, first run) if you want a lighter day 5. |
| 5 | 4 | Short seminar: MLOps, Docker, the last part of the project | Web app guide parts 1 to 7 |

Rough time per step of the project, for a class with no experience:

| Steps | Content | Time |
|---|---|---|
| 1 - 3 | files, look at the data, split | 45 min |
| 4 | transforms | 45 min |
| 5 | Dataset and DataLoader | 45 min |
| 6 - 7 | model, loss, optimizer, metric | 60 min |
| 8 | one training step by hand | 30 min |
| 9 | training loop (plus the training itself) | 75 min |
| 10 - 11 | curves, predictions, download | 45 min |
| 12 | test set and worst cases | 60 min |
| 13 | function for a new image | 45 min |

That is about 7.5 hours of the 13.5 available on days 2 to 4: the rest is for explanations at the board, slow
students, and the extras. These times are estimates: nobody has taught this version yet.

## Before the course

1. **Build the dataset and put it on Kaggle**: `teachers/data_prep/README.md`. Name it `luna16-lung-slices`.
   Use `--volumes 100` if you want notebook 04 available.
2. **Run the solution notebooks on Kaggle with the GPU and the full dataset.** Write down: the time of one
   epoch, the validation Dice after 20 epochs, the mean test Dice. Then check the three sentences in
   notebook 03 that state expected values ("around 0.5 - 0.6" in STEP 7, "above 0.9" in STEP 9 and STEP 12)
   and correct them if needed. See "What was tested" below.
3. **Keep your `lung_segmenter.pth`**: it is the spare model for students who arrive on day 5 without one.
4. **Check the path of the dataset**: the notebooks use `/kaggle/input/luna16-lung-slices`. If Kaggle mounts
   your dataset elsewhere, change `DATA_DIR` in the five notebooks.
5. **Check the names of buttons and menus** in `SETUP.md` and `webapp/README.md` against the current Kaggle,
   VS Code and Streamlit Cloud screens. They were written from the documentation, not from the screens.
6. **Publish the reference app once** (`solutions/app.py` renamed `app.py`, with your model) on Streamlit
   Community Cloud. This confirms that the CPU-only PyTorch line in `requirements.txt` works there.
7. **Build the Docker image once** on your laptop. It was never built while preparing the material.
8. **Ask who has a Mac with an Intel processor.** Recent PyTorch versions have no installer for it:
   those students work in pairs on day 5.
9. Remind the students: Kaggle phone verification before day 1, setup part C before day 5.

## Things students get wrong

Day 1
- Forgetting `.T` when plotting (the image appears rotated).
- DICOM pixels are not Hounsfield units before slope and intercept.
- `"bilinear"` for the mask: values like 0.5 appear. Check 02, section 8.

Days 2 - 4
- The dataset is not attached, or the GPU is off: STEP 0 prints `cpu`.
- STEP 4: a transform missing or in the wrong order. The check cell under the step shows it: label values
  must be `[0. 1.]`.
- STEP 6: images not moved to the GPU (`Expected all tensors to be on the same device`).
- STEP 8 and 9: one of the five lines missing. Without `zero_grad` the loss behaves strangely; without
  `backward` or `step` it does not change.
- STEP 9: running the training cell twice continues the training of the same model. To start again, run from
  STEP 6.
- STEP 11: closing Kaggle without downloading the model.
- STEP 12: the worst test case is sometimes a patient whose "true" mask is wrong (in subset 0, `case_0011`:
  the air around the body is labelled as lung). The model is right. Use it: this is the moment to explain
  that the masks are automatic and what that means for the numbers.

Day 5
- Virtual environment not active (no `(.venv)` in the terminal): `streamlit` is not found.
- On Windows, the "execution policy" error when activating: the fix is in the troubleshooting table of the guide.
- The network in `app.py` different from the one trained: error when loading the weights.
- `.venv` pushed to GitHub: only if they removed `.gitignore`.

## Extras, in order of size

1. Inside the steps: one-line "Extra" suggestions (look at more patients, and so on).
2. At the end of notebook 03: six small experiments, each one a single change and a new training.
3. Larger challenges, with a teacher: right and left lung, threshold baseline, find the bugs
   (`teachers/find_the_bugs.md`), their own U-Net.
4. For the whole class: notebook 04.
5. Day 5: Dice with the true mask, sliders, download button, choice among several models.

Not developed, only mentioned in the references: lung nodules, nnU-Net, TotalSegmentator.

## What was tested

Everything was run on a computer without GPU, on 89 real LUNA16 patients (subset 0), not on the full 888.

| | Result |
|---|---|
| Notebooks 00, 01, 02 (solutions) | run without errors |
| Notebook 03 (solution) | runs in under 3 minutes on CPU. With 62 training patients: validation Dice 0.93 after 20 epochs, mean test Dice 0.94 on 14 patients, lowest 0.67 (the patient with the wrong mask) |
| Notebook 04 (solution) | runs. With 6 training scans: validation Dice 0.94 after 30 epochs, under 4 minutes on CPU |
| Web app | the completed app runs with the trained model: upload, three pictures, area, Dice. The student `app.py` runs as it is, with nothing filled in |
| Student notebooks | every cell is valid Python. They were not completed by hand as a student would |

Not tested: a GPU, Kaggle itself, the full dataset, the deploy on Streamlit Community Cloud, the Docker build,
the GitHub steps.

## Changing the material

The notebooks are plain `.ipynb` files: edit them directly. When you change an exercise, change the same cell
in `solutions/`.
