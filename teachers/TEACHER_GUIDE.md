# Teacher guide

## The idea

Students have no practical experience of medical images, PyTorch, Git or web apps. The course makes them do
one small project from start to finish, slowly, so that the basics stay. It follows last year's notebook and
web app step by step, moved from chest X-rays to CT.

- Little material, on purpose. Everything a student must do is written as numbered instructions, with the
  expected result after each step ("You should see").
- **The guaranteed result is the lung model**: every student must reach the end of the project (notebook 03,
  STEP 13) with a trained U-Net. While they work we walk around and help who is behind.
- **Who is ahead does not wait.** After the project they go on to notebook 04, lung nodules: the same code on
  a harder problem. Lungs are easy and leave little to improve; with nodules the choices change the result,
  so the experiments are there (baseline, noise between two runs, loss, augmentation, and a list of others).
- The theory is in one PDF, `theory/theory.pdf`, in the order of the course. Each section says in which
  notebook it is used; the notebooks contain only the minimum explanation.

## Material

| File | What it is | Size |
|---|---|---|
| `SETUP.md` | accounts, Kaggle, laptop: three parts with checks | 2 pages |
| `00_environment_check.ipynb` | libraries, GPU, dataset: prints OK or what to fix | 5 minutes |
| `01_medical_images.ipynb` | lesson: CT, HU, windowing, masks, DICOM, NIfTI, 3D. 4 exercises | 60-75 min |
| `02_monai_transforms.ipynb` | lesson: dataset, MONAI transforms one by one, augmentation. 3 exercises | 60-75 min |
| `03_project_lung_segmentation.ipynb` | the project, 13 steps | days 2 to 4 |
| `04_nodule_segmentation.ipynb` | advanced, for who has finished the project: nodule patches, baseline, experiments. 10 steps | 3 hours or more |
| `05_optional_3d_segmentation.ipynb` | optional: the same pipeline in 3D, 8 steps | 2 hours |
| `theory/theory.pdf` | handbook for the students, 13 pages (source `theory.tex`, figures from `make_figures.py`) | |
| `webapp/` | day 5: guide (`README.md`), `app.py` to complete in 6 steps, Dockerfile, 3 sample slices | day 5 |
| `solutions/` | the same notebooks completed, with the answers to the "Check yourself" questions; `app.py` completed | |
| `teachers/` | this guide, `find_the_bugs.md`, the script that builds the dataset | |

Give the students everything except `solutions/` and `teachers/`.

## Schedule (24 hours)

| Day | Hours | Opening | Hands-on |
|---|---|---|---|
| 1 | 5 | Full introduction (about 1 h) | Setup part B and notebook 00 (30 min). Notebook 01. Notebook 02. |
| 2 | 5 | Seminar 45 min: segmentation, U-Net, Dice | Project STEP 1 to 7 |
| 3 | 5 | Seminar 45 min: how a network learns, curves, overfitting | Project STEP 8 to 11. Everybody downloads `lung_segmenter.pth`. Fast students: STEP 12 and 13, then notebook 04. |
| 4 | 5 | No seminar | Project STEP 12 and 13 for who still has them, with our help. Everybody else: notebook 04. At the end of the day, if there is time: parts 1-3 of the web app guide (laptop, venv, first run), for a lighter day 5. |
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

That is about 7.5 hours of the 13.5 available on days 2 to 4. A student at this pace finishes the project in
the first half of day 4 and has two or three hours for the nodules; a fast one reaches them on day 3; a slow
one uses all of day 4 for the lungs, and that is fine. These times are estimates: nobody has taught this
version yet.

Notebook 04, for reference: STEP 1 to 4 (data) 40 min, STEP 5 to 7 (baseline and its mistakes) 40 min,
STEP 8 (noise) 15 min, STEP 9 (two guided experiments, then free ones) from one hour up, STEP 10 (test) 15 min.

## Before the course

1. **Build the dataset and put it on Kaggle**: `teachers/data_prep/README.md`. Name it `luna16-lung-slices`.
   It needs about 130 GB of free disk. Step 4 adds the nodule patches for notebook 04; `--volumes 100` adds
   the 3D scans for notebook 05.
2. **Run the solution notebooks on Kaggle with the GPU and the full dataset.** Write down: the time of one
   epoch, the validation Dice after 20 epochs, the mean test Dice. Then check the three sentences in
   notebook 03 that state expected values ("around 0.5 - 0.6" in STEP 7, "above 0.9" in STEP 9 and STEP 12)
   and correct them if needed. Do the same for notebook 04: note the baseline Dice, the noise between the two
   seeds and the effect of the two guided experiments, so that you know what to expect when students show you
   their numbers. See "What was tested" below.
3. **Keep your `lung_segmenter.pth`**: it is the spare model for students who arrive on day 5 without one.
4. **Check the path of the dataset**: the notebooks use `/kaggle/input/luna16-lung-slices`. If Kaggle mounts
   your dataset elsewhere, change `DATA_DIR` in the six notebooks.
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

Notebook 04
- Comparing two runs that differ in more than one thing.
- Calling a difference smaller than the noise (STEP 8) an improvement.
- Forgetting to create a new model before an experiment: the training continues from the previous one.
- Reading the Dice as if the masks were real outlines: they are discs (see below).

## For who is ahead, in order

1. Inside the steps of the project: one-line "Extra" suggestions.
2. **Notebook 04, lung nodules**: the main path after the project.
3. At the end of notebook 03, three short challenges on the lungs: threshold baseline, right and left lung,
   find the bugs (`teachers/find_the_bugs.md`).
4. Notebook 05, 3D.
5. Day 5: Dice with the true mask, sliders, download button, choice among several models.

## About the nodule data

LUNA16 gives the centre and the diameter of each nodule, not its outline. The masks of notebook 04 are discs
of the annotated diameter, drawn on the slice through the centre of the nodule, in a 64 x 64 patch where the
nodule is shifted at random from the centre. The notebook says so to the students and asks them to keep it in
mind: part of the "error" is the mask.

If your colleague prepares the radiologist outlines from LIDC-IDRI, put them in the same layout
(`teachers/data_prep/README.md`): the notebook does not change, and the discussion on how much radiologists
disagree becomes possible.

Every patch contains a nodule. Patches without nodules, and detection in a whole scan, are only mentioned in
the questions at the end.

## What was tested

Everything was run on a computer without GPU, on 89 real LUNA16 patients (subset 0), not on the full 888.

| | Result |
|---|---|
| Notebooks 00, 01, 02 (solutions) | run without errors |
| Notebook 03 (solution) | runs in under 3 minutes on CPU. With 62 training patients: validation Dice 0.93 after 20 epochs, mean test Dice 0.94 on 14 patients, lowest 0.67 (the patient with the wrong mask) |
| Notebook 04 (solution) | runs in 5 minutes on CPU. With 78 training nodules: baseline validation Dice 0.44, second seed 0.39 (noise 0.05), DiceCE loss 0.55, augmentation 0.39, test Dice 0.50 on 18 nodules. With so few nodules these numbers are unstable: the full dataset has ten times more |
| Notebook 05 (solution) | runs. With 6 training scans: validation Dice 0.94 after 30 epochs, under 4 minutes on CPU |
| Handbook | compiles with pdflatex, 13 pages |
| Web app | the completed app runs with the trained model: upload, three pictures, area, Dice. The student `app.py` runs as it is, with nothing filled in |
| Student notebooks | every cell is valid Python. They were not completed by hand as a student would |

Not tested: a GPU, Kaggle itself, the full dataset, the deploy on Streamlit Community Cloud, the Docker build,
the GitHub steps.

## Changing the material

The notebooks are plain `.ipynb` files: edit them directly. When you change an exercise, change the same cell
in `solutions/`.
