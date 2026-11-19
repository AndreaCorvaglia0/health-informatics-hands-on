# Health Informatics - Advanced Topics in Machine Learning: hands-on

In five days we go from a CT scan to a web app that finds the lungs in it.

You will open medical images, prepare them for a neural network, train a segmentation model,
check how good it is, and publish it as a web page.

## Programme

| Day | Topic | Material |
|---|---|---|
| 1 | Medical images and how to prepare them | `00_environment_check`, `01_medical_images`, `02_monai_transforms` |
| 2 | Project: data, transforms, model, loss | `03_project_lung_segmentation`, STEP 1 to 7 |
| 3 | Project: training and results | `03_project_lung_segmentation`, STEP 8 to 11 |
| 4 | Project: test set, using the model. Then lung nodules | `03_project_lung_segmentation`, STEP 12 and 13; `04_nodule_segmentation` |
| 5 | The web app | folder `webapp` |

Everybody finishes the project: a trained model that segments the lungs. Who finishes early goes on, without
waiting, to `04_nodule_segmentation`: the same code on a harder problem, with room for experiments.
`05_optional_3d_segmentation` is one more optional topic.

## Where to start

1. Do part A of [SETUP.md](SETUP.md) at home, before the first day.
2. On day 1 we do part B together and run `00_environment_check`.
3. Do part C of the setup before day 5.

## How the material works

- **Notebooks 01 and 02** are lessons: run the cells, read, and do the exercises at the end.
- **Notebook 03** is the project, where you spend most of the time. It is divided in steps. Each step has a
  numbered list of things to do, a cell to complete, and a "You should see" to check your result.
- **Notebook 04** is the advanced part: a baseline, its mistakes, and experiments to improve it.
- **`theory/theory.pdf`** is the handbook: the theory behind each part, in the order of the course.
- **`webapp/README.md`** is the guide for day 5.
- [SYLLABUS.md](SYLLABUS.md) lists the topics of each day and where to study them further.
- **`extra_multimodal/`** is a separate optional module on models that use images and clinical data together.
  Open it only if you have finished notebooks 03 and 04.

Notebooks 00 to 05 run on Kaggle. The web app runs on your laptop.

## Data

The images come from LUNA16 (https://luna16.grand-challenge.org/), licence CC BY 4.0, a subset of
LIDC-IDRI (CC BY 3.0). Reference: Setio et al., Medical Image Analysis, 2017.

What we build here is a student project, not a medical device.

---

For teachers: the folders `solutions` and `teachers` are not for the students. Start from
`teachers/TEACHER_GUIDE.md`. The slides of the seminars are listed in `seminars/README.md`.
