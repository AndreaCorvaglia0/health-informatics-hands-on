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
| 4 | Project: test set, using the model | `03_project_lung_segmentation`, STEP 12 and 13 |
| 5 | The web app | folder `webapp` |

If the class is ahead of schedule: `04_optional_3d_segmentation`.

## Where to start

1. Do part A of [SETUP.md](SETUP.md) at home, before the first day.
2. On day 1 we do part B together and run `00_environment_check`.
3. Do part C of the setup before day 5.

## How the material works

- **Notebooks 01 and 02** are lessons: run the cells, read, and do the exercises at the end.
- **Notebook 03** is the project, where you spend most of the time. It is divided in steps. Each step has a
  numbered list of things to do, a cell to complete, and a "You should see" to check your result.
- **`webapp/README.md`** is the guide for day 5.
- [SYLLABUS.md](SYLLABUS.md) lists the topics of each day and where to study them further.

Notebooks 01 to 04 run on Kaggle. The web app runs on your laptop.

## Data

The images come from LUNA16 (https://luna16.grand-challenge.org/), licence CC BY 4.0, a subset of
LIDC-IDRI (CC BY 3.0). Reference: Setio et al., Medical Image Analysis, 2017.

What we build here is a student project, not a medical device.

---

For teachers: the folders `solutions` and `teachers` are not for the students. Start from
`teachers/TEACHER_GUIDE.md`.
