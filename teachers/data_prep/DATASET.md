# From LUNA16 to the course dataset: what we do and why

This document describes every transformation between the original LUNA16 files and the dataset
`luna16-lung-slices` used in the course, with the reason for each choice. The commands are in `README.md`.

## 1. The source

**LUNA16** (https://luna16.grand-challenge.org/) is a public dataset for lung nodule analysis, licence CC BY 4.0.

| What | Details |
|---|---|
| Scans | 888 chest CT scans, selected from LIDC-IDRI (scans with slices thicker than 2.5 mm were left out) |
| Format | MetaImage: a text header `.mhd` and the voxels in `.raw`, values in Hounsfield units |
| Organisation | 10 folders `subset0` ... `subset9` of about 89 scans each; one file per scan, named with its series UID |
| Lung masks | `seg-lungs-LUNA16`: one mask per scan, produced by an **automatic algorithm**. Values: 3 = left lung, 4 = right lung, 5 = trachea (checked on the data; the LUNA16 page does not state them) |
| Nodules | `annotations.csv`: 1186 nodules of at least 3 mm accepted by at least 3 of 4 radiologists. For each: position of the centre in millimetres and diameter. **No outlines.** |
| Size | 68 GB to download, about 120 GB extracted |

A scan has between 100 and 400 slices of 512 x 512 pixels. The whole dataset is far too large for a classroom:
the course dataset keeps a very small part of it, chosen so that the exercises stay simple and fast.

## 2. Lung slices (`images/`, `labels/`, `metadata.csv`)

For each scan:

| Step | What we do | Why |
|---|---|---|
| 1. Read | Read the `.mhd` scan and its lung mask with SimpleITK | |
| 2. Orient | Reorient both to the LPS convention (x towards the patient's left, y towards the back, z towards the head) | A few LUNA16 scans are stored with flipped axes. After this step all images look the same way and `plt.imshow(image.T)` always gives the standard radiological view |
| 3. Choose one slice | Among the axial slices where the lung area is at least 30% of the largest lung area of that scan, pick one at random (fixed seed) | See below |
| 4. Keep the values | Save the slice in Hounsfield units, as 16-bit integers, without windowing and without removing the padding values (-2000, -3024) outside the field of view | Windowing and clipping are things the students must learn to do (notebooks 01 and 02). A dataset already cleaned would hide them |
| 5. Relabel the mask | 4 -> 1 (right lung), 3 -> 2 (left lung), 5 -> 3 (trachea), everything else 0 | Consecutive labels are easier to explain. The two lungs stay separate, so the "right and left lung" challenge is possible. The binary lung mask is built in the notebooks (`keep_lungs`), which is one more transform to learn |
| 6. Save | One 2D NIfTI file for the image and one for the mask, same name. The header carries the pixel spacing and the slice thickness | NIfTI is the format the students will meet in research. The spacing is needed to compute areas in cm² |
| 7. Describe | One row in `metadata.csv`: identifiers, original subset, position of the slice (`slice_index`, `n_slices`, `z_rel`), spacing, thickness, lung areas, `has_nodule`, and quality flags | Lets the students explore the dataset, and lets us explain bad results (for example slices near the top of the lungs) |

**Why one slice per patient.** Three reasons. The dataset becomes small (about 300 kB per patient) and a
training takes minutes. Each patient appears once, so a random split into training, validation and test can never
put the same patient in two sets: the students can use the simple `train_test_split` they already know. And the
2D case is the easiest place to learn the pipeline.

**Why a random slice, and why the 30% rule.** If we always took the middle slice, all images would look alike:
two large lungs and the heart. The model would be excellent on those and we would learn nothing about its limits.
A random height gives slices from the top to the base of the lungs, including small and oddly shaped ones, which
are the cases where the model fails and where the discussion is interesting. The 30% threshold removes the
extreme slices where there is almost no lung: there the mask is a few pixels and the Dice becomes meaningless.

**Patient numbering.** Scans are sorted by subset and then by series UID and numbered `case_0001`, `case_0002`,
... The mapping to the original series UID is in `metadata.csv`.

**What we do not correct: wrong masks.** The lung masks are automatic and a few are wrong (in subset 0, one
scan out of 89 has the air around the body labelled as lung). We keep them and flag them: the column
`mask_outside_body` is the fraction of "lung" pixels that lie outside the body, and `--check` lists the cases
above 2%. When the students look at their worst test cases they find some of these, see that the model is right
and the reference is wrong, and learn what "automatic labels" means. `--drop-bad-masks` removes them if you
prefer a clean dataset.

## 3. Nodule patches (`nodules/`)

Made by `prepare_nodule_patches.py` from the same scans and `annotations.csv`. For each nodule:

| Step | What we do | Why |
|---|---|---|
| 1. Find the slice | Convert the centre of the nodule from millimetres to voxel position, and take the axial slice closest to it | It is the slice where the nodule is largest and the annotation most reliable |
| 2. Cut a patch | 64 x 64 pixels at the original resolution (about 45 mm), in Hounsfield units | A nodule is a few pixels in a 512 x 512 slice: on the whole slice the network would learn almost only background. 64 pixels contain the largest nodules (30 mm) with some surroundings, and are divisible by 8 as the network requires |
| 3. Shift the nodule | The nodule is placed at a random position within 14 pixels of the centre of the patch | If the nodule were always in the middle, the model would learn "draw a blob in the middle" and reach a good Dice without looking at the image |
| 4. Draw the mask | A disc. The nodule is treated as a sphere with the annotated diameter; the disc is the section of that sphere at the distance between the slice and the centre | LUNA16 has no outlines. The simple choice, a disc of the full diameter, is too large when the slice does not pass through the centre: with 2.5 mm slices the offset can reach 1.25 mm, and for a 4 mm nodule the full disc would be about 30% too wide. The section of the sphere corrects this |
| 5. Describe | One row in `nodules/metadata.csv`: nodule and patient identifiers, subset, annotated diameter, diameter of the disc actually drawn, spacing | The patient identifier is needed to split by patient: one patient can have several nodules |

**One patch per nodule.** We could also cut the neighbouring slices and get two to four times more patches. We
do not: they would be near copies of each other, and the further from the centre, the worse the "sphere"
approximation of the mask.

**Limits to state openly.** The masks are discs, real nodules are not: no model can reach a Dice of 1, and part
of the measured error is in the masks. Every patch contains a nodule, so this is segmentation of a nodule already
found, not detection in a scan. Both points are written in notebook 04.

**If radiologist outlines become available** (they exist in LIDC-IDRI), save them in the same layout and nothing
else changes: `nodules/images/<id>.nii.gz`, `nodules/labels/<id>.nii.gz` with 0 and 1, and
`nodules/metadata.csv` with at least `nodule_id`, `case_id` and `diameter_mm`.

## 4. The demonstration scan (`demo/`)

Notebook 01 needs one complete scan, to show the 3D volume and the DICOM format. From the first patient we save:

| File | Content | Why |
|---|---|---|
| `demo/volume/case_0001_image.nii.gz`, `..._label.nii.gz` | the whole scan and its mask, as NIfTI | the three views (axial, coronal, sagittal) |
| `demo/dicom/case_0001/*.dcm` | 40 central slices written as a DICOM series | LUNA16 does not distribute DICOM files, and the students must open some. We write them ourselves from the scan: the pixels are real, the header is minimal and the patient name is clearly fake (`LUNA16^case_0001`). The stored values are HU + 1024 with `RescaleIntercept = -1024`, so that the conversion to HU has to be done |
| `demo/mhd/` | the same 40 slices in MetaImage | to show the original format |

Only 40 slices are written as DICOM to keep the dataset small.

## 5. Low-resolution volumes (`volumes_lowres/`)

For the optional 3D notebook (05). The first 40 scans are resampled to voxels of 2.5 mm in every direction
(linear interpolation for the image, nearest neighbour for the mask): about 150 x 150 x 120 voxels and 3 MB per
scan.

Why reduced: a full scan has tens of millions of voxels and does not fit in the memory of a free GPU without
techniques (patches, sliding window) that are beyond this course. Why 40: the lungs are an easy target and a
3D model learns them from a few dozen scans; more would only make each epoch slower.

## 6. What the course dataset is not

- It is 2D: one slice says nothing about what is above and below.
- The lung masks are the output of an algorithm, the nodule masks are discs. Neither is a reference drawn by
  experts.
- All scans come from one public collection of adults who were screened or examined for lung nodules.
- It is for teaching. Nothing trained on it is a clinical tool.

## 7. Reproducing it

- The random choices (slice of each patient, shift of each nodule) depend only on `--seed` and on the patient,
  so building one subset at a time (`--append`, used by `build_dataset.sh`) gives the same result as building all
  together.
- `build_info.json` in the dataset records the options used.
- `prepare_luna16_slices.py --check` verifies the result: every image has its mask, same size, labels in
  {0, 1, 2, 3}, orientation, right lung on the correct side, plausible Hounsfield range, consistency of the demo
  files.

## 8. Numbers of the dataset built for the course

BUILD_NUMBERS

## 9. Licence and citation

LUNA16 is distributed under CC BY 4.0; it derives from LIDC-IDRI (CC BY 3.0). The dataset built here can be
redistributed with attribution: keep the files `README.md` and `LICENSE.txt` that the script writes in it.

- Setio et al., *Validation, comparison, and combination of algorithms for automatic detection of pulmonary
  nodules in computed tomography images: the LUNA16 challenge*, Medical Image Analysis, 2017.
  https://doi.org/10.1016/j.media.2017.06.015
- Armato et al., *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)*,
  Medical Physics, 2011. https://doi.org/10.1118/1.3528204
