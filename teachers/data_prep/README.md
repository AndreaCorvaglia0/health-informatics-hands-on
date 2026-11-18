# Building the course dataset

`prepare_luna16_slices.py` turns LUNA16 into the small dataset used in the course: one axial CT slice per
patient with its lung mask. You run it once and upload the result to Kaggle.

Needs: Python with `numpy pandas nibabel SimpleITK pydicom tqdm`, and about 15 GB of free disk
(the script downloads and deletes one LUNA16 subset at a time; the ten subsets are 68 GB in total).

## Steps

```bash
# 1. Download LUNA16 from Zenodo (10 subsets of 89 scans each). About 20 minutes per subset.
python prepare_luna16_slices.py --download --luna-dir /data/luna16

# 2. Build the dataset. About 2 seconds per scan.
#    --volumes 100 also writes 100 low-resolution 3D scans for the optional notebook 04 (about 300 MB).
python prepare_luna16_slices.py --luna-dir /data/luna16 --out luna16-lung-slices --volumes 100

# 3. Check it. The last line must be "RESULT: OK".
python prepare_luna16_slices.py --check --out luna16-lung-slices
```

For a rehearsal one subset is enough: add `--subsets 0` to steps 1 and 2 (89 patients, 7 GB, two minutes).

## Upload to Kaggle

1. kaggle.com -> Create -> New Dataset. Upload the folder `luna16-lung-slices`.
2. Title: `luna16-lung-slices`. The notebooks expect the path `/kaggle/input/luna16-lung-slices`, with
   `images/` and `labels/` directly inside. If Kaggle mounts it under another path, change the line
   `DATA_DIR = ...` at the top of each notebook before giving them to the students.
3. Licence: CC BY 4.0. Keep the files `README.md` and `LICENSE.txt` written by the script (attribution).
4. Make it public, or private and shared with the students' accounts.

## What is inside

| | |
|---|---|
| `images/case_0001.nii.gz` | 2D NIfTI, 512 x 512, Hounsfield units |
| `labels/case_0001.nii.gz` | 0 background, 1 right lung, 2 left lung, 3 trachea |
| `metadata.csv` | one row per patient: pixel spacing, slice thickness, position of the slice (`z_rel`), lung area |
| `demo/` | one complete scan as a DICOM series (40 slices) and as a NIfTI volume, for notebook 01 |
| `volumes_lowres/` | 3D scans resampled to 2.5 mm, for notebook 04 (only with `--volumes`) |

Arrays are stored with x as the first axis, so `plt.imshow(image.T)` shows the usual radiological view.

## Things to know

- **The masks are automatic.** LUNA16 states that they are not a reference standard. Most are very good.
- **Some masks are wrong.** In subset 0 one patient (`case_0011`) has the air around the body labelled as lung.
  `--check` lists these cases. With `--drop-bad-masks` the script leaves them out. We suggest keeping them:
  when students look at their worst test cases (STEP 12 of the project) they find that the model is right and
  the "true" mask is wrong, which is the best discussion of the week.
- **Raw labels.** In the original LUNA16 masks 3 is the left lung and 4 the right lung (checked on the data).
  The script converts them to 1 = right, 2 = left.
- Tested on subset 0 only (89 scans). The download of all ten subsets was not run end to end.
