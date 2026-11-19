# Building the course dataset

The course uses one small dataset, `luna16-lung-slices`, built from LUNA16: one CT slice per patient with its
lung mask, a patch around each nodule, one demonstration scan and a few low-resolution 3D scans.

**What is done to the original data, and why, is described in [DATASET.md](DATASET.md).** This page only has
the commands.

A ready copy of the dataset exists: `luna16-lung-slices.zip` (396 MB), built on 18 November 2026 (nodule masks
replaced with the outlines of the radiologists on 19 November) with these
scripts. Unless you want to change how it is built, use that zip and go to "Upload to Kaggle". To keep it next
to the code without putting it in the repository, attach it to a Release (on GitHub: Releases, Draft a new
release, drop the zip in the assets box).

## Build it

Needs: Linux or macOS with `curl`, `unzip`, `md5sum`; Python with `numpy pandas nibabel SimpleITK pydicom tqdm`;
about **20 GB of free disk** and two to three hours.

```bash
./build_dataset.sh /data/luna_work luna16-lung-slices
```

The script works one LUNA16 subset at a time: it downloads it from Zenodo (7 GB), extracts it (12 GB), adds
its slices and its nodule patches to the dataset, deletes the scans, and goes on with the next one. At the end
it checks the result: the last lines must contain `RESULT: OK`. It can be interrupted and started again.

Options:

- fewer subsets, for a rehearsal: `./build_dataset.sh /data/luna_work luna16-lung-slices "0 1"`
  (each subset has about 89 patients and 110 nodules);
- number of 3D scans for notebook 05 (default 28): `VOLUMES=60 ./build_dataset.sh ...`
- nodule masks of the radiologists: `MASKS_DIR=/data/segmentations/masks ./build_dataset.sh ...` (one 3D mask per scan,
  `<seriesuid>.nii.gz`, on the grid of the LUNA16 scan). Without it the nodule masks are discs

The three Python scripts can also be used by hand, for example to change the rule that chooses the slice:
`prepare_luna16_slices.py` (slices, demonstration scan, 3D scans; `--help` lists the options, `--check`
validates a dataset), `prepare_nodule_patches.py` (nodule patches).

## Upload to Kaggle

1. kaggle.com -> Create -> New Dataset. Upload `luna16-lung-slices.zip` as it is (inside it, `images/`, `labels/` and the other folders are at the top level). If you built the dataset yourself, zip the content of the folder, not the folder.
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
| `nodules/` | 64 x 64 patches around each annotated nodule, with the mask drawn by the radiologists, for notebook 04 |
| `demo/` | one complete scan as a DICOM series (40 slices) and as a NIfTI volume, for notebook 01 |
| `volumes_lowres/` | 3D scans resampled to 2.5 mm, for notebook 05 |

The reasons behind each of these, the known limits (automatic lung masks, a few of them wrong; nodule masks
that come from LIDC-IDRI) and the numbers of the dataset are in [DATASET.md](DATASET.md).
