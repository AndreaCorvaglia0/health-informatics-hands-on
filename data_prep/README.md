# data_prep: building the course dataset (for teachers)

Goal: turn LUNA16 into the small dataset **`luna16-lung-slices`** (one axial CT slice per patient,
~250 MB for all 888 scans plus demo files), publish it as a Kaggle dataset and hand the same zip to
students who work on Colab or locally. You do this **once**, a few days before the course.

| Script | Purpose |
|---|---|
| `prepare_luna16_slices.py` | LUNA16 -> course dataset (`--download`, build, `--check`) |
| `train_baseline.py` | trains the "safety-net" U-Net and stores it in `pretrained/unet-baseline/` |
| `make_synthetic_dataset.py` | fake dataset with the same layout, for dry runs and CI |

Requirements: `pip install numpy pandas SimpleITK nibabel pydicom tqdm` (`torch` + `monai` for the baseline).

## 1. The data and your obligations

* **LUNA16**: 888 chest CTs (slice thickness <= 2.5 mm) from LIDC-IDRI, hosted on Zenodo
  ([part 1/2, DOI 10.5281/zenodo.3723295](https://zenodo.org/records/3723295): subsets 0-6, lung
  masks `seg-lungs-LUNA16.zip`, `annotations.csv`; [part 2/2, DOI 10.5281/zenodo.4121926](https://zenodo.org/records/4121926):
  subsets 7-9). Challenge page: <https://luna16.grand-challenge.org/Data/>.
* **Licence: CC BY 4.0** (the underlying LIDC-IDRI images are CC BY 3.0). You may redistribute a
  derivative, including on Kaggle, **if you give attribution and state that you changed it**. The
  script writes `LICENSE.txt` and a dataset card `README.md` into the output with the citation
  (Setio et al., Medical Image Analysis 42, 2017); keep both files when you publish, and put the same
  attribution in the Kaggle dataset description. Do not remove them from the zip.
* Raw data: CT in MetaImage (`subsetN/<seriesuid>.mhd` + `.raw`, int16 HU), masks in
  `seg-lungs-LUNA16/<seriesuid>.mhd` + `.zraw`.
* **Lung masks are "silver standard"**: computed by an automatic algorithm (van Rikxoort 2009); the
  organisers state they are *not intended as a reference standard*. Keep this as a teaching point.
* Not for clinical use. The data are de-identified; do not add any patient information.

## 2. Build the dataset

Disk and time (measured on a 2-core VM): raw subset ~6-7 GB each, all ten ~68 GB. Processing takes
~1.5 s per scan with 2 workers (demo and low-res volumes included). Output for all 888 scans: ~250 MB
of slices + ~75 MB demo + ~3 MB per low-res volume (40 volumes = ~115 MB).
`--download` extracts one zip at a time and deletes it, so the peak extra disk use is about 2 x 7 GB on top of what
is already extracted; but the build step reads all subsets in `--luna-dir`, so for the full dataset plan ~70 GB
(or build, then delete the raw data).

```bash
# a) fetch from Zenodo (prints sizes first, resumable, md5-checked, 16 parallel connections each)
python data_prep/prepare_luna16_slices.py --download --luna-dir /data/luna16 --dry-run     # just list
python data_prep/prepare_luna16_slices.py --download --luna-dir /data/luna16               # all 10 subsets
python data_prep/prepare_luna16_slices.py --download --luna-dir /data/luna16 --subsets 0   # or a few

# b) build (one random slice per patient; seed makes it reproducible)
python data_prep/prepare_luna16_slices.py --luna-dir /data/luna16 --out data/luna16-lung-slices \
    --volumes 40 --workers 4 --seed 0
```

Useful options: `--strategy {random,max_area,middle}` (default `random`: uniform among slices whose lung
area is >= `--min-area-frac` 0.3 of the scan maximum), `--demo-case case_0001` (default: first scan; its
full volume goes to `demo/volume`, a 40-slice central slab to `demo/dicom` and `demo/mhd`),
`--volumes N` (low-res 2.5 mm 3D volumes for the optional 3D track), `--drop-bad-masks`, `--limit N` (test).
A scan that cannot be read is skipped and reported at the end (and stored in `build_info.json`).
Do not use `--slices-per-patient > 1` for the course dataset (it breaks the one-slice-per-patient contract).

For a quick trial use `--subsets 0` (89 scans, ~7 GB, about 2 minutes of processing).
The case ids are `case_0001...` in sorted (subset, seriesuid) order.

## 3. Validate

```bash
python data_prep/prepare_luna16_slices.py --check --out data/luna16-lung-slices
```

Checks: every image/label pair loads and matches, 2D int16 / uint8, labels within {0,1,2,3}, affine axcodes
`('L','P','S')`, metadata rows == files and pixel counts == labels, right-lung centroid x < left-lung
centroid x in > 95% of cases, plausible HU range, demo DICOM/mhd readable with the same geometry and HU
as the NIfTI, low-res volumes. It prints a summary (cases per subset, spacing, % with padding values) and
exits with status 1 on any FAIL; WARN lines are informative (e.g. failed masks). It works on the synthetic
dataset too. Then **look at some cases yourself** (`plt.imshow(img.T, cmap="gray")` with the label
contour: right lung on the left of the picture, spine at the bottom).

## 4. Train the safety-net model and add it to the dataset

A model that everybody can load, in case a student's own training fails on day 2.

```bash
python data_prep/train_baseline.py --data data/luna16-lung-slices --epochs 30   # use a GPU (Kaggle/Colab)
# writes data/luna16-lung-slices/pretrained/unet-baseline/{model.pt,config.json}
```

Keep `pretrained/unet-baseline/` inside the dataset folder (and copy it to `app/models/` for day 5).

## 5. Publish on Kaggle and share with the class

1. Zip the folder so that the dataset root is at the top level: `cd data/luna16-lung-slices && zip -r ../luna16-lung-slices.zip .`
2. kaggle.com -> **Datasets -> New Dataset**, upload the files (or the zip, Kaggle unzips it), title
   `luna16-lung-slices`, licence **CC BY 4.0** (not "other"), paste the attribution/citation from `README.md`.
   CLI alternative: `kaggle datasets init -p data/luna16-lung-slices`, edit `dataset-metadata.json`
   (`licenses: [{"name": "CC-BY-4.0"}]`), `kaggle datasets create -p data/luna16-lung-slices -r zip`.
3. Visibility: keep it **private** while testing, then **Settings -> Sharing -> Public** (simplest for 30 students:
   they just click *Add Input* in a notebook) or add every student as collaborator/viewer for a private dataset.
4. Test with a fresh Kaggle notebook (GPU on, internet off): the data appear under `/kaggle/input/luna16-lung-slices/`.
5. Colab/local fallback: attach the same zip to a GitHub release or a shared drive link; students unzip it and set
   `HI_DATA_DIR` (see `tools/AUTHORING.md`). Dataset size is small enough for both.

## 6. Dry runs without any download: synthetic data

```bash
python data_prep/make_synthetic_dataset.py --out data/synthetic --n 60 --volumes 6
python data_prep/prepare_luna16_slices.py --check --out data/synthetic
```

Fake ellipses with the same layout (slices, `demo/`, `volumes_lowres/`). The generator is deterministic and
never deletes anything in `--out` (an existing `pretrained/` folder survives).

## 7. Layout and conventions (the contract)

| Item | Convention |
|---|---|
| `images/case_XXXX.nii.gz` | 2D NIfTI, int16, **HU**, typically 512x512, **no clipping** (padding -2000/-3024 kept) |
| `labels/case_XXXX.nii.gz` | 2D NIfTI, uint8: **0** background, **1 right lung, 2 left lung, 3 trachea** |
| Orientation | arrays in LPS+ index order (x -> patient's left, y -> posterior), affine in RAS world with negative x/y diagonal; `plt.imshow(img.T)` = radiological view; right lung at low x |
| `metadata.csv` | `case_id, seriesuid, subset, slice_index, n_slices, z_rel, spacing_x, spacing_y, slice_thickness, size_x, size_y, right_lung_px, left_lung_px, trachea_px, lung_area_mm2, has_nodule` + extras `padding_value_present, z_mm, reoriented, mask_outside_body` |
| `z_rel` | 0 = most inferior slice of the scan, 1 = most superior |
| `has_nodule` | 1 if the scan has an entry in LUNA16 `annotations.csv`, -1 if that file was not available |
| `demo/volume/<case>_{image,label}.nii.gz` | full 3D scan (day 1) |
| `demo/dicom/<case>/IM_XXXX.dcm` | central 40-slice slab as DICOM (stored value = HU + 1024, `RescaleIntercept` -1024, patient `LUNA16^case_0001`) |
| `demo/mhd/<case>.mhd` + `.raw` | same slab in MetaImage |
| `volumes_lowres/{images,labels}/`, `metadata.csv` | 3D at 2.5 mm isotropic, image linear / labels nearest; columns `case_id, seriesuid, subset, size_x, size_y, size_z, spacing_x, spacing_y, spacing_z` |
| Course split | subsets 0-6 train, 7-8 validation, 9 test |

Label mapping of the raw masks: seg-lungs value **3 = LEFT lung, 4 = RIGHT lung, 5 = trachea** -> course labels
2, 1, 3. The official page does not list these values; the mapping was verified on the data (the
smaller lung, the heart and the descending aorta are on the patient's left, which is the side of value 3 after
reorienting to LPS). If you ever change the mapping, change `SEG_TO_LABEL` and re-run `--check` plus a visual look.

## 8. Known caveats (teach them, do not hide them)

* **Silver-standard masks**: automatic, with local errors (vessels/hila, lung bases near liver or effusion, large
  consolidations). One scan of subset 0 (`case_0011`) has a *failed* mask (air outside the body labelled as lung,
  `mask_outside_body` ~0.45): the script avoids such slices when it can, flags the rest in `metadata.csv` and
  `--check` prints a WARN; use `--drop-bad-masks` to exclude them, or keep one as a classroom example.
* **Orientation**: the script reorients every scan to LPS with SimpleITK (`reoriented` = 1 if it had to flip
  axes; flipped scans are unusual in LUNA16 and none were found in subset 0). Tested on fake scans with flipped
  direction matrices.
* **Padding values** (-2000, -3024, -1000, -1024...) outside the circular field of view are kept on purpose
  (about two thirds of the scans of subset 0 have them): students must clip before normalising.
* **Heterogeneity**: pixel spacing 0.5-1.0 mm, slice thickness 0.5-2.5 mm, different scanners/kernels; the chosen
  slice can be at the lung base (liver, heart) or apex (small lungs, trachea only) - that is intended.
* The `has_nodule` flag only says that some nodule >= 3 mm was annotated somewhere in the scan, not on this slice.
* Case ids are assigned over the subsets found in `--luna-dir`: build from all ten subsets for canonical ids.
* Resampled volumes (`volumes_lowres/`) use plain linear interpolation (no anti-aliasing filter): thin structures
  and padding borders get intermediate values. Fine for the optional 3D track.
