#!/usr/bin/env python3
"""Build the course dataset ``luna16-lung-slices`` (one axial chest-CT slice per patient) from LUNA16.

Teacher workflow (run once, then publish the output folder as a Kaggle dataset)::

    # 1. (optional) fetch LUNA16 from Zenodo, ~68 GB for all ten subsets, ~7 GB per subset.
    #    Zips are extracted one at a time and deleted afterwards (peak disk use ~ 2 x 7 GB).
    python prepare_luna16_slices.py --download --luna-dir /data/luna16 --subsets 0 1 2 3 4 5 6 7 8 9

    # 2. build the course dataset (about 1-2 s per scan + the demo/volumes extras)
    python prepare_luna16_slices.py --luna-dir /data/luna16 --out data/luna16-lung-slices \\
        --volumes 40 --workers 4

    # 3. validate the result
    python prepare_luna16_slices.py --check --out data/luna16-lung-slices

    # 4. upload the output folder to Kaggle as a dataset named luna16-lung-slices (see README.md)

Input  (``--luna-dir``): ``subset0/ ... subset9/`` (``<seriesuid>.mhd`` + ``.raw``, HU, int16),
``seg-lungs-LUNA16/`` (``<seriesuid>.mhd`` + ``.zraw``; 3 = LEFT lung, 4 = RIGHT lung, 5 = trachea,
verified on the data - see the README), optionally ``annotations.csv``.
Output (``--out``): ``images/``, ``labels/``, ``metadata.csv``,
``demo/{volume,dicom,mhd}``, ``volumes_lowres/``, plus ``README.md`` and ``LICENSE.txt``.

Conventions: every volume is reoriented to LPS (``sitk.DICOMOrient(img, "LPS")``), arrays are stored
[x, y(, z)] and the NIfTI affine is the RAS-world affine of an LPS array (``lps_affine``), so ``plt.imshow(img.T)`` is the radiological view.  Labels:
0 background, 1 right lung, 2 left lung, 3 trachea.  HU values are never clipped.

Licence: LUNA16 is CC BY 4.0 (derived from LIDC-IDRI, CC BY 3.0).  Keep the attribution that this
script writes into the output.  Lung masks are automatic (silver standard).  Not for clinical use.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional: fall back to a plain iterator
    def tqdm(it, **_kw):
        return it

# --------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------
# LUNA16 seg-lungs values -> course labels.  Verified empirically on real scans (heart, aorta and the
# smaller volume are on the side of value 3 = patient's LEFT); the official page does not list values.
SEG_TO_LABEL = {3: 2, 4: 1, 5: 3}  # 3 left lung -> 2, 4 right lung -> 1, 5 trachea -> 3
ZENODO_RECORDS = (3723295, 4121926)  # subsets 0-6 (+ seg, csv) and subsets 7-9
METADATA_COLUMNS = [
    "case_id", "seriesuid", "subset", "slice_index", "n_slices", "z_rel", "spacing_x", "spacing_y",
    "slice_thickness", "size_x", "size_y", "right_lung_px", "left_lung_px", "trachea_px",
    "lung_area_mm2", "has_nodule",
]
EXTRA_COLUMNS = ["padding_value_present", "z_mm", "reoriented", "mask_outside_body"]
VOLUME_COLUMNS = ["case_id", "seriesuid", "subset", "size_x", "size_y", "size_z", "spacing_x", "spacing_y",
                  "spacing_z"]
PADDING_THRESHOLD = -1500  # HU below this cannot be tissue/air: padding outside the field of view
CITATION = (
    "Setio, A. A. A., Traverso, A., de Bel, T., et al. (2017). Validation, comparison, and combination of "
    "algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 "
    "challenge. Medical Image Analysis, 42, 1-13. https://doi.org/10.1016/j.media.2017.06.015")


def lps_affine(spacing, origin=(0.0, 0.0, 0.0)):
    """NIfTI affine (RAS+ world) for an array stored in LPS+ index order (same as the synthetic generator)."""
    sx, sy, sz = spacing
    aff = np.diag([-sx, -sy, sz, 1.0])
    aff[:3, 3] = (-origin[0], -origin[1], origin[2])
    return aff


def save_nifti(arr, affine, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr, affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


# --------------------------------------------------------------------------------------------------
# Reading a LUNA16 scan
# --------------------------------------------------------------------------------------------------
def read_scan(mhd: Path, seg_mhd: Path):
    """Read CT + lung mask, reorient both to LPS, return (img_xyz int16, label_xyz uint8, spacing, origin, reoriented).

    Arrays are indexed [x, y, z]; ``z`` = 0 is the most inferior slice (LPS: z grows towards Superior).
    The label array is remapped to the course convention.  Raises ValueError if the two grids disagree.
    """
    import SimpleITK as sitk

    img0 = sitk.ReadImage(str(mhd))
    seg0 = sitk.ReadImage(str(seg_mhd))
    reoriented = not np.allclose(img0.GetDirection(), (1, 0, 0, 0, 1, 0, 0, 0, 1), atol=1e-4)
    img = sitk.DICOMOrient(img0, "LPS")
    seg = sitk.DICOMOrient(seg0, "LPS")
    if not np.allclose(img.GetDirection(), np.eye(3).ravel(), atol=1e-4):
        raise ValueError(f"direction not identity after reorientation: {img.GetDirection()}")
    if img.GetSize() != seg.GetSize() or not np.allclose(img.GetOrigin(), seg.GetOrigin(), atol=1e-2) \
            or not np.allclose(img.GetSpacing(), seg.GetSpacing(), atol=1e-4):
        # different grid: resample the mask onto the CT grid (nearest neighbour keeps the label values)
        seg = sitk.Resample(seg, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg.GetPixelID())
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)  # sitk is [z, y, x] -> [x, y, z]
    raw = sitk.GetArrayFromImage(seg).transpose(2, 1, 0)
    lut = np.zeros(256, np.uint8)
    for src, dst in SEG_TO_LABEL.items():
        lut[src] = dst
    lab = lut[np.clip(raw, 0, 255).astype(np.uint8)]
    return (np.ascontiguousarray(arr.astype(np.int16)), np.ascontiguousarray(lab), img.GetSpacing(),
            img.GetOrigin(), reoriented)


def outside_body_frac(img2: np.ndarray, lab2: np.ndarray) -> float:
    """Fraction of the lung-labelled pixels of a slice that lie outside the body.

    The automatic masks occasionally fail by labelling the air between the body and the edge of the
    field of view as lung.  Body = largest connected tissue (> -300 HU) component with holes filled.
    """
    import SimpleITK as sitk

    lung = (lab2 == 1) | (lab2 == 2)
    if not lung.any():
        return 0.0
    tissue = sitk.GetImageFromArray((img2.T > -300).astype(np.uint8))
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(tissue), sortByObjectSize=True)
    body = sitk.GetArrayFromImage(sitk.BinaryFillhole(cc == 1)).astype(bool).T
    return float((lung & ~body).sum() / lung.sum())


def choose_slices(vol: np.ndarray, lab: np.ndarray, strategy: str, n: int, min_area_frac: float, rng,
                  max_outside: float = 0.02) -> list[int]:
    """Pick slice indices (z) according to the strategy; only slices that contain lung are eligible.

    Candidates whose mask visibly failed (more than ``max_outside`` of the lung pixels outside the body)
    are skipped unless no other candidate exists.
    """
    area = ((lab == 1) | (lab == 2)).sum(axis=(0, 1))
    if area.max() == 0:
        raise ValueError("no lung in the mask")
    lung_z = np.flatnonzero(area)
    if strategy == "max_area":
        cand = [int(i) for i in np.argsort(-area, kind="stable")[: max(n * 10, 20)] if area[i] > 0]
    elif strategy == "middle":
        mid = int(round((lung_z[0] + lung_z[-1]) / 2))
        cand = sorted((int(z) for z in lung_z), key=lambda z: abs(z - mid))[: max(n * 10, 20)]
    else:
        eligible = np.flatnonzero(area >= min_area_frac * area.max())
        cand = [int(z) for z in rng.permutation(eligible)]
    picked = []
    for z in cand:  # candidates are in preference order; take the first n that pass the QC
        if len(picked) == n:
            break
        if outside_body_frac(vol[:, :, z], lab[:, :, z]) <= max_outside:
            picked.append(z)
    for z in cand:  # fallback: keep the scan even if every candidate looks bad
        if len(picked) >= n:
            break
        if z not in picked:
            picked.append(z)
    return sorted(picked)


# --------------------------------------------------------------------------------------------------
# Writers: demo (NIfTI / DICOM / MetaImage) and low-res volumes
# --------------------------------------------------------------------------------------------------
def write_dicom_series(vol, spacing, origin, out_dir: Path, case_id: str, seriesuid: str, z0: int, z1: int):
    """Write slices z0..z1-1 of ``vol`` [x, y, z] (HU, LPS) as an uncompressed CT DICOM series.

    Stored value = HU + 1024 (int16), RescaleIntercept -1024, RescaleSlope 1, so readers recover HU.
    """
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    out_dir.mkdir(parents=True, exist_ok=True)
    study = generate_uid(entropy_srcs=[seriesuid, "study"])
    frame = generate_uid(entropy_srcs=[seriesuid, "frame"])
    series = seriesuid if len(seriesuid) <= 64 else generate_uid(entropy_srcs=[seriesuid, "series"])
    sx, sy, sz = spacing
    for n, k in enumerate(range(z0, z1)):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid(entropy_srcs=[seriesuid, f"sop-{k}"])
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.PatientName, ds.PatientID, ds.Modality = f"LUNA16^{case_id}", f"LUNA16^{case_id}", "CT"
        ds.StudyDescription, ds.SeriesDescription = "LUNA16 chest CT (course demo)", "AXIAL"
        ds.Manufacturer, ds.ImageType = "HI-Course", ["DERIVED", "SECONDARY", "AXIAL"]
        ds.InstanceNumber = n + 1
        ds.ImagePositionPatient = [float(origin[0]), float(origin[1]), float(origin[2] + k * sz)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing, ds.SliceThickness, ds.SpacingBetweenSlices = [sy, sx], sz, sz
        ds.Rows, ds.Columns = int(vol.shape[1]), int(vol.shape[0])
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = -1024, 1, "HU"
        ds.WindowCenter, ds.WindowWidth = -600, 1500
        stored = (vol[:, :, k].T.astype(np.int32) + 1024).astype(np.int16)  # rows = y, columns = x
        ds.PixelData = stored.tobytes()
        ds.save_as(out_dir / f"IM_{n + 1:04d}.dcm", enforce_file_format=True)


def write_demo(out: Path, case_id: str, seriesuid: str, vol, lab, spacing, origin, slab: int):
    """Full 3D NIfTI + a central slab as DICOM series and MetaImage (day-1 notebook)."""
    import SimpleITK as sitk

    aff = lps_affine(spacing, origin)
    save_nifti(vol, aff, out / "demo/volume" / f"{case_id}_image.nii")
    save_nifti(lab, aff, out / "demo/volume" / f"{case_id}_label.nii")
    lung_z = np.flatnonzero(((lab == 1) | (lab == 2)).any(axis=(0, 1)))
    mid = int(round((lung_z[0] + lung_z[-1]) / 2))
    z0 = int(np.clip(mid - slab // 2, 0, max(0, vol.shape[2] - slab)))
    z1 = min(vol.shape[2], z0 + slab)
    write_dicom_series(vol, spacing, origin, out / "demo/dicom" / case_id, case_id, seriesuid, z0, z1)
    itk = sitk.GetImageFromArray(np.ascontiguousarray(vol[:, :, z0:z1].transpose(2, 1, 0)))
    itk.SetSpacing(tuple(float(s) for s in spacing))
    itk.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2] + z0 * spacing[2])))
    (out / "demo/mhd").mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(itk, str(out / "demo/mhd" / f"{case_id}.mhd"))


def write_lowres(out: Path, case_id: str, vol, lab, spacing, origin, target: float = 2.5):
    """Resample to ``target`` mm isotropic (linear for the image, nearest for labels); return metadata dict."""
    import SimpleITK as sitk

    def to_itk(a):
        im = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)))
        im.SetSpacing(tuple(float(s) for s in spacing))
        im.SetOrigin(tuple(float(o) for o in origin))
        return im

    size = [int(np.ceil(n * s / target)) for n, s in zip(vol.shape, spacing)]
    org, direction, osp = tuple(float(o) for o in origin), tuple(np.eye(3).ravel().tolist()), (target,) * 3

    def resample(a, interp, default):
        return sitk.Resample(to_itk(a), size, sitk.Transform(), interp, org, osp, direction, default)

    im = resample(vol.astype(np.float32), sitk.sitkLinear, -1024.0)
    lb = resample(lab, sitk.sitkNearestNeighbor, 0)
    a_img = np.round(sitk.GetArrayFromImage(im)).astype(np.int16).transpose(2, 1, 0)
    a_lab = sitk.GetArrayFromImage(lb).astype(np.uint8).transpose(2, 1, 0)
    aff = lps_affine((target,) * 3, origin)
    save_nifti(np.ascontiguousarray(a_img), aff, out / "volumes_lowres/images" / f"{case_id}.nii")
    save_nifti(np.ascontiguousarray(a_lab), aff, out / "volumes_lowres/labels" / f"{case_id}.nii")
    return dict(size_x=size[0], size_y=size[1], size_z=size[2], spacing_x=target, spacing_y=target,
                spacing_z=target)


# --------------------------------------------------------------------------------------------------
# One scan -> one (or more) slices
# --------------------------------------------------------------------------------------------------
def process_scan(job: dict) -> dict:
    """Worker: process one scan.  Returns {"rows": [...], "vol_row": dict|None, "error": str|None, "seconds": float}."""
    t0 = time.time()
    out = Path(job["out"])
    try:
        vol, lab, spacing, origin, reoriented = read_scan(Path(job["mhd"]), Path(job["seg"]))
        rng = np.random.default_rng([job["seed"], job["index"]])  # independent of workers / subsets chosen
        zs = choose_slices(vol, lab, job["strategy"], job["n_slices_per_patient"], job["min_area_frac"], rng)
        if job["drop_bad_masks"] and any(outside_body_frac(vol[:, :, z], lab[:, :, z]) > 0.02 for z in zs):
            raise ValueError("failed automatic lung mask (labels outside the body), dropped by --drop-bad-masks")
        sx, sy, sz = spacing
        nz = vol.shape[2]
        rows = []
        for j, z in enumerate(zs):
            case = job["case_id"] if job["n_slices_per_patient"] == 1 else f"{job['case_id']}_s{j + 1:02d}"
            img2, lab2 = vol[:, :, z], lab[:, :, z]
            aff = lps_affine((sx, sy, sz), (origin[0], origin[1], origin[2] + z * sz))
            save_nifti(np.ascontiguousarray(img2), aff, out / "images" / f"{case}.nii")
            save_nifti(np.ascontiguousarray(lab2), aff, out / "labels" / f"{case}.nii")
            lung_px = int(((lab2 == 1) | (lab2 == 2)).sum())
            rows.append(dict(
                case_id=case, seriesuid=job["seriesuid"], subset=job["subset"], slice_index=int(z),
                n_slices=int(nz), z_rel=round(z / max(nz - 1, 1), 4), spacing_x=round(sx, 6),
                spacing_y=round(sy, 6), slice_thickness=round(sz, 6), size_x=int(img2.shape[0]),
                size_y=int(img2.shape[1]), right_lung_px=int((lab2 == 1).sum()),
                left_lung_px=int((lab2 == 2).sum()), trachea_px=int((lab2 == 3).sum()),
                lung_area_mm2=round(lung_px * sx * sy, 1), has_nodule=job["has_nodule"],
                padding_value_present=int(img2.min() <= PADDING_THRESHOLD),
                mask_outside_body=round(outside_body_frac(img2, lab2), 4),
                z_mm=round(float(origin[2] + z * sz), 3), reoriented=int(reoriented)))
        if job["demo"]:
            write_demo(out, job["case_id"], job["seriesuid"], vol, lab, spacing, origin, job["demo_slab"])
        vol_row = None
        if job["volume"]:
            vol_row = dict(case_id=job["case_id"], seriesuid=job["seriesuid"], subset=job["subset"],
                           **write_lowres(out, job["case_id"], vol, lab, spacing, origin))
        return dict(rows=rows, vol_row=vol_row, error=None, seconds=time.time() - t0)
    except Exception as exc:  # corrupted / unexpected scan: report and carry on
        msg = " ".join(str(exc).split())[:160]
        return dict(rows=[], vol_row=None, error=f"{type(exc).__name__}: {msg}", seconds=time.time() - t0,
                    trace=traceback.format_exc(limit=3))


def discover_scans(luna_dir: Path, subsets: list[int] | None):
    """Return a sorted list of (subset, seriesuid, mhd path) for the selected subsets found in ``luna_dir``."""
    found = []
    for s in range(10):
        d = luna_dir / f"subset{s}"
        if d.is_dir():
            found += [(s, p.stem, p) for p in d.glob("*.mhd")]
    found.sort(key=lambda t: (t[0], t[1]))
    present = sorted({s for s, _, _ in found})
    if subsets is not None:
        missing = sorted(set(subsets) - set(present))
        if missing:
            sys.exit(f"subsets {missing} not found in {luna_dir} (found: {present}); use --download")
    return found, present


def build(args) -> int:
    luna, out = Path(args.luna_dir), Path(args.out)
    seg_dir = luna / "seg-lungs-LUNA16"
    if not seg_dir.is_dir():
        sys.exit(f"{seg_dir} not found (run with --download, or extract seg-lungs-LUNA16.zip there)")
    scans, present = discover_scans(luna, args.subsets)
    if not scans:
        sys.exit(f"no subset*/ folders with .mhd files in {luna}")
    selected = [t for t in scans if args.subsets is None or t[0] in args.subsets]
    # --append: continue a dataset built from the previous subsets (case ids go on from the last one)
    old = old_vol = None
    if args.append and (out / "metadata.csv").exists():
        old = pd.read_csv(out / "metadata.csv")
        selected = [t for t in selected if t[1] not in set(old["seriesuid"])]
        if not selected:
            print("nothing to add: all these scans are already in the dataset")
            return 0
        if (out / "volumes_lowres/metadata.csv").exists():
            old_vol = pd.read_csv(out / "volumes_lowres/metadata.csv")
    offset = 0 if old is None else len(old)
    # case ids follow the sorted (subset, seriesuid) order -> stable ids
    numbered = selected if args.append else scans
    ids = {uid: f"case_{offset + i:04d}" for i, (_, uid, _) in enumerate(numbered, 1)}
    nodule_uids = set()
    ann = Path(args.annotations) if args.annotations else luna / "annotations.csv"
    if ann.exists():
        nodule_uids = set(pd.read_csv(ann)["seriesuid"])
    has_ann = ann.exists()
    if not has_ann:
        print(f"[warn] {ann} not found: has_nodule will be -1")
    if args.limit:
        selected = selected[: args.limit]
    demo_uid = None
    if args.demo_case:
        demo_uid = next((u for _, u, _ in scans if args.demo_case in (u, ids[u])), None)
        if demo_uid is None:
            sys.exit(f"--demo-case {args.demo_case} not found")
    elif not (out / "demo").exists():
        demo_uid = selected[0][1]
    n_volumes = max(0, args.volumes - (0 if old_vol is None else len(old_vol)))
    vol_uids = {u for _, u, _ in selected[:n_volumes]}
    jobs = []
    for idx, (s, uid, mhd) in enumerate(selected):
        jobs.append(dict(
            out=str(out), mhd=str(mhd), seg=str(seg_dir / f"{uid}.mhd"), seriesuid=uid, subset=s, index=offset + idx,
            case_id=ids[uid], seed=args.seed, strategy=args.strategy, min_area_frac=args.min_area_frac,
            n_slices_per_patient=args.slices_per_patient, demo=(uid == demo_uid), demo_slab=args.demo_slab,
            drop_bad_masks=args.drop_bad_masks,            volume=uid in vol_uids, has_nodule=(int(uid in nodule_uids) if has_ann else -1)))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} scans from subsets {sorted({j['subset'] for j in jobs})} -> {out} "
          f"(strategy={args.strategy}, workers={args.workers}, demo={ids.get(demo_uid, 'already there')}, "
          f"volumes={len(vol_uids)})")
    rows, vol_rows, errors, t0 = [], [], [], time.time()
    if args.workers > 1:
        with cf.ProcessPoolExecutor(args.workers) as ex:
            results = list(tqdm(ex.map(process_scan, jobs), total=len(jobs), desc="scans"))
    else:
        results = [process_scan(j) for j in tqdm(jobs, desc="scans")]
    for job, res in zip(jobs, results):
        if res["error"]:
            errors.append((job["seriesuid"], res["error"]))
            print(f"[skip] {job['case_id']} {job['seriesuid']}: {res['error']}")
        else:
            rows += res["rows"]
            if res["vol_row"]:
                vol_rows.append(res["vol_row"])
    wall = time.time() - t0
    if not rows:
        sys.exit("nothing was written")
    df = pd.DataFrame(rows)[METADATA_COLUMNS + EXTRA_COLUMNS]
    if old is not None:
        df = pd.concat([old, df], ignore_index=True)
    df = df.sort_values("case_id")
    df.to_csv(out / "metadata.csv", index=False)
    if vol_rows:
        vol_df = pd.DataFrame(vol_rows)[VOLUME_COLUMNS]
        if old_vol is not None:
            vol_df = pd.concat([old_vol, vol_df], ignore_index=True)
        vol_df.sort_values("case_id").to_csv(out / "volumes_lowres/metadata.csv", index=False)
    write_cards(out, args, len(df))
    (out / "build_info.json").write_text(json.dumps(dict(
        script="data_prep/prepare_luna16_slices.py", strategy=args.strategy, seed=args.seed,
        min_area_frac=args.min_area_frac, slices_per_patient=args.slices_per_patient,
        subsets=sorted(df["subset"].unique().tolist()), n_cases=len(df), skipped=errors), indent=2))
    print(f"done: {len(df)} cases written, {len(errors)} skipped, {wall:.0f} s wall "
          f"({wall / len(jobs):.1f} s per scan with {args.workers} worker(s))")
    return 0


def write_cards(out: Path, args, n: int):
    """Dataset card + licence/attribution files shipped inside the dataset."""
    (out / "README.md").write_text(f"""# luna16-lung-slices

One axial CT slice per patient from LUNA16 ({n} cases) with lung masks, prepared for a teaching
hands-on (MSc Health Informatics, lung segmentation on 2D chest CT).  **Not for clinical use.**

## Source and licence
* LUNA16 (LUng Nodule Analysis 2016), https://luna16.grand-challenge.org/ , Zenodo
  https://doi.org/10.5281/zenodo.3723295 and https://doi.org/10.5281/zenodo.4121926 .
  Licence **CC BY 4.0**; the CT images come from LIDC-IDRI (CC BY 3.0, TCIA).
* This dataset is a modified derivative (one slice per scan, cropped metadata, relabelled masks).
  Attribution is required, see `LICENSE.txt`.

## How it was derived
Each scan was reoriented to LPS, the lung mask of `seg-lungs-LUNA16` was remapped, and one slice was
chosen with strategy `{args.strategy}` (random slice among those with lung area >= {args.min_area_frac} of the
maximum, seed {args.seed}).  Intensities are untouched Hounsfield units (int16): padding values such
as -2000 or -3024 outside the field of view are kept on purpose.

## Layout
* `images/case_XXXX.nii`: 2D NIfTI, int16 HU.  `labels/case_XXXX.nii`: 2D NIfTI, uint8.
* Labels: **0 background, 1 right lung, 2 left lung, 3 trachea/main airways**.
* `metadata.csv`: `case_id, seriesuid, subset (0-9, official folds), slice_index, n_slices, z_rel,
  spacing_x, spacing_y, slice_thickness, size_x, size_y, right_lung_px, left_lung_px, trachea_px,
  lung_area_mm2, has_nodule` + `padding_value_present, z_mm, reoriented, mask_outside_body`
  (the last one is the fraction of lung-labelled pixels outside the body: >0.02 = failed automatic mask).
* `demo/`: one full scan as 3D NIfTI, a central slab as DICOM series and as MetaImage.
* `volumes_lowres/` (if present): 3D scans at 2.5 mm isotropic (image linear, labels nearest).
* Course split: subsets 0-6 train, 7-8 validation, 9 test.

## Orientation convention
Arrays are stored in LPS+ index order: axis 0 grows towards the patient's left, axis 1 towards
posterior.  `plt.imshow(img.T, cmap="gray")` shows the radiological view (right lung on the left of
the screen, spine at the bottom).  The NIfTI affine encodes this (RAS world, negative x/y diagonal).

## Caveats
* The lung masks come from an automatic algorithm (van Rikxoort et al., 2009) and are **not a
  reference standard** (LUNA16 organisers): silver-standard labels with local errors.
* LUNA16 only includes scans with slice thickness <= 2.5 mm, but spacing and field of view vary.

## Citation
{CITATION}
""")
    (out / "LICENSE.txt").write_text(f"""Dataset: luna16-lung-slices (derived from LUNA16)

Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

Original data: LUNA16 challenge data (https://luna16.grand-challenge.org/, Zenodo 3723295 and 4121926),
derived from the LIDC-IDRI collection (The Cancer Imaging Archive, CC BY 3.0).
Changes: one axial slice per scan selected, lung masks relabelled, files converted to NIfTI/DICOM,
optional down-sampled volumes.  The changes are not endorsed by the original authors.

Please cite:
{CITATION}
Armato et al. (2011), The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI),
Medical Physics 38(2), 915-931.  Clark et al. (2013), The Cancer Imaging Archive, J Digit Imaging 26(6), 1045-1057.

Not for clinical use.
""")


# --------------------------------------------------------------------------------------------------
# --download helper (Zenodo, parallel ranged requests with resume, md5 check, extract, delete zip)
# --------------------------------------------------------------------------------------------------
def zenodo_files() -> dict:
    """{filename: (url, size, md5)} for both LUNA16 Zenodo records."""
    files = {}
    for rec in ZENODO_RECORDS:
        for attempt in range(5):  # the Zenodo API is occasionally slow
            try:
                with urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=60) as r:
                    data = json.load(r)
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(3 * (attempt + 1))
        for f in data["files"]:
            files[f["key"]] = (f["links"]["self"], int(f["size"]), f.get("checksum", "").replace("md5:", ""))
    return files


def _fetch_range(url: str, dest: Path, start: int, end: int, retries: int = 20):
    """Download bytes [start, end] into ``dest`` (appending; resumes from the existing size)."""
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        if start + have > end:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except Exception:
            time.sleep(min(2 * (attempt + 1), 30))
    if dest.stat().st_size < end - start + 1:
        raise RuntimeError(f"could not complete {dest}")


def download_file(url: str, size: int, md5: str, dest: Path, connections: int = 16):
    """Parallel, resumable download.  Parts live next to the target as ``<name>.partN`` until complete."""
    if dest.exists() and dest.stat().st_size == size:
        return
    chunk = -(-size // connections)
    ranges = [(i, i * chunk, min(size, (i + 1) * chunk) - 1) for i in range(connections) if i * chunk < size]
    parts = [dest.with_name(f"{dest.name}.part{i}") for i, _, _ in ranges]
    with cf.ThreadPoolExecutor(len(ranges)) as ex:
        futs = [ex.submit(_fetch_range, url, p, s, e) for p, (_, s, e) in zip(parts, ranges)]
        t0 = time.time()
        while not all(f.done() for f in futs):
            got = sum(p.stat().st_size for p in parts if p.exists())
            print(f"\r  {dest.name}: {got / 1e9:6.2f}/{size / 1e9:.2f} GB  {got / 1e6 / max(time.time() - t0, 1):5.1f} MB/s",
                  end="", flush=True)
            time.sleep(2)
        for f in futs:
            f.result()
    print()
    with open(dest, "wb") as out_f:
        for p in parts:
            with open(p, "rb") as f:
                while chunk_b := f.read(1 << 24):
                    out_f.write(chunk_b)
            p.unlink()
    if md5:
        h = hashlib.md5()
        with open(dest, "rb") as f:
            while chunk_b := f.read(1 << 24):
                h.update(chunk_b)
        if h.hexdigest() != md5:
            dest.unlink()
            raise RuntimeError(f"md5 mismatch for {dest.name}: delete and retry")


def download(args):
    luna = Path(args.luna_dir)
    luna.mkdir(parents=True, exist_ok=True)
    files = zenodo_files()
    subsets = args.subsets if args.subsets is not None else list(range(10))
    wanted = ["annotations.csv", "seg-lungs-LUNA16.zip"] + [f"subset{s}.zip" for s in subsets]
    todo = []
    for name in wanted:
        done = (luna / name.replace(".zip", "")).is_dir() if name.endswith(".zip") else (luna / name).exists()
        todo.append((name, files[name], done))
    total = sum(f[1] for _, f, d in todo if not d)
    print("Files to fetch from Zenodo (licence CC BY 4.0):")
    for name, (_, size, _), done in todo:
        print(f"  {name:24s} {size / 1e9:7.2f} GB  {'(already extracted)' if done else ''}")
    print(f"Total {total / 1e9:.1f} GB; peak extra disk about {max((f[1] for _, f, _ in todo), default=0) * 2 / 1e9:.1f} GB "
          f"(zip + extracted copy of the largest file).")
    if args.dry_run:
        return
    for name, (url, size, md5), done in todo:
        if done:
            continue
        dest = luna / name
        download_file(url, size, md5, dest, args.connections)
        if name.endswith(".zip"):
            print(f"  extracting {name} ...")
            with zipfile.ZipFile(dest) as z:
                z.extractall(luna)
            if not args.keep_archives:
                dest.unlink()
    print(f"LUNA16 is ready in {luna}")


# --------------------------------------------------------------------------------------------------
# --check: validate an output folder (course dataset or synthetic dataset)
# --------------------------------------------------------------------------------------------------
def check_dataset(root: Path) -> bool:
    """Validate a dataset folder; print a summary and return True if every hard check passes."""
    root = Path(root)
    problems, warnings = [], []
    md_path = root / "metadata.csv"
    if not md_path.exists():
        print(f"FAIL: {md_path} missing")
        return False
    md = pd.read_csv(md_path)
    if list(md.columns[: len(METADATA_COLUMNS)]) != METADATA_COLUMNS:
        problems.append(f"metadata columns differ from the contract: {list(md.columns)}")
    img_files = sorted(p.name for p in (root / "images").glob("*.nii"))
    lab_files = sorted(p.name for p in (root / "labels").glob("*.nii"))
    expected = sorted(f"{c}.nii" for c in md["case_id"])
    if img_files != expected or lab_files != expected:
        problems.append(f"metadata rows ({len(md)}) != files (images {len(img_files)}, labels {len(lab_files)})")
    if md["case_id"].duplicated().any():
        problems.append("duplicated case_id")
    stats, bad_axcodes, bad_label, bad_shape, bad_hu, n_rl_ok, n_lr = [], [], [], [], [], 0, 0
    odd_hu = []
    for _, r in tqdm(list(md.iterrows()), desc="checking", leave=False):
        c = r["case_id"]
        try:
            im, lb = nib.load(root / "images" / f"{c}.nii"), nib.load(root / "labels" / f"{c}.nii")
            a, b = np.asarray(im.dataobj), np.asarray(lb.dataobj)
        except Exception as exc:
            problems.append(f"{c}: cannot load ({exc})")
            continue
        if a.shape != b.shape or a.ndim != 2 or a.shape != (r["size_x"], r["size_y"]):
            bad_shape.append(c)
        if a.dtype != np.int16 or b.dtype != np.uint8:
            problems.append(f"{c}: dtypes {a.dtype}/{b.dtype} (expected int16/uint8)")
        if not set(np.unique(b)).issubset({0, 1, 2, 3}):
            bad_label.append(c)
        if nib.aff2axcodes(im.affine) != ("L", "P", "S") or not np.allclose(im.affine, lb.affine):
            bad_axcodes.append(c)
        sp = np.abs(np.diag(im.affine)[:3])
        if not np.allclose(sp, [r["spacing_x"], r["spacing_y"], r["slice_thickness"]], atol=1e-3):
            warnings.append(f"{c}: affine spacing {sp} differs from metadata")
        extreme = (a < -4000) | (a > 5000)
        if extreme.mean() > 0.01:       # a large part of the image is not in Hounsfield units
            bad_hu.append(c)
        elif extreme.any():             # a few pixels: metal or reconstruction artefacts, clipped by the transforms
            odd_hu.append(c)
        lung = (b == 1) | (b == 2)
        if (b == 1).any() and (b == 2).any():
            n_lr += 1
            n_rl_ok += np.argwhere(b == 1)[:, 0].mean() < np.argwhere(b == 2)[:, 0].mean()
        for col, val in (("right_lung_px", (b == 1).sum()), ("left_lung_px", (b == 2).sum()),
                         ("trachea_px", (b == 3).sum())):
            if int(r[col]) != int(val):
                problems.append(f"{c}: {col} in metadata ({r[col]}) != label ({val})")
        mean_lung = float(a[lung].mean()) if lung.any() else np.nan
        stats.append(dict(case_id=c, hu_min=int(a.min()), hu_max=int(a.max()), lung_mean_hu=mean_lung,
                          pad=int(a.min() <= PADDING_THRESHOLD), lung_px=int(lung.sum())))
    for name, lst in (("shape mismatch / not 2D", bad_shape), ("labels outside {0,1,2,3}", bad_label),
                      ("affine axcodes != ('L','P','S')", bad_axcodes), ("implausible HU range", bad_hu)):
        if lst:
            problems.append(f"{name}: {len(lst)} case(s), e.g. {lst[:3]}")
    if odd_hu:
        warnings.append(f"{len(odd_hu)} case(s) with a few pixels of extreme value (artefacts): {odd_hu[:6]}")
    frac = n_rl_ok / n_lr if n_lr else 0.0
    if frac <= 0.95:
        problems.append(f"right lung centroid x < left lung centroid x only in {frac:.1%} of cases (need > 95%)")
    st = pd.DataFrame(stats)
    if len(st):
        implausible = ((st["lung_mean_hu"] > -300) | (st["lung_mean_hu"] < -1000)).sum()
        if implausible > 0.05 * len(st):
            problems.append(f"mean HU inside the lungs implausible in {implausible} cases")
        if (st["lung_px"] == 0).any():
            warnings.append(f"{(st['lung_px'] == 0).sum()} case(s) with an empty lung mask")
    problems += _check_extras(root, md, warnings)
    # ---- summary ----
    print(f"\n=== {root} : {len(md)} cases ===")
    per = md.groupby("subset").size().to_dict()
    print("cases per subset :", per)
    for col in ("spacing_x", "spacing_y", "slice_thickness"):
        print(f"{col:16s}: min {md[col].min():.3f}  median {md[col].median():.3f}  max {md[col].max():.3f}")
    print(f"image sizes      : {md.groupby(['size_x', 'size_y']).size().to_dict()}")
    if len(st):
        print(f"HU range         : {st['hu_min'].min()} .. {st['hu_max'].max()}   mean HU in lungs "
              f"{st['lung_mean_hu'].mean():.0f}")
        print(f"padding value    : {st['pad'].mean():.1%} of cases have HU <= {PADDING_THRESHOLD} (padding)")
    print(f"right x < left x : {frac:.1%} of {n_lr} cases with both lungs")
    if "has_nodule" in md:
        print(f"has_nodule       : {md['has_nodule'].value_counts().to_dict()}")
    if "mask_outside_body" in md and (md["mask_outside_body"] > 0.02).any():
        bad = md.loc[md["mask_outside_body"] > 0.02, "case_id"].tolist()
        warnings.append(f"{len(bad)} case(s) with lung labels outside the body (failed silver-standard mask): {bad[:5]}")
    for w in warnings:
        print("WARN:", w)
    for p in problems:
        print("FAIL:", p)
    print("RESULT:", "OK" if not problems else f"{len(problems)} problem(s)")
    return not problems


def _check_extras(root: Path, md: pd.DataFrame, warnings: list) -> list[str]:
    """Check demo/ and volumes_lowres/ when present."""
    problems = []
    vols = sorted((root / "demo/volume").glob("*_image.nii"))
    if not vols:
        warnings.append("demo/volume missing")
    for v in vols:
        case = v.name.replace("_image.nii", "")
        try:
            im, lb = nib.load(v), nib.load(root / "demo/volume" / f"{case}_label.nii")
            if im.shape != lb.shape or im.ndim != 3 or nib.aff2axcodes(im.affine) != ("L", "P", "S"):
                problems.append(f"demo volume {case}: bad shape/axcodes")
            for d in (root / "demo/dicom" / case, ):
                problems += _check_dicom(d, im)
            mhd = root / "demo/mhd" / f"{case}.mhd"
            if not mhd.exists():
                problems.append(f"missing {mhd}")
            else:
                import SimpleITK as sitk
                m = sitk.ReadImage(str(mhd))
                if not np.allclose(m.GetSpacing(), np.abs(np.diag(im.affine)[:3]), atol=1e-3):
                    problems.append(f"{mhd.name}: spacing differs from demo volume")
        except Exception as exc:
            problems.append(f"demo {case}: {exc}")
    lr = root / "volumes_lowres"
    if lr.exists():
        vm = pd.read_csv(lr / "metadata.csv")
        if list(vm.columns) != VOLUME_COLUMNS:
            problems.append(f"volumes_lowres/metadata.csv columns: {list(vm.columns)}")
        for _, r in vm.iterrows():
            try:
                im = nib.load(lr / "images" / f"{r['case_id']}.nii")
                lb = nib.load(lr / "labels" / f"{r['case_id']}.nii")
                ok = im.shape == lb.shape == (r["size_x"], r["size_y"], r["size_z"])
                ok &= nib.aff2axcodes(im.affine) == ("L", "P", "S")
                ok &= im.get_data_dtype() == np.int16 and lb.get_data_dtype() == np.uint8
                ok &= set(np.unique(np.asarray(lb.dataobj))).issubset({0, 1, 2, 3})
                if not ok:
                    problems.append(f"volumes_lowres/{r['case_id']}: shape/axcodes/dtype/labels")
            except Exception as exc:
                problems.append(f"volumes_lowres/{r['case_id']}: {exc}")
        print(f"volumes_lowres   : {len(vm)} volumes, e.g. size {tuple(int(v) for v in vm.iloc[0][['size_x', 'size_y', 'size_z']])}")
    return problems


def _check_dicom(d: Path, nifti) -> list[str]:
    """The DICOM series must be readable by pydicom and SimpleITK with HU and geometry equal to the NIfTI."""
    import pydicom
    import SimpleITK as sitk

    files = sorted(d.glob("*.dcm"))
    if not files:
        return [f"no DICOM files in {d}"]
    ds = pydicom.dcmread(files[0])
    reader = sitk.ImageSeriesReader()
    names = reader.GetGDCMSeriesFileNames(str(d))
    reader.SetFileNames(names)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    ref = np.asarray(nifti.dataobj)
    spacing_ok = np.allclose(img.GetSpacing(), np.abs(np.diag(nifti.affine)[:3]), atol=1e-3)
    # locate the slab inside the full volume through the z origin (LPS) of the first slice
    z0 = int(round((img.GetOrigin()[2] - (nifti.affine[2, 3])) / abs(nifti.affine[2, 2])))
    hu_ok = z0 >= 0 and z0 + arr.shape[2] <= ref.shape[2] and np.array_equal(arr, ref[:, :, z0:z0 + arr.shape[2]])
    out = []
    if not (spacing_ok and hu_ok):
        out.append(f"{d.name}: DICOM series differs from the demo volume (spacing_ok={spacing_ok}, HU_ok={hu_ok})")
    if getattr(ds, "RescaleIntercept", None) != -1024:
        out.append(f"{d.name}: RescaleIntercept != -1024")
    return out


# --------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--luna-dir", type=Path, help="folder with subset0..9/ and seg-lungs-LUNA16/ (and annotations.csv)")
    ap.add_argument("--out", type=Path, help="output dataset folder (or the folder to --check)")
    ap.add_argument("--subsets", type=int, nargs="+", help="LUNA16 subsets to process (default: all found)")
    ap.add_argument("--strategy", choices=["random", "max_area", "middle"], default="random")
    ap.add_argument("--min-area-frac", type=float, default=0.3,
                    help="random strategy: eligible slices have lung area >= this fraction of the scan maximum")
    ap.add_argument("--slices-per-patient", type=int, default=1,
                    help="default 1 (the course contract). More gives case_XXXX_sNN ids: NOT leakage-safe across splits by case_id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo-case", help="case_id or seriesuid of the demo scan (default: first scan processed)")
    ap.add_argument("--demo-slab", type=int, default=40, help="central slices kept for the DICOM/mhd demo")
    ap.add_argument("--volumes", type=int, default=0, help="also write N low-res (2.5 mm) 3D volumes")
    ap.add_argument("--annotations", type=Path, help="annotations.csv (default: <luna-dir>/annotations.csv)")
    ap.add_argument("--drop-bad-masks", action="store_true",
                    help="skip scans whose chosen slice has lung labels outside the body (failed automatic mask)")
    ap.add_argument("--append", action="store_true",
                    help="add the scans to an existing dataset in --out, continuing the case numbers. "
                         "Lets you build the dataset one subset at a time (process the subsets in order)")
    ap.add_argument("--limit", type=int, help="process only the first N scans (testing)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2))))
    ap.add_argument("--download", action="store_true", help="fetch the needed zips from Zenodo into --luna-dir and extract")
    ap.add_argument("--dry-run", action="store_true", help="with --download: only print what would be fetched")
    ap.add_argument("--connections", type=int, default=16, help="parallel connections per download")
    ap.add_argument("--keep-archives", action="store_true", help="do not delete the zips after extraction")
    ap.add_argument("--check", action="store_true", help="validate the dataset in --out and exit")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.check:
        if not args.out:
            sys.exit("--check needs --out <dataset folder>")
        return 0 if check_dataset(args.out) else 1
    if args.download:
        if not args.luna_dir:
            sys.exit("--download needs --luna-dir")
        download(args)
        if not args.out:
            return 0
    if not args.luna_dir or not args.out:
        sys.exit("need --luna-dir and --out (see --help)")
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
