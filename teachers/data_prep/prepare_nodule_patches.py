#!/usr/bin/env python3
"""Build the nodule patches for notebook 04 from LUNA16.

For every nodule listed in LUNA16 ``annotations.csv`` (centre and diameter) the script cuts a small
2D patch from the axial slice through the centre of the nodule, and draws the mask as a disc with
the annotated diameter.  The nodule is NOT in the middle of the patch: it is shifted at random, so
that a model cannot learn "the nodule is always at the centre".

    python prepare_nodule_patches.py --luna-dir /data/luna16 --out luna16-lung-slices

Run it after ``prepare_luna16_slices.py``, with the same ``--out``: it adds the folder ``nodules/``
to the course dataset and uses its ``metadata.csv`` to give each patch the ``case_id`` of the patient.

Output::

    nodules/images/nodule_0001.nii.gz   2D NIfTI, int16, Hounsfield units, 64 x 64 pixels
    nodules/labels/nodule_0001.nii.gz   2D NIfTI, uint8: 0 background, 1 nodule (a disc)
    nodules/metadata.csv                nodule_id, case_id, seriesuid, subset, diameter_mm, spacing_x, spacing_y

Important: the masks are discs, not the real outline of the nodules.  LUNA16 does not provide
outlines.  Radiologist outlines exist in LIDC-IDRI; if you have them, save them in the same layout
and the notebook works unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

PATCH = 64        # patch size in pixels
MAX_SHIFT = 14    # maximum random shift of the nodule from the centre of the patch, in pixels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--luna-dir", type=Path, required=True, help="folder with subset0..9/ and annotations.csv")
    ap.add_argument("--out", type=Path, required=True, help="course dataset folder (contains metadata.csv)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    annotations = pd.read_csv(args.luna_dir / "annotations.csv")
    cases = pd.read_csv(args.out / "metadata.csv").set_index("seriesuid")
    scans = {p.stem: p for p in args.luna_dir.glob("subset*/*.mhd")}
    out = args.out / "nodules"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    xx, yy = np.meshgrid(np.arange(PATCH), np.arange(PATCH), indexing="ij")
    rows = []
    for uid, group in annotations.groupby("seriesuid", sort=True):
        if uid not in scans or uid not in cases.index:
            continue  # scan not downloaded, or not in the course dataset
        scan = sitk.DICOMOrient(sitk.ReadImage(str(scans[uid])), "LPS")
        volume = sitk.GetArrayFromImage(scan)            # [z, y, x]
        sx, sy, sz = scan.GetSpacing()
        for a in group.itertuples():
            ix, iy, iz = scan.TransformPhysicalPointToContinuousIndex((a.coordX, a.coordY, a.coordZ))
            k = int(round(iz))
            if not 0 <= k < volume.shape[0]:
                continue
            # position of the nodule inside the patch
            px = PATCH / 2 + rng.integers(-MAX_SHIFT, MAX_SHIFT + 1)
            py = PATCH / 2 + rng.integers(-MAX_SHIFT, MAX_SHIFT + 1)
            x0, y0 = int(round(ix - px)), int(round(iy - py))
            padded = np.pad(volume[k], PATCH, constant_values=-1000)   # so that patches at the border work
            patch = padded[y0 + PATCH:y0 + 2 * PATCH, x0 + PATCH:x0 + 2 * PATCH].T   # [x, y]
            cx, cy = ix - x0, iy - y0
            radius = a.diameter_mm / 2
            mask = (((xx - cx) * sx) ** 2 + ((yy - cy) * sy) ** 2 <= radius ** 2).astype(np.uint8)
            if mask.sum() == 0:
                mask[int(round(cx)), int(round(cy))] = 1

            nodule_id = f"nodule_{len(rows) + 1:04d}"
            affine = np.diag([-sx, -sy, sz, 1.0])
            nib.save(nib.Nifti1Image(patch.astype(np.int16), affine), out / "images" / f"{nodule_id}.nii.gz")
            nib.save(nib.Nifti1Image(mask, affine), out / "labels" / f"{nodule_id}.nii.gz")
            rows.append(dict(nodule_id=nodule_id, case_id=cases.loc[uid, "case_id"], seriesuid=uid,
                             subset=int(cases.loc[uid, "subset"]), diameter_mm=round(a.diameter_mm, 2),
                             spacing_x=round(sx, 4), spacing_y=round(sy, 4)))
    meta = pd.DataFrame(rows)
    meta.to_csv(out / "metadata.csv", index=False)
    print(f"{len(meta)} nodules from {meta.case_id.nunique()} patients -> {out}")
    print(meta.diameter_mm.describe().round(1).to_string())


if __name__ == "__main__":
    main()
