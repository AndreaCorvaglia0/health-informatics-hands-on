#!/usr/bin/env python3
"""Build the nodule patches for notebook 04 from LUNA16.

For every nodule listed in LUNA16 ``annotations.csv`` (centre and diameter) the script cuts a small
2D patch from the axial slice closest to the centre of the nodule.  The nodule is NOT in the middle
of the patch: it is shifted at random, so that a model cannot learn "the nodule is always at the centre".

The mask of the patch comes from one of two sources:

* ``--masks-dir`` given (what the course uses): the outlines drawn by the LIDC-IDRI radiologists, one
  3D mask per scan (``<seriesuid>.nii.gz``, same grid as the LUNA16 scan).  The mask of the patch is
  the same cut of that 3D mask, so it holds every nodule pixel of the patch, also of a second nodule
  that happens to be near.  Nodules without a mask on their slice are left out and listed in
  ``nodules/excluded.csv``.
* no ``--masks-dir``: a disc.  The nodule is treated as a sphere of the annotated diameter and the disc
  is the section of that sphere at the position of the slice.

    python prepare_nodule_patches.py --luna-dir /data/luna16 --out luna16-lung-slices --masks-dir /data/masks

Run it after ``prepare_luna16_slices.py``, with the same ``--out``: it adds the folder ``nodules/``
to the course dataset and uses its ``metadata.csv`` to give each patch the ``case_id`` of the patient.
It can be run again after each new subset: nodules already written are kept, new ones are added.

``--relabel`` replaces the masks of patches that already exist, without the raw scans: the position of
every patch is computed again from the grid of the 3D masks (same code, same random numbers), the image
patches are left as they are.  It needs ``annotations.csv`` in ``--luna-dir`` and ``--masks-dir``.

Output::

    nodules/images/nodule_0001.nii.gz   2D NIfTI, int16, Hounsfield units, 64 x 64 pixels
    nodules/labels/nodule_0001.nii.gz   2D NIfTI, uint8: 0 background, 1 nodule
    nodules/metadata.csv                nodule_id, case_id, seriesuid, subset, diameter_mm (LUNA16 annotation),
                                        spacing_x, spacing_y, slice_thickness, mask_source, mask_area_mm2,
                                        mask_diameter_mm (circle with the area of the object at the annotated
                                        centre), n_objects (separate objects in the mask of the patch)
    nodules/excluded.csv                nodules left out, with the reason
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

PATCH = 64        # patch size in pixels
MAX_SHIFT = 14    # maximum random shift of the nodule from the centre of the patch, in pixels


def patch_positions(image, annotations, rng):
    """Where the patch of each nodule is.  ``image`` is the scan or its 3D mask (same grid), oriented LPS.

    Yields (annotation, slice index, x0, y0, centre x in the patch, centre y in the patch, distance of the
    slice from the centre of the nodule in mm).  The random numbers are drawn in the order of the annotations.
    """
    n_slices = image.GetSize()[2]
    sz = image.GetSpacing()[2]
    for a in annotations.itertuples():
        ix, iy, iz = image.TransformPhysicalPointToContinuousIndex((a.coordX, a.coordY, a.coordZ))
        k = int(round(iz))
        if not 0 <= k < n_slices:
            continue
        px = PATCH / 2 + rng.integers(-MAX_SHIFT, MAX_SHIFT + 1)   # position of the nodule inside the patch
        py = PATCH / 2 + rng.integers(-MAX_SHIFT, MAX_SHIFT + 1)
        x0, y0 = int(round(ix - px)), int(round(iy - py))
        yield a, k, x0, y0, ix - x0, iy - y0, (iz - k) * sz


def cut(slice_2d, x0, y0, fill):
    """The 64 x 64 patch that starts at (x0, y0), as an array [x, y]; outside the slice it is ``fill``."""
    padded = np.pad(slice_2d, PATCH, constant_values=fill)
    return padded[y0 + PATCH:y0 + 2 * PATCH, x0 + PATCH:x0 + 2 * PATCH].T


def disc(cx, cy, sx, sy, diameter_mm, dz_mm):
    """The nodule as a sphere: its section at the distance dz_mm from the centre."""
    xx, yy = np.meshgrid(np.arange(PATCH), np.arange(PATCH), indexing="ij")
    radius = np.sqrt(max((diameter_mm / 2) ** 2 - dz_mm ** 2, 0.0))
    mask = (((xx - cx) * sx) ** 2 + ((yy - cy) * sy) ** 2 <= radius ** 2).astype(np.uint8)
    if mask.sum() == 0:
        mask[int(round(cx)), int(round(cy))] = 1
    return mask


def describe(mask, cx, cy, sx, sy):
    """Area of the mask, diameter of the object at the annotated centre, number of objects."""
    objects, n = ndimage.label(mask)
    if n == 0:
        return 0.0, 0.0, 0
    centres = ndimage.center_of_mass(mask, objects, range(1, n + 1))
    nearest = 1 + int(np.argmin([(x - cx) ** 2 + (y - cy) ** 2 for x, y in centres]))
    here = objects[int(round(np.clip(cx, 0, PATCH - 1))), int(round(np.clip(cy, 0, PATCH - 1)))]
    area_object = (objects == (here or nearest)).sum() * sx * sy
    return round(float(mask.sum() * sx * sy), 1), round(float(2 * np.sqrt(area_object / np.pi)), 2), n


def write_readme_section(readme, meta, n_excluded):
    """Write (or replace) the part of the dataset README that describes the nodule patches."""
    if (meta.mask_source == "radiologists").all():
        masks = ("`nodules/labels` contains the outline drawn by the radiologists of LIDC-IDRI (0 background, 1 nodule): "
                 "the same cut of the 3D mask of the scan, so a second nodule near the first one is in the mask too. "
                 f"{n_excluded} nodules without a mask on their slice were left out (`nodules/excluded.csv`).")
    else:
        masks = ("`nodules/labels` contains a disc (0 background, 1 nodule): the section of a sphere of the annotated "
                 "diameter at the position of the slice. These are not real outlines.")
    section = (f"## Nodule patches (`nodules/`)\n\n{len(meta)} patches of 64 x 64 pixels from {meta.case_id.nunique()} patients, "
               "one per nodule of the LUNA16 `annotations.csv`, cut from the axial slice closest to the centre of the nodule; "
               f"the nodule is not in the middle of the patch. {masks} `nodules/metadata.csv` links each patch to its patient "
               "(`case_id`) and gives the annotated diameter and the area of the mask.\n")
    text = readme.read_text() if readme.exists() else ""
    start = text.find("## Nodule patches")
    if start >= 0:
        end = text.find("\n## ", start + 5)
        text = text[:start] + section + (text[end + 1:] if end >= 0 else "")
    else:
        text = text.rstrip("\n") + "\n\n" + section
    readme.write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--luna-dir", type=Path, required=True, help="folder with subset0..9/ and annotations.csv")
    ap.add_argument("--out", type=Path, required=True, help="course dataset folder (contains metadata.csv)")
    ap.add_argument("--masks-dir", type=Path, help="folder with the 3D masks of the radiologists, <seriesuid>.nii.gz")
    ap.add_argument("--relabel", action="store_true", help="only replace the masks of the patches that exist")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.relabel and not args.masks_dir:
        ap.error("--relabel needs --masks-dir")

    annotations = pd.read_csv(args.luna_dir / "annotations.csv")
    cases = pd.read_csv(args.out / "metadata.csv").set_index("seriesuid")
    scans = {p.stem: p for p in args.luna_dir.glob("subset*/*.mhd")}
    out = args.out / "nodules"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(out / "metadata.csv") if (out / "metadata.csv").exists() else pd.DataFrame(columns=["seriesuid"])
    old_excluded = pd.read_csv(out / "excluded.csv") if (out / "excluded.csv").exists() else pd.DataFrame(columns=["seriesuid"])
    if args.relabel:
        rows, excluded = [], old_excluded.to_dict("records")
        todo = list(dict.fromkeys(old.seriesuid))                      # the patients that have patches, in order
    else:
        rows, excluded = old.to_dict("records"), old_excluded.to_dict("records")
        done = set(old.seriesuid) | set(old_excluded.seriesuid)
        todo = [uid for uid in sorted(annotations.seriesuid.unique()) if uid in scans and uid in cases.index and uid not in done]
    number = len(old) + len(old_excluded)                              # last nodule number given so far

    for uid in todo:
        group = annotations[annotations.seriesuid == uid]
        case_id = cases.loc[uid, "case_id"]
        # one random generator per patient: the result does not depend on which subsets are on disk
        rng = np.random.default_rng([args.seed, int(case_id.split("_")[1])])
        mask_3d = None
        mask_file = args.masks_dir / f"{uid}.nii.gz" if args.masks_dir else None
        if mask_file is not None and mask_file.exists():
            mask_image = sitk.DICOMOrient(sitk.ReadImage(str(mask_file)), "LPS")
            mask_3d = sitk.GetArrayFromImage(mask_image)               # [z, y, x]
        if args.relabel:
            ids = list(old[old.seriesuid == uid].nodule_id)            # patches of this patient, in the same order
            if mask_3d is None:
                excluded += [dict(nodule_id=i, case_id=case_id, seriesuid=uid, reason="no mask for this scan") for i in ids]
                continue
            reference = mask_image
        else:
            reference = sitk.DICOMOrient(sitk.ReadImage(str(scans[uid])), "LPS")
            volume = sitk.GetArrayFromImage(reference)
            if mask_3d is not None and mask_3d.shape != volume.shape:
                raise SystemExit(f"{uid}: the mask and the scan have different sizes")
        sx, sy, sz = reference.GetSpacing()
        affine = np.diag([-sx, -sy, sz, 1.0])

        for n, (a, k, x0, y0, cx, cy, dz) in enumerate(patch_positions(reference, group, rng)):
            if args.relabel:
                nodule_id = ids[n]
            else:
                number += 1
                nodule_id = f"nodule_{number:04d}"
            if args.masks_dir:
                mask = cut(mask_3d[k], x0, y0, 0).astype(np.uint8) if mask_3d is not None else np.zeros((PATCH, PATCH), np.uint8)
                mask = (mask > 0).astype(np.uint8)
                source = "radiologists"
                if mask.sum() == 0:
                    reason = "no mask for this scan" if mask_3d is None else "the mask is empty on this slice"
                    excluded.append(dict(nodule_id=nodule_id, case_id=case_id, seriesuid=uid, reason=reason))
                    continue
            else:
                mask, source = disc(cx, cy, sx, sy, a.diameter_mm, dz), "disc"
            if not args.relabel:
                patch = cut(volume[k], x0, y0, -1000)
                nib.save(nib.Nifti1Image(patch.astype(np.int16), affine), out / "images" / f"{nodule_id}.nii.gz")
            nib.save(nib.Nifti1Image(mask, affine), out / "labels" / f"{nodule_id}.nii.gz")
            area, diameter, n_objects = describe(mask, cx, cy, sx, sy)
            rows.append(dict(nodule_id=nodule_id, case_id=case_id, seriesuid=uid, subset=int(cases.loc[uid, "subset"]),
                             diameter_mm=round(a.diameter_mm, 2), spacing_x=round(sx, 4), spacing_y=round(sy, 4),
                             slice_thickness=round(sz, 4), mask_source=source, mask_area_mm2=area,
                             mask_diameter_mm=diameter, n_objects=n_objects))

    meta = pd.DataFrame(rows)
    for e in excluded:                                                 # a nodule left out has no files
        for folder in ("images", "labels"):
            (out / folder / f"{e['nodule_id']}.nii.gz").unlink(missing_ok=True)
    meta.to_csv(out / "metadata.csv", index=False)
    pd.DataFrame(excluded, columns=["nodule_id", "case_id", "seriesuid", "reason"]).to_csv(out / "excluded.csv", index=False)
    write_readme_section(args.out / "README.md", meta, len(excluded))
    print(f"{len(meta)} nodules from {meta.case_id.nunique()} patients -> {out}  ({len(excluded)} left out)")
    print(meta.mask_source.value_counts().to_string())
    print(meta.diameter_mm.describe().round(1).to_string())


if __name__ == "__main__":
    main()
