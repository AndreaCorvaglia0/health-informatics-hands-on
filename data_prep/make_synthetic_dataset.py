#!/usr/bin/env python3
"""Generate a SYNTHETIC dataset with exactly the same layout as the course dataset.

It contains fake "CT slices" (ellipses + noise) - useless for learning anatomy, but
perfect for testing the notebooks and the web app on a laptop, with no download and
no GPU.  Teachers: use it to dry-run the material; CI can use it as well.

    python data_prep/make_synthetic_dataset.py --out data/synthetic --n 60

Layout produced (identical to ``prepare_luna16_slices.py``, see data_prep/README.md)::

    <out>/images/case_0001.nii.gz      2D int16, Hounsfield units, LPS+ orientation
    <out>/labels/case_0001.nii.gz      2D uint8: 0 background, 1 right lung, 2 left lung, 3 trachea
    <out>/metadata.csv
    <out>/demo/volume/case_0001_image.nii.gz, case_0001_label.nii.gz   (3D)
    <out>/demo/dicom/case_0001/IM_0001.dcm ...                          (DICOM series)
    <out>/demo/mhd/case_0001.mhd + .raw                                 (MetaImage, LUNA16 native)
    <out>/volumes_lowres/{images,labels}/case_0001.nii.gz, metadata.csv (tiny 3D volumes, --volumes N)

The generator is deterministic and only adds/overwrites files it owns: it never deletes anything
in ``--out`` (e.g. a ``pretrained/`` folder survives), and the same ``--seed`` gives identical
pixel data.  The low-res volumes use their own random stream, so ``--volumes`` never changes
the 2D slices.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def ellipse(xx, yy, cx, cy, rx, ry):
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def fake_slice(rng, size=512, z_rel=0.5):
    """Return (image HU int16, label uint8) as arrays indexed [x, y] in LPS+ orientation.

    LPS+: x grows towards the patient's Left, y grows towards Posterior.
    So the RIGHT lung sits at LOW x and the spine at HIGH y.
    """
    xx, yy = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    s = size / 512.0
    img = np.full((size, size), -1000.0)
    cx, cy = size / 2 + rng.normal(0, 6 * s), size / 2 + rng.normal(0, 6 * s)
    body = ellipse(xx, yy, cx, cy, rng.uniform(190, 215) * s, rng.uniform(135, 155) * s)
    img[body] = rng.normal(40, 12, size=body.sum())
    # lungs shrink towards apex (z_rel -> 1) and base (z_rel -> 0)
    k = 0.45 + 0.55 * np.sin(np.pi * np.clip(z_rel, 0.05, 0.95))
    rx, ry = rng.uniform(62, 78) * s * k, rng.uniform(88, 104) * s * k
    right = ellipse(xx, yy, cx - 95 * s, cy - 5 * s, rx, ry) & body
    left = ellipse(xx, yy, cx + 95 * s, cy - 5 * s, rx * rng.uniform(0.85, 1.0), ry) & body
    heart = ellipse(xx, yy, cx + 35 * s, cy - 30 * s, 55 * s, 45 * s)
    left &= ~heart
    trachea = ellipse(xx, yy, cx, cy - 55 * s, 9 * s, 9 * s) & body
    spine = ellipse(xx, yy, cx, cy + 95 * s, 22 * s, 20 * s) & body
    for mask, mean, sd in ((right, -850, 45), (left, -850, 45), (trachea, -985, 10), (spine, 450, 80)):
        img[mask] = rng.normal(mean, sd, size=mask.sum())
    # a few "vessels" inside the lungs
    for lung in (right, left):
        idx = np.argwhere(lung)
        for _ in range(12):
            px, py = idx[rng.integers(len(idx))]
            v = ellipse(xx, yy, px, py, rng.uniform(2, 5) * s, rng.uniform(2, 5) * s) & lung
            img[v] = rng.normal(30, 15, size=v.sum())
    # scanner table + padding value outside the field of view, like many LUNA16 scans
    table = (np.abs(yy - (cy + 175 * s)) < 4 * s) & (np.abs(xx - cx) < 230 * s)
    img[table] = 200
    fov = ellipse(xx, yy, size / 2, size / 2, size / 2, size / 2)
    if rng.random() < 0.4:
        img[~fov] = -3024 if rng.random() < 0.5 else -2000
    label = np.zeros((size, size), np.uint8)
    label[right], label[left], label[trachea] = 1, 2, 3
    return np.round(img).astype(np.int16), label


def lps_affine(spacing, origin=(0.0, 0.0, 0.0)):
    """NIfTI affine (RAS+ world) for an array stored in LPS+ index order."""
    sx, sy, sz = spacing
    aff = np.diag([-sx, -sy, sz, 1.0])
    aff[:3, 3] = (-origin[0], -origin[1], origin[2])
    return aff


def save_nifti(arr, affine, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr, affine)
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))


def fake_volume(rng, size, n_slices, z_lo=0.1, z_hi=0.9):
    """Stack of fake slices with the same anatomy on every slice (only z changes): [x, y, z]."""
    state = rng.bit_generator.state
    vol, lab = [], []
    for z in np.linspace(z_lo, z_hi, n_slices):
        rng.bit_generator.state = state
        i, l = fake_slice(rng, size, z)
        vol.append(i)
        lab.append(l)
    return np.stack(vol, -1), np.stack(lab, -1)


def write_volumes(out: Path, n: int, seed: int, shape=(48, 48, 32)):
    """Tiny 3D volumes for the optional 3D side track (``volumes_lowres/``)."""
    if n <= 0:
        return
    rng = np.random.default_rng(seed + 1000)  # own stream: never perturbs the 2D slices
    sx, sy, sz = shape
    spacing = (384.0 / sx,) * 3  # ~8 mm: tiny on purpose (real data: ~2.5 mm, ~140x140x120)
    rows = []
    for i in range(1, n + 1):
        case = f"case_{i:04d}"
        vol, lab = fake_volume(rng, sx, sz)
        aff = lps_affine(spacing)
        save_nifti(vol, aff, out / "volumes_lowres/images" / f"{case}.nii.gz")
        save_nifti(lab, aff, out / "volumes_lowres/labels" / f"{case}.nii.gz")
        rows.append(dict(case_id=case, seriesuid=f"synthetic.{i}", subset=(i - 1) % 10,
                         size_x=sx, size_y=sy, size_z=sz,
                         spacing_x=spacing[0], spacing_y=spacing[1], spacing_z=spacing[2]))
    pd.DataFrame(rows).to_csv(out / "volumes_lowres/metadata.csv", index=False)


def write_demo(out: Path, rng, n_slices=24, size=128):
    """Small 3D volume in the three formats used on day 1."""
    import pydicom
    import SimpleITK as sitk
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    spacing = (2.8, 2.8, 5.0)
    vol, lab = fake_volume(rng, size, n_slices)  # [x, y, z]
    origin = (-180.0, -180.0, -300.0)
    aff = lps_affine(spacing, origin)
    save_nifti(vol, aff, out / "demo/volume/case_0001_image.nii.gz")
    save_nifti(lab, aff, out / "demo/volume/case_0001_label.nii.gz")

    # MetaImage (.mhd + .raw), the native LUNA16 format
    mhd_dir = out / "demo/mhd"
    mhd_dir.mkdir(parents=True, exist_ok=True)
    itk = sitk.GetImageFromArray(np.transpose(vol, (2, 1, 0)))  # sitk wants [z, y, x]
    itk.SetSpacing(spacing)
    itk.SetOrigin(origin)
    sitk.WriteImage(itk, str(mhd_dir / "case_0001.mhd"))

    # DICOM series (one file per slice)
    dcm_dir = out / "demo/dicom/case_0001"
    dcm_dir.mkdir(parents=True, exist_ok=True)
    study, series, frame = (generate_uid(entropy_srcs=[f"hi-course-synthetic-{n}"]) for n in ("study", "series", "frame"))
    for k in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid(entropy_srcs=[f"hi-course-synthetic-sop-{k}"])
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.PatientName, ds.PatientID, ds.Modality = "SYNTHETIC^CASE", "case_0001", "CT"
        ds.StudyDescription, ds.SeriesDescription = "Synthetic chest CT", "AXIAL"
        ds.Manufacturer, ds.KVP, ds.ConvolutionKernel = "HI-Course", "120", "STANDARD"
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + k * spacing[2]]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing, ds.SliceThickness = [spacing[1], spacing[0]], spacing[2]
        ds.Rows, ds.Columns = size, size
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.RescaleIntercept, ds.RescaleSlope, ds.RescaleType = -1024, 1, "HU"
        ds.WindowCenter, ds.WindowWidth = -600, 1500
        stored = (vol[:, :, k].T.astype(np.int32) + 1024).astype(np.int16)  # rows = y, cols = x
        ds.PixelData = stored.tobytes()
        ds.save_as(dcm_dir / f"IM_{k + 1:04d}.dcm", enforce_file_format=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--n", type=int, default=60, help="number of cases (= patients)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--volumes", type=int, default=6, help="number of tiny 3D volumes (0 = none)")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(1, args.n + 1):
        case = f"case_{i:04d}"
        z_rel = float(rng.uniform(0.15, 0.9))
        img, lab = fake_slice(rng, args.size, z_rel)
        sp = float(rng.choice([0.6, 0.7, 0.75, 0.8])) * 512 / args.size
        thick = float(rng.choice([1.0, 1.25, 2.5]))
        aff = lps_affine((sp, sp, thick))
        save_nifti(img, aff, args.out / "images" / f"{case}.nii.gz")
        save_nifti(lab, aff, args.out / "labels" / f"{case}.nii.gz")
        n_slices = int(rng.integers(110, 400))
        px = sp * sp
        rows.append(dict(
            case_id=case, seriesuid=f"synthetic.{i}", subset=(i - 1) % 10,
            slice_index=int(z_rel * n_slices), n_slices=n_slices, z_rel=round(z_rel, 4),
            spacing_x=sp, spacing_y=sp, slice_thickness=thick,
            size_x=args.size, size_y=args.size,
            right_lung_px=int((lab == 1).sum()), left_lung_px=int((lab == 2).sum()),
            trachea_px=int((lab == 3).sum()),
            lung_area_mm2=round(float(((lab == 1) | (lab == 2)).sum() * px), 1),
            has_nodule=int(rng.random() < 0.3),
        ))
    pd.DataFrame(rows).to_csv(args.out / "metadata.csv", index=False)
    write_demo(args.out, rng)
    write_volumes(args.out, args.volumes, args.seed)
    (args.out / "README.md").write_text(
        "SYNTHETIC data with the same layout as the LUNA16 lung-slices course dataset.\n"
        "Generated by data_prep/make_synthetic_dataset.py - not real anatomy.\n")
    print(f"wrote {args.n} synthetic cases to {args.out}")


if __name__ == "__main__":
    main()
