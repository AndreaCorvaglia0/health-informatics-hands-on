#!/usr/bin/env python3
"""Train the teachers' reference model (the "safety net") outside of the notebooks.

The result is a model folder that follows the course contract (model.pt + config.json).
Put it in ``<dataset>/pretrained/unet-baseline/`` before publishing the dataset, so that
students who did not finish day 2 can still do days 3-5, and copy it to ``app/models/``.

    python data_prep/train_baseline.py --data data/luna16-lung-slices --epochs 30
    python data_prep/train_baseline.py --data data/synthetic --epochs 3 --img-size 64   # dry run

It is the same pipeline as notebook 03, condensed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (AsDiscrete, Compose, EnsureChannelFirstd, EnsureTyped, Lambdad, LoadImaged,
                              RandAffined, RandGaussianNoised, Resized, ScaleIntensityRanged)
from monai.utils import set_determinism


def lungs_only(x):
    """1 = right lung, 2 = left lung -> 1; everything else (incl. trachea) -> 0."""
    return ((x == 1) | (x == 2)).to(x.dtype)


def get_transforms(img_size, hu_min=-1000, hu_max=400, augment=False):
    keys = ["image", "label"]
    t = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Lambdad(keys="label", func=lungs_only),
        ScaleIntensityRanged(keys="image", a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=keys, spatial_size=(img_size, img_size), mode=("bilinear", "nearest")),
    ]
    if augment:
        t += [
            RandAffined(keys=keys, prob=0.5, rotate_range=0.15, scale_range=0.1, translate_range=img_size * 0.05,
                        mode=("bilinear", "nearest"), padding_mode="zeros"),
            RandGaussianNoised(keys="image", prob=0.2, std=0.02),
        ]
    t += [EnsureTyped(keys=keys)]
    return Compose(t)


def make_datalists(data_dir: Path):
    meta = pd.read_csv(data_dir / "metadata.csv")
    def rows(mask):
        return [{"image": str(data_dir / "images" / f"{c}.nii.gz"),
                 "label": str(data_dir / "labels" / f"{c}.nii.gz"), "case_id": c}
                for c in meta.loc[mask, "case_id"]]
    return rows(meta.subset <= 6), rows(meta.subset.isin([7, 8])), rows(meta.subset == 9)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <data>/pretrained/unet-baseline")
    ap.add_argument("--name", default="unet-baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)

    set_determinism(42)
    out = args.out or args.data / "pretrained" / args.name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files, val_files, _ = make_datalists(args.data)
    train_ds = CacheDataset(train_files, get_transforms(args.img_size, augment=True), num_workers=args.workers)
    val_ds = CacheDataset(val_files, get_transforms(args.img_size), num_workers=args.workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    arch_kwargs = dict(spatial_dims=2, in_channels=1, out_channels=2, channels=[16, 32, 64, 128, 256],
                       strides=[2, 2, 2, 2], num_res_units=2)
    model = UNet(**arch_kwargs).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(model.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    metric = DiceMetric(include_background=False, reduction="mean")
    post_pred, post_label = AsDiscrete(argmax=True, to_onehot=2), AsDiscrete(to_onehot=2)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["image"].to(device), batch["label"].to(device)
                pred = [post_pred(p) for p in decollate_batch(model(x))]
                metric(y_pred=pred, y=[post_label(l) for l in decollate_batch(y)])
        dice = metric.aggregate().item()
        metric.reset()
        print(f"epoch {epoch:3d}/{args.epochs}  loss {total / len(train_loader):.4f}  val dice {dice:.4f}")
        if dice > best:
            best = dice
            torch.save(model.state_dict(), out / "model.pt")
            config = {
                "name": args.name, "arch": "UNet", "arch_kwargs": arch_kwargs,
                "preprocessing": {"hu_min": -1000, "hu_max": 400, "img_size": [args.img_size, args.img_size]},
                "labels": {"0": "background", "1": "lung"},
                "metrics": {"val_dice": round(best, 4)},
                "trained_by": "teachers (data_prep/train_baseline.py)",
                "date": dt.date.today().isoformat(),
                "training_data": args.data.name, "epochs": args.epochs,
            }
            (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"best val dice {best:.4f} -> {out}")


if __name__ == "__main__":
    main()
