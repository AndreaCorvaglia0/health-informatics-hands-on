"""Compare image embeddings for the multimodal fusion notebook (teacher script).

What it does
------------
For the largest nodule of each patient (601 patients, 64 x 64 CT patches) it
  1. builds the invented clinical table of the notebook (same formula, same seed);
  2. cuts a 32 x 32 crop around the centre of the mask of each nodule;
  3. computes embeddings with two pretrained networks, on the whole patch and on the crop:
       - DenseNet121 (MONAI, ImageNet weights), 1024 numbers;
       - DINOv2 ViT-S/14 (Hugging Face "facebook/dinov2-small"), 384 numbers: the first
         token ("CLS") and, for comparison, the mean of the patch tokens;
  4. reduces each embedding to 10 numbers with a PCA fitted on the training patients only;
  5. prints a table with, for each set of features: the R2 of a ridge regression that predicts
     the log diameter of the nodule, and the test AUC for "alive after one year", alone and
     together with the clinical data. The first rows are the clinical data alone and the five
     measurements taken from the mask.
The table is also saved as emb_results.csv in the current folder.

How to run
----------
    python embedding_comparison.py /path/to/luna16-lung-slices

Needs: numpy, pandas, nibabel, torch, scikit-learn, monai, transformers.
The pretrained weights are downloaded the first time (Internet needed), then cached.
It runs on the CPU in a few minutes. The split (random_state=42) is the one of the notebook.
"""
import sys
import time
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

warnings.filterwarnings("ignore")

CROP_HALF = 16          # the crop is 32 x 32 pixels
N_COMPONENTS = 10       # size of the PCA


def load_patients(data_dir):
    """One row per patient (the largest nodule), with the images and the masks."""
    metadata = pd.read_csv(data_dir / "metadata.csv")
    patients = (metadata.sort_values("diameter_mm", ascending=False)
                .drop_duplicates("case_id").sort_values("case_id").reset_index(drop=True))
    images = np.stack([nib.load(data_dir / "images" / f"{name}.nii.gz").get_fdata() for name in patients["nodule_id"]])
    masks = np.stack([nib.load(data_dir / "labels" / f"{name}.nii.gz").get_fdata() for name in patients["nodule_id"]]) > 0.5
    return patients, images, masks


def make_clinical_table(patients, images, masks, seed=0):
    """The invented clinical data of the notebook (same formula). Returns the table, the diameters and the mean HU."""
    rng = np.random.default_rng(seed)
    n = len(patients)
    z = lambda v: (v - v.mean()) / v.std()
    area = masks.sum(axis=(1, 2)) * patients["spacing_x"].values * patients["spacing_y"].values
    diameter = 2 * np.sqrt(area / np.pi)
    mean_hu = np.array([img[m].mean() for img, m in zip(images, masks)])
    age = np.clip(rng.normal(66, 8, n), 45, 88).round()
    sex = rng.choice(["F", "M"], n)
    pack_years = np.clip(rng.gamma(2.5, 14, n), 0, 120).round()
    performance_status = rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15])
    risk = (-1.0 + 1.0 * z(np.log(diameter)) + 0.4 * z(mean_hu) + 0.6 * z(age)
            + 0.8 * (performance_status - 0.65) + 0.4 * z(pack_years))
    alive_1y = (rng.random(n) > 1 / (1 + np.exp(-risk))).astype(int)
    table = pd.DataFrame({"age": age, "sex": sex, "pack_years": pack_years,
                          "performance_status": performance_status, "alive_1y": alive_1y})
    return table, diameter, mean_hu


def crop_around_mask(images, masks):
    """32 x 32 crop centred on the centre of the mask, moved if needed to stay inside the 64 x 64 patch."""
    crops = []
    for image, mask in zip(images, masks):
        centre_x, centre_y = [int(round(v)) for v in np.argwhere(mask).mean(axis=0)]
        centre_x = min(max(centre_x, CROP_HALF), 64 - CROP_HALF)
        centre_y = min(max(centre_y, CROP_HALF), 64 - CROP_HALF)
        crops.append(image[centre_x - CROP_HALF:centre_x + CROP_HALF, centre_y - CROP_HALF:centre_y + CROP_HALF])
    return np.stack(crops)


def prepare_input(patches, size=224):
    """Window -1000..400 HU to 0..1, three equal channels, 224 x 224, mean and std of ImageNet."""
    x = torch.tensor(np.clip((patches + 1000) / 1400, 0, 1), dtype=torch.float32)[:, None]
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (x - mean) / std


def embed(get_features, patches, batch_size=50):
    """Run a network on all patches, in batches, without training."""
    x = prepare_input(patches)
    with torch.no_grad():
        return torch.cat([get_features(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]).numpy()


def main(data_dir):
    patients, images, masks = load_patients(Path(data_dir) / "nodules")
    clinical, diameter, mean_hu = make_clinical_table(patients, images, masks)
    outcome = clinical["alive_1y"].values
    clinical_features = (clinical[["age", "pack_years", "performance_status"]]
                         .assign(male=(clinical["sex"] == "M").astype(int)).values.astype(float))
    train_idx, test_idx = train_test_split(np.arange(len(outcome)), test_size=0.3, stratify=outcome, random_state=42)

    def test_auc(features):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(features[train_idx], outcome[train_idx])
        return roc_auc_score(outcome[test_idx], model.predict_proba(features[test_idx])[:, 1])

    crops = crop_around_mask(images, masks)

    # The two networks and what we keep from each
    densenet = DenseNet121(spatial_dims=2, in_channels=3, out_channels=1000, pretrained=True).eval()
    dino = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    densenet_features = lambda batch: F.adaptive_avg_pool2d(F.relu(densenet.features(batch)), 1).flatten(1)
    dino_first_token = lambda batch: dino(pixel_values=batch).last_hidden_state[:, 0]
    dino_mean_of_patches = lambda batch: dino(pixel_values=batch).last_hidden_state[:, 1:].mean(1)

    experiments = [
        ("DenseNet121 ImageNet, whole patch", densenet_features, images),
        ("DenseNet121 ImageNet, crop on the nodule", densenet_features, crops),
        ("DINOv2 ViT-S/14 CLS, whole patch", dino_first_token, images),
        ("DINOv2 ViT-S/14 mean of patches, whole patch", dino_mean_of_patches, images),
        ("DINOv2 ViT-S/14 CLS, crop on the nodule", dino_first_token, crops),
    ]
    embeddings = {}
    for name, get_features, patches in experiments:
        start = time.time()
        embeddings[name] = embed(get_features, patches)
        print(name, embeddings[name].shape, round(time.time() - start), "s", flush=True)

    # Five measurements taken from the mask (as in STEP 5 of the notebook)
    area = masks.sum(axis=(1, 2)) * patients["spacing_x"].values * patients["spacing_y"].values
    measurements = np.c_[area, diameter, mean_hu,
                         [img[m].std() for img, m in zip(images, masks)],
                         [img[m].max() for img, m in zip(images, masks)]]

    rows = [("clinical only", 4, None, test_auc(clinical_features), None),
            ("five measurements from the mask", 5, None, test_auc(measurements),
             test_auc(np.c_[measurements, clinical_features]))]
    for name, table in embeddings.items():
        pca = PCA(N_COMPONENTS, random_state=0).fit(table[train_idx])   # training patients only
        reduced = pca.transform(table)
        ridge = make_pipeline(StandardScaler(), RidgeCV()).fit(reduced[train_idx], np.log(diameter[train_idx]))
        r2_diameter = r2_score(np.log(diameter[test_idx]), ridge.predict(reduced[test_idx]))
        rows.append((name, table.shape[1], r2_diameter, test_auc(reduced),
                     test_auc(np.c_[reduced, clinical_features])))

    results = pd.DataFrame(rows, columns=["features", "size", "R2 diameter (test)", "AUC alone", "AUC with clinical"]).round(3)
    print(results.to_string())
    results.to_csv("emb_results.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python embedding_comparison.py /path/to/luna16-lung-slices")
    main(sys.argv[1])
