import tempfile

import nibabel as nib
import numpy as np
import streamlit as st
import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Resize, ScaleIntensityRange

IMG_SIZE = 256
MODEL_PATH = "lung_segmenter.pth"


# ----------------------------------------------------------
# Functions already written: you do not need to change them
# ----------------------------------------------------------

def save_upload(uploaded_file):
    """Save an uploaded file to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.getvalue())
    return tmp.name


def make_overlay(image, mask):
    """Grey image with the mask in red. Returns an RGB array."""
    rgb = np.stack([image, image, image], axis=-1)
    rgb[mask == 1] = 0.6 * rgb[mask == 1] + 0.4 * np.array([1.0, 0.0, 0.0])
    return rgb


def pixel_area_cm2(path):
    """Area of one pixel of the resized image, in cm2."""
    nii = nib.load(path)
    spacing_x, spacing_y = nii.header.get_zooms()[:2]
    size_x, size_y = nii.shape[:2]
    return (spacing_x * size_x / IMG_SIZE) * (spacing_y * size_y / IMG_SIZE) / 100


# ----------------------------------------------------------
# STEP 1: Title
# ----------------------------------------------------------

st.title("Lung segmentation")
st.write("Upload a chest CT slice (NIfTI file) and the model finds the lungs.")
st.caption("Student project. Not a medical device.")


# ----------------------------------------------------------
# STEP 2: Load the model
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()


# ----------------------------------------------------------
# STEP 3: Preprocessing and prediction
# ----------------------------------------------------------

inference_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensityRange(a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    Resize(spatial_size=(IMG_SIZE, IMG_SIZE), mode="bilinear"),
])


def segment(image_path):
    image = inference_transforms(image_path)
    batch = image[None]
    with torch.no_grad():
        outputs = model(batch)
    pred = outputs.argmax(dim=1)[0]
    return image[0].numpy(), pred.numpy()


# ----------------------------------------------------------
# STEP 4: Upload a file
# ----------------------------------------------------------

uploaded_file = st.file_uploader("CT slice (.nii.gz)", type=["nii", "gz"])

if uploaded_file is None:
    st.info("Upload a file to start. You can use the files in the folder 'samples'.")
    st.stop()

image_path = save_upload(uploaded_file)
image, pred = segment(image_path)


# ----------------------------------------------------------
# STEP 5: Show the results
# ----------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.image(image.T, caption="Input", clamp=True)

with col2:
    st.image(pred.T.astype(float), caption="Predicted mask")

with col3:
    st.image(make_overlay(image, pred).transpose(1, 0, 2), caption="Overlay", clamp=True)


# ----------------------------------------------------------
# STEP 6: Lung area
# ----------------------------------------------------------

area = pred.sum() * pixel_area_cm2(image_path)
st.metric("Lung area in this slice", f"{area:.1f} cm²")


# ----------------------------------------------------------
# EXTRA: Compare with the true mask
# ----------------------------------------------------------

mask_file = st.file_uploader("True mask (.nii.gz), optional", type=["nii", "gz"], key="mask")

if mask_file is not None:
    mask_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=(IMG_SIZE, IMG_SIZE), mode="nearest"),
    ])
    mask = mask_transforms(save_upload(mask_file))[0].numpy()
    true = (mask == 1) | (mask == 2)

    dice = 2 * (pred * true).sum() / (pred.sum() + true.sum())
    st.metric("Dice with the true mask", f"{dice:.3f}")
