import tempfile

import nibabel as nib
import numpy as np
import streamlit as st
import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Resize, ScaleIntensityRange

IMG_SIZE = 256
MODEL_PATH = "lung_segmenter.pth"

# HOW TO WORK
# - Do one step at a time, from top to bottom. The instructions are in the comments.
# - Write your code under the comments of the step, where it says "your code here".
# - After each step: save the file (Ctrl+S) and look at the browser. "You should see" tells you what to expect.
# - The full guide is in README.md.


# ----------------------------------------------------------
# Functions already written: you do not need to change them
# ----------------------------------------------------------

def save_upload(uploaded_file):
    """Save an uploaded file to a temporary file and return its path."""
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"     # nibabel reads the extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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
# 1. Show a title with st.title("...")
# 2. Show one line of description with st.write("...")
# 3. Show the note "Student project. Not a medical device." with st.caption("...")
#
# You should see: the title and the two lines of text in the browser.

# your code here


# ----------------------------------------------------------
# STEP 2: Load the model
# ----------------------------------------------------------
# The function load_model creates the network and loads your trained weights.
# @st.cache_resource tells Streamlit to run it only once, not at every click.
#
# 1. Remove the # in front of the lines of the function (select them and press Ctrl+/).
# 2. Fill the three ... with the same values you used in the notebook (STEP 6 of the project).
# 3. Remove the # in front of the last two lines: they call the function and show a message.
#
# You should see: the green message "Model loaded".
# If you see "FileNotFoundError": the file lung_segmenter.pth is not in this folder.

# @st.cache_resource
# def load_model():
#     model = UNet(
#         spatial_dims=...,
#         in_channels=...,
#         out_channels=...,
#         channels=(16, 32, 64, 128, 256),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#     )
#     model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
#     model.eval()
#     return model
#
# model = load_model()
# st.success("Model loaded")


# ----------------------------------------------------------
# STEP 3: Preprocessing and prediction
# ----------------------------------------------------------
# The app must prepare the image exactly as in the training.
#
# 1. Copy here inference_transforms from the notebook (STEP 13 of the project).
# 2. Copy here the function segment from the notebook (STEP 13), with two changes:
#    - there is no GPU here: remove .to(device) and .cpu()
#    - the last line becomes:  return image[0].numpy(), pred.numpy()
#
# You should see: nothing new in the browser, and no error.

# your code here


# ----------------------------------------------------------
# STEP 4: Upload a file
# ----------------------------------------------------------
# 1. Create the upload button:
#        uploaded_file = st.file_uploader("CT slice (.nii or .nii.gz)", type=["nii", "gz"])
# 2. If no file was uploaded yet, show a message and stop the script here:
#        if uploaded_file is None:
#            st.info("Upload a file to start.")
#            st.stop()
# 3. Save the uploaded file and get its path:
#        image_path = save_upload(uploaded_file)
# 4. Call your function segment on image_path and save the two results in image and pred.
# 5. Show the shape of pred with st.write(pred.shape)
#
# You should see: the upload button. After you upload a file from the folder "samples": (256, 256).

# your code here


# ----------------------------------------------------------
# STEP 5: Show the results
# ----------------------------------------------------------
# 1. Create three columns:  col1, col2, col3 = st.columns(3)
# 2. In the first column show the input image:
#        with col1:
#            st.image(image.T, caption="Input", clamp=True)
# 3. In the second column show the predicted mask: pred.T.astype(float), caption "Predicted mask".
# 4. In the third column show the overlay: make_overlay(image, pred).transpose(1, 0, 2), caption "Overlay".
# 5. Delete the st.write(pred.shape) of STEP 4 and the st.success of STEP 2: we do not need them any more.
#
# You should see: three pictures side by side, with the lungs in white in the second and in red in the third.

# your code here


# ----------------------------------------------------------
# STEP 6: Lung area
# ----------------------------------------------------------
# pred contains 1 for each lung pixel. The function pixel_area_cm2(image_path) gives the area of one pixel.
#
# 1. Compute area = number of lung pixels * area of one pixel.
# 2. Show it with:  st.metric("Lung area in this slice", f"{area:.1f} cm²")
#
# You should see: a number of the order of 100 - 400 cm² under the pictures.

# your code here


# ----------------------------------------------------------
# EXTRA: Compare with the reference mask (if you have finished)
# ----------------------------------------------------------
# 1. Add a second upload button for the reference mask. Give it key="mask":
#        mask_file = st.file_uploader("Reference mask (.nii or .nii.gz), optional", type=["nii", "gz"], key="mask")
# 2. If a mask was uploaded: load it with LoadImage, EnsureChannelFirst and Resize (mode="nearest"),
#    and keep only the lungs (labels 1 and 2).
# 3. Compute the Dice score between pred and the reference mask:
#        dice = 2 * (pred * true).sum() / (pred.sum() + true.sum())
# 4. Show it with st.metric.
#
# Other ideas: a slider to choose the colour or the transparency of the overlay (st.slider),
# a button to download the predicted mask (st.download_button).

# your code here
