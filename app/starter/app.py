"""Lung segmentation web app - STARTER (Day 5).

The app RUNS at every stage:  streamlit run app.py
Each step below has a  # TODO  with hints. Replace the placeholder, save the file, and look at the
browser (Streamlit offers "Rerun" / "Always rerun" in the top-right corner). Do the steps in order.

  STEP 1  title + choose / upload a CT slice                        (mandatory)
  STEP 2  read the file and show the CT                             (mandatory)
  STEP 3  load the model ONCE with caching                          (mandatory)
  STEP 4  predict and show the overlay                              (mandatory)
  STEP 5  lung area + Dice and error map vs ground truth            (mandatory)
  STEP 6  sidebar controls: window, threshold, post-processing      (additional)  <- block at the top
  STEP 7  choose among several models + download the mask           (additional)
  STEP 8  compare two models; slice slider for 3D volumes           (optional)

The real work (preprocessing, model, readers) is in inference.py, the file you built on day 4.
"""
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from scipy import ndimage as ndi

import inference as inf

APP_DIR = Path(__file__).parent
MODELS_DIR, SAMPLES_DIR = APP_DIR / "models", APP_DIR / "samples"
WINDOWS = {"Lung": (-600, 1500), "Mediastinum": (40, 400), "Bone": (400, 1800)}  # (center, width) in HU

st.set_page_config(page_title="Lung segmentation", page_icon="🫁", layout="wide")


# ---------------------------------------------------------------------------------------------
# GIVEN helpers: read them, but you do not need to change them
# ---------------------------------------------------------------------------------------------
def list_models() -> dict:
    """{name: folder} for every sub-folder of models/ that contains config.json and model.pt."""
    return {str(p.relative_to(MODELS_DIR)): p for p in inf.find_model_dirs(MODELS_DIR)}


def list_samples() -> list:
    ok = (".nii", ".nii.gz", ".dcm")
    return [p for p in sorted(SAMPLES_DIR.glob("*"))
            if p.name.lower().endswith(ok) and "_label" not in p.name.lower()]


def read_file(name: str, content: bytes):
    """Bytes of a file -> (array, spacing_xy, affine or None). nibabel needs a real file on disk."""
    lower = name.lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        path.write_bytes(content)
        if lower.endswith((".nii", ".nii.gz")):
            return inf.load_nifti(path)
        if lower.endswith(".dcm"):
            hu, spacing = inf.load_dicom(path)
            return hu, spacing, None
    raise ValueError("Unsupported file type. Please use .nii, .nii.gz or .dcm (PNG/JPG have no Hounsfield units).")


def read_gt(gt_bytes, shape):
    """Ground-truth bytes -> binary lung mask (labels 1 and 2), or None if missing / wrong shape."""
    if gt_bytes is None:
        return None
    gt = np.isin(inf.pick_slice(read_file("gt.nii.gz", gt_bytes)[0]), (1, 2)).astype(np.uint8)
    if gt.shape != shape:
        st.error(f"The mask has shape {gt.shape} but the image has shape {shape}: ignoring the mask.")
        return None
    return gt


def to_display(hu, center, width):
    """HU -> RGB float image in [0, 1] for DISPLAY only, radiological orientation (.T)."""
    gray = np.clip((hu - (center - width / 2)) / width, 0, 1).T
    return np.stack([gray] * 3, axis=-1)


def draw_mask(rgb, mask, color, alpha, contour):
    """Paint a mask on an RGB image: a thin contour or a transparent fill."""
    m = mask.T > 0
    region = ndi.binary_dilation(m & ~ndi.binary_erosion(m)) if contour else m
    out = rgb.copy()
    out[region] = (1 - alpha) * out[region] + alpha * np.array(color)
    return out


def compare_masks(a, b):
    """Agreement map: green = both, red = only a, blue = only b (a = prediction, b = reference)."""
    a, b = a.T > 0, b.T > 0
    out = np.zeros(a.shape + (3,))
    out[a & b] = (0.1, 0.8, 0.1)
    out[a & ~b] = (0.9, 0.1, 0.1)
    out[~a & b] = (0.2, 0.4, 1.0)
    return out


# ---------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------
st.sidebar.warning("Research and education only. NOT a medical device: never use it for diagnosis "
                   "or for any clinical decision.")

# STEP 6 (additional) - controls in the sidebar. They sit at the top of the file because the widgets
# must exist before the later steps use their values. For now they are plain constants.
# TODO 6.1: replace the constants by widgets, for example
#     preset = st.sidebar.selectbox("Window preset", [*WINDOWS, "Custom"])
#     center, width = WINDOWS.get(preset, (-600, 1500))
#     if preset == "Custom": center = st.sidebar.slider("Center (HU)", -1000, 1000, -600, 10) ...
# TODO 6.2: threshold = st.sidebar.slider("Probability threshold", 0.05, 0.95, 0.5, 0.05)
#           clean = st.sidebar.checkbox("Post-processing", value=True)
#           alpha = st.sidebar.slider("Overlay opacity", 0.1, 1.0, 0.6, 0.1)
#           contour = st.sidebar.radio("Overlay style", ["Contour", "Filled"]) == "Contour"
# Hint: the window changes only the DISPLAY. The model always receives the same HU window.
center, width = WINDOWS["Lung"]
threshold, clean, alpha, contour = 0.5, True, 0.6, True

models = list_models()
if not models:
    st.title("Lung segmentation")
    st.error("No model found. Put a model folder (containing `config.json` and `model.pt`) inside `models/`, "
             "see `models/README.md`.")
    st.stop()

# ---------------------------------------------------------------------------------------------
# STEP 1 (mandatory) - title and input
# ---------------------------------------------------------------------------------------------
st.title("🫁 Lung segmentation on a chest-CT slice")
samples = list_samples()
source = st.radio("Input", ["Bundled sample", "Upload a file"] if samples else ["Upload a file"], horizontal=True)

name, content, gt_bytes = None, None, None
if source == "Bundled sample":
    sample = st.selectbox("Sample", samples, format_func=lambda p: p.name)
    name, content = sample.name, sample.read_bytes()
    label_path = SAMPLES_DIR / (sample.name.split(".")[0] + "_label.nii.gz")
    if label_path.exists():
        gt_bytes = label_path.read_bytes()
else:
    # TODO 1: replace None with a file uploader that accepts .nii, .nii.gz and .dcm
    #   Hint 1: st.file_uploader(label, type=[...]) returns None until a file is uploaded.
    #   Hint 2: type=["nii", "gz", "dcm"]  (the extension is checked on the last suffix)
    up = None
    if up is not None:
        name, content = up.name, up.getvalue()

if content is None:
    st.info("STEP 1 - upload a CT slice (or add files to samples/) to continue.")
    st.stop()

# ---------------------------------------------------------------------------------------------
# STEP 2 (mandatory) - read the file and show the CT
# ---------------------------------------------------------------------------------------------
data, spacing, affine = read_file(name, content)  # array (2D or 3D), pixel spacing (mm), affine or None

# TODO 2.1: get a 2D slice [x, y] in HU from `data`
#   Hint 1: inference.py has a helper for 2D arrays and single-slice volumes.
#   Hint 2: hu = inf.pick_slice(data)
#   Bonus: a volume with several slices raises inf.NotASliceError -> see STEP 8, or show st.error(...)
hu = None

if hu is None:
    st.info("STEP 2 - get the 2D slice `hu` from `data` (TODO 2.1).")
    st.stop()

# TODO 2.2: show the CT with st.image
#   Hint: ct = to_display(hu, center, width) gives an RGB image ready to show (radiological orientation)
#         st.image(ct, caption="CT", width="stretch")
ct = to_display(hu, center, width)

# ---------------------------------------------------------------------------------------------
# STEP 3 (mandatory) - load the model once
# ---------------------------------------------------------------------------------------------
# Streamlit re-runs this whole script at every interaction. Without caching, the model would be rebuilt
# from disk each time (watch the terminal: the print happens at every click).
# TODO 3.1: add the decorator @st.cache_resource on the line above `def get_model`
#   Hint: cache_resource is for objects that must be shared and not copied (models, connections).
def get_model(model_dir: str):
    print("loading model from", model_dir)
    return inf.load_model(model_dir)


# STEP 7 (additional) will replace this line by a selectbox
model_name = next(iter(models))
model, cfg = get_model(str(models[model_name]))

# TODO 3.2: show the model information in the sidebar (name, architecture, val Dice, trained_by, date)
#   Hint: cfg is the content of config.json: cfg["name"], cfg["arch"], cfg.get("metrics", {}), ...
#         st.sidebar.caption(f"**{cfg['name']}** ...")

# ---------------------------------------------------------------------------------------------
# STEP 4 (mandatory) - predict and show the overlay
# ---------------------------------------------------------------------------------------------
# TODO 4.1: run the model on the HU slice
#   Hint 1: one function does everything (preprocess -> predict -> postprocess -> measure).
#   Hint 2: res = inf.segment_slice(hu, spacing, model, cfg, threshold=threshold, clean=clean)
#   `res` has: res.mask, res.prob, res.measurements (dict), res.seconds
res = None

if res is None:
    st.info("STEP 4 - run the model (TODO 4.1).")
    st.stop()

# TODO 4.2: three columns: CT | predicted mask overlay | probability map
#   Hint 1: c1, c2, c3 = st.columns(3)  and then  c2.image(...)
#   Hint 2: overlay = draw_mask(ct, res.mask, (1.0, 0.1, 0.1), alpha, contour)
#           the probability map is res.prob.T (transpose for the radiological view), use clamp=True
st.subheader("Result")

# ---------------------------------------------------------------------------------------------
# STEP 5 (mandatory) - numbers and ground truth
# ---------------------------------------------------------------------------------------------
# TODO 5.1: show the lung area in cm2 and the inference time with st.metric
#   Hint: res.measurements["area_cm2"], res.seconds * 1000 (ms). Where does the cm2 come from? (spacing!)
# TODO 5.2: if a ground truth exists (gt is not None): show the Dice and the error map
#   Hint 1: inf.dice_score(res.mask, gt)   and   compare_masks(res.mask, gt)  (red = FP, blue = FN)
#   Hint 2: use st.columns(2) and st.image(..., caption="Green = agree, red = false positive, blue = false negative")
gt = read_gt(gt_bytes, hu.shape)
# TODO 5.3 (additional): let the user upload their own ground-truth mask with a second st.file_uploader
#   (labels 1 and 2 are the lungs) and pass its bytes to read_gt.

# ---------------------------------------------------------------------------------------------
# STEP 7 (additional) - choose the model, download the mask
# ---------------------------------------------------------------------------------------------
# TODO 7.1: in STEP 3 replace `model_name = next(iter(models))` by
#           model_name = st.sidebar.selectbox("Model", list(models))
# TODO 7.2: add a download button for the predicted mask as NIfTI
#   Hint 1: st.download_button(label, data, file_name=...)
#   Hint 2: data = inf.mask_to_nifti_bytes(res.mask, affine, spacing)   (affine can be None for DICOM)

# ---------------------------------------------------------------------------------------------
# STEP 8 (optional) - go further
# ---------------------------------------------------------------------------------------------
# TODO 8.1: compare two models side by side and show the Dice between their predictions
#   (a second selectbox, a second get_model call, inf.dice_score(res.mask, res_b.mask), compare_masks)
# TODO 8.2: if `data` is a 3D volume (data.ndim == 3 and data.shape[2] > 1) let the user pick a slice
#   with st.slider and give it to inf.pick_slice(data, slice_idx) in STEP 2. Try samples/volume.nii.gz if present.
# TODO 8.3: add the model card (MODEL_CARD.md) in an st.expander, or an orientation warning
#   with inf.orientation_warning(affine). Add a 3D idea of your own.
st.caption("Do not upload real patient data to a public app: DICOM headers contain personal information.")
