"""Lung segmentation on a chest-CT slice - reference Streamlit app (Day 5).

Run:  streamlit run app.py
All the "real work" (preprocessing, model, post-processing, readers) lives in inference.py,
the file we built on day 4. This file only deals with the user interface.
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
# Helpers (no Streamlit calls except the cached model loader)
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model(model_dir: str):
    """Load a model once per server process (not at every rerun of the script)."""
    return inf.load_model(model_dir)


def list_models() -> dict:
    return {str(p.relative_to(MODELS_DIR)): p for p in inf.find_model_dirs(MODELS_DIR)}


def list_samples() -> list:
    ok = (".nii", ".nii.gz", ".dcm")
    return [p for p in sorted(SAMPLES_DIR.glob("*"))
            if p.name.lower().endswith(ok) and "_label" not in p.name.lower()]


def read_file(name: str, content: bytes):
    """Bytes of an uploaded file -> (array, spacing_xy, affine or None). nibabel needs a real file."""
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


def to_display(hu: np.ndarray, center: float, width: float) -> np.ndarray:
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
# Sidebar: notice, model, options, display window
# ---------------------------------------------------------------------------------------------
st.sidebar.warning("Research and education only. NOT a medical device: never use it for diagnosis "
                   "or for any clinical decision.")
models = list_models()
if not models:
    st.title("Lung segmentation")
    st.error("No model found. Put a model folder (containing `config.json` and `model.pt`) inside `models/`, "
             "see `models/README.md`.")
    st.stop()

st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Model", list(models))
model, cfg = get_model(str(models[model_name]))
metrics = cfg.get("metrics", {})
st.sidebar.caption(
    f"**{cfg.get('name', model_name)}** - {cfg['arch']}, input {cfg['preprocessing']['img_size']}  \n"
    f"val Dice: {metrics.get('val_dice', 'n/a')} - test Dice: {metrics.get('test_dice', 'n/a')}  \n"
    f"trained by {cfg.get('trained_by', 'n/a')} on {cfg.get('date', 'n/a')}")

st.sidebar.header("Options")
threshold = st.sidebar.slider("Probability threshold", 0.05, 0.95, 0.5, 0.05)
clean = st.sidebar.checkbox("Post-processing (2 largest components, fill holes)", value=True)
contour = st.sidebar.radio("Overlay style", ["Contour", "Filled"], horizontal=True) == "Contour"
alpha = st.sidebar.slider("Overlay opacity", 0.1, 1.0, 0.6, 0.1)

st.sidebar.header("Display window")
preset = st.sidebar.selectbox("Preset", [*WINDOWS, "Custom"])
center, width = WINDOWS.get(preset, (-600, 1500))
if preset == "Custom":
    center = st.sidebar.slider("Center (HU)", -1000, 1000, -600, 10)
    width = st.sidebar.slider("Width (HU)", 50, 3000, 1500, 50)
st.sidebar.caption("The window changes only how the image is DISPLAYED. The model always sees the same "
                   "HU window it was trained with.")

compare = st.sidebar.checkbox("Compare with a second model", disabled=len(models) < 2)
model_b = None
if compare:
    name_b = st.sidebar.selectbox("Second model", [m for m in models if m != model_name])
    model_b, cfg_b = get_model(str(models[name_b]))

# ---------------------------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------------------------
st.title("🫁 Lung segmentation on a chest-CT slice")
samples = list_samples()
source = st.radio("Input", ["Bundled sample", "Upload a file"] if samples else ["Upload a file"], horizontal=True)

gt_bytes, name, content = None, None, None
if source == "Bundled sample":
    sample = st.selectbox("Sample", samples, format_func=lambda p: p.name)
    name, content = sample.name, sample.read_bytes()
    label_path = SAMPLES_DIR / (sample.name.split(".")[0] + "_label.nii.gz")
    if label_path.exists():
        gt_bytes = label_path.read_bytes()
else:
    up = st.file_uploader("CT slice (.nii, .nii.gz or .dcm)", type=["nii", "gz", "dcm"])
    if up is not None:
        name, content = up.name, up.getvalue()
gt_up = st.file_uploader("Optional: ground-truth mask (.nii.gz, labels 1 and 2 = lungs)", type=["nii", "gz"])
if gt_up is not None:
    gt_bytes = gt_up.getvalue()
if content is None:
    st.info("Upload a CT slice or choose a bundled sample to start.")
    st.stop()

try:
    data, spacing, affine = read_file(name, content)
    slice_idx = None
    if data.ndim == 3 and data.shape[2] > 1:  # a real volume: let the user pick a slice
        slice_idx = st.slider("This file is a 3D volume: choose an axial slice", 0, data.shape[2] - 1,
                              data.shape[2] // 2)
    hu = inf.pick_slice(data, slice_idx)
    gt = None
    if gt_bytes is not None:
        gt_data = read_file("gt.nii.gz", gt_bytes)[0]
        gt = np.isin(inf.pick_slice(gt_data, slice_idx), (1, 2)).astype(np.uint8)
        if gt.shape != hu.shape:
            st.error(f"The mask has shape {gt.shape} but the image has shape {hu.shape}: ignoring the mask.")
            gt = None
except Exception as err:  # bad file: say it nicely instead of showing a traceback
    st.error(f"Could not read this file: {err}")
    st.stop()

if affine is not None and (warning := inf.orientation_warning(affine)):
    st.warning(warning)
if hu.min() >= 0 and hu.max() <= 255:
    st.warning("The values look like an 8-bit image (0-255), not Hounsfield units. "
               "The model expects HU: the result will probably be wrong.")

# ---------------------------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------------------------
res = inf.segment_slice(hu, spacing, model, cfg, threshold=threshold, clean=clean)
ct = to_display(hu, center, width)

st.subheader("Result")
c1, c2, c3 = st.columns(3)
c1.image(ct, caption=f"CT ({preset} window)", width="stretch")
c2.image(draw_mask(ct, res.mask, (1.0, 0.1, 0.1), alpha, contour), caption="Predicted lungs", width="stretch")
c3.image(res.prob.T, caption="Lung probability (white = 1)", clamp=True, width="stretch")

meas = res.measurements
cols = st.columns(4)
cols[0].metric("Lung area", f"{meas['area_cm2']:.1f} cm²")
if meas["right_cm2"] is not None:
    cols[1].metric("Right / left", f"{meas['right_cm2']:.0f} / {meas['left_cm2']:.0f} cm²")
else:
    cols[1].metric("Components found", meas["n_components"])
cols[2].metric("Inference time", f"{res.seconds * 1000:.0f} ms")
cols[3].metric("Image size", f"{hu.shape[0]}x{hu.shape[1]} px")
cols[3].caption(f"pixel spacing {spacing[0]:.2f} x {spacing[1]:.2f} mm")

# ---------------------------------------------------------------------------------------------
# Ground truth, comparison, download
# ---------------------------------------------------------------------------------------------
if gt is not None:
    st.subheader("Prediction vs ground truth")
    g1, g2, g3 = st.columns(3)
    g1.metric("Dice", f"{inf.dice_score(res.mask, gt):.3f}")
    g2.image(draw_mask(ct, gt, (0.1, 0.9, 0.1), alpha, contour), caption="Ground truth", width="stretch")
    g3.image(compare_masks(res.mask, gt), caption="Green = agree, red = false positive, blue = false negative",
             width="stretch")
    st.caption("Reference masks of LUNA16 are 'silver standard' (made by an algorithm): a disagreement is not "
               "always the model's fault.")

if model_b is not None:
    st.subheader(f"Comparison: {model_name} (A) vs {name_b} (B)")
    res_b = inf.segment_slice(hu, spacing, model_b, cfg_b, threshold=threshold, clean=clean)
    b1, b2, b3 = st.columns(3)
    b1.metric("Dice between A and B", f"{inf.dice_score(res.mask, res_b.mask):.3f}")
    b2.image(draw_mask(ct, res_b.mask, (1.0, 0.6, 0.0), alpha, contour),
             caption=f"B: {res_b.measurements['area_cm2']:.1f} cm² in {res_b.seconds * 1000:.0f} ms", width="stretch")
    b3.image(compare_masks(res.mask, res_b.mask), caption="Green = agree, red = only A, blue = only B",
             width="stretch")

st.download_button("⬇️ Download predicted mask (NIfTI)", inf.mask_to_nifti_bytes(res.mask, affine, spacing),
                   file_name=f"{name.split('.')[0]}_lung_mask.nii.gz")
st.caption("Do not upload real patient data to a public app: DICOM headers contain personal information.")
