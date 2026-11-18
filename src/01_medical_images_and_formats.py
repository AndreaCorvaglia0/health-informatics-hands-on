# %% [markdown]
# # 01 - Medical images and file formats
# **Day 1 - part 1** | Advanced Topics in Machine Learning - hands-on | MSc Health Informatics
#
# A CT scan is not a picture: it is a **3D grid of physical measurements** with a position in
# space, a voxel size and a lot of metadata. In this notebook we open the same chest CT in the
# three formats you will meet in practice (DICOM, MetaImage, NIfTI), learn what is inside, and
# finish with a classical "just threshold it" lung segmentation, which will motivate everything
# we do in the following days.
#
# **Learning objectives.** After this notebook you can:
# * read a CT scan in DICOM, MetaImage (`.mhd/.raw`) and NIfTI and say what is inside each (pixels + geometry + metadata);
# * convert stored pixel values to **Hounsfield units (HU)** and apply a **window**;
# * explain voxel space vs. world space (mm), the affine, and the **LPS vs RAS** conventions;
# * display axial, coronal and sagittal views with the correct aspect ratio;
# * build a thresholding baseline, measure it with the **Dice coefficient**, and explain why it is not enough.
#
# **Estimated time:** about 90 minutes (plus the optional challenges).
#
# **Legend**
#
# | Marker | Meaning |
# |---|---|
# | ✏️ **Exercise** | Mandatory. Fill the blanks `____`, then run the check cell. Hints are hidden in 💡 blocks. |
# | 🔍 **Explore** | Small experiments: change something and re-run. |
# | 🚀 **Challenge** | Additional task for those who finish early. |
# | 🌶️ **Going further** | Optional, open-ended. |
# | 🧠 **Question** | Think (or discuss with your neighbour) before moving on. |
# | ⚠️ **Pitfall** | A classic mistake. |
# | 📚 **Read more** | Links for later. |

# %% [markdown]
# ## 1. Setup
# The next cell installs what is missing (it does nothing if everything is already there), the one
# after finds the course dataset. On Kaggle, attach the course dataset with *Add input*.

# %%
# Install what is missing (quiet). On Kaggle/Colab most packages are already there.
import importlib.util
import subprocess
import sys

for module, pip_name in {"pydicom": "pydicom", "SimpleITK": "SimpleITK", "nibabel": "nibabel"}.items():
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)

# %%
import os
from pathlib import Path

# --- Debug run: tiny & fast, to check that everything works before a real run ---
FAST = os.environ.get("HI_FAST", "0") == "1"   # set FAST = True by hand to debug quickly

# --- Where is the data? -----------------------------------------------------------
def find_data_dir() -> Path:
    """Locate the course dataset (folder containing images/, labels/, metadata.csv)."""
    candidates = [os.environ.get("HI_DATA_DIR")]
    candidates += sorted(Path("/kaggle/input").glob("*/")) if Path("/kaggle/input").exists() else []
    candidates += sorted(Path("/kaggle/input").glob("*/*/")) if Path("/kaggle/input").exists() else []
    candidates += ["data/luna16-lung-slices", "../data/luna16-lung-slices", "data/synthetic", "../data/synthetic"]
    for c in candidates:
        if c and (Path(c) / "metadata.csv").exists():
            return Path(c)
    raise FileNotFoundError(
        "Dataset not found. On Kaggle: 'Add input' and attach the course dataset. "
        "Elsewhere: download it (see README) and set HI_DATA_DIR."
    )

DATA_DIR = find_data_dir()
WORK_DIR = Path(os.environ.get("HI_WORK_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "outputs"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
print(f"data: {DATA_DIR}\noutputs: {WORK_DIR}\nFAST mode: {FAST}")

# %% [markdown]
# The folder `demo/` contains **one complete 3D scan stored three times**: as a DICOM series (one
# file per slice), as a MetaImage (`.mhd` header + `.raw` data, the native LUNA16 format) and as a
# NIfTI volume (plus its lung mask). We find them with `glob`, so we never depend on a particular case id.

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pydicom
import SimpleITK as sitk
import nibabel as nib
import scipy.ndimage as ndi

plt.rcParams.update({"figure.dpi": 100, "image.cmap": "gray", "image.interpolation": "nearest"})

DEMO_DIR = DATA_DIR / "demo"
dicom_dir = sorted(p for p in (DEMO_DIR / "dicom").iterdir() if p.is_dir())[0]
dicom_files = sorted(dicom_dir.glob("*.dcm"))
mhd_path = sorted((DEMO_DIR / "mhd").glob("*.mhd"))[0]
nifti_path = sorted((DEMO_DIR / "volume").glob("*_image.nii.gz"))[0]
label_path = sorted((DEMO_DIR / "volume").glob("*_label.nii.gz"))[0]

print("DICOM series :", dicom_dir.name, f"({len(dicom_files)} files)")
print("MetaImage    :", mhd_path.name)
print("NIfTI        :", nifti_path.name, "+", label_path.name)

# %% [markdown]
# ## 2. A CT image is a matrix of physical measurements
# A CT scanner measures how strongly X-rays are attenuated along many directions and reconstructs,
# for every voxel, the **linear attenuation coefficient** $\mu$. To make the number comparable
# across scanners, it is rescaled with water and air as reference points. This gives the
# **Hounsfield unit**:
#
# $$\mathrm{HU} = 1000 \cdot \frac{\mu - \mu_\text{water}}{\mu_\text{water} - \mu_\text{air}}$$
#
# So water is 0 HU and air is about -1000 HU **by definition**. The same tissue has (nearly) the same
# HU on every scanner, which is a big advantage over e.g. MRI.
#
# | Tissue / material | Typical HU |
# |---|---|
# | Air | -1000 |
# | Lung parenchyma (aerated) | -900 to -500 (emphysema: below -950) |
# | Fat | -120 to -80 |
# | Water, CSF | 0 to 15 |
# | Blood (non-contrast) | 30 to 45 |
# | Soft tissue (muscle, liver, heart wall) | 20 to 80 |
# | Contrast-enhanced vessels | 150 to 400 |
# | Cancellous bone | 200 to 400 |
# | Cortical bone | 500 to 1900 |
# | Metal (implants, pacemaker leads) | above 3000 (saturates, causes streak artefacts) |
#
# The seminar covered the other modalities. This is the one-slide summary of **what the pixel value means**:
#
# | Modality | Pixel value is... | Dimensions | Typical formats |
# |---|---|---|---|
# | X-ray (radiograph) | attenuation summed along each ray: a *projection*, no absolute scale | 2D | DICOM (PNG/JPEG in public datasets) |
# | CT | attenuation relative to water, in **HU** (calibrated) | 3D (stack of slices) | DICOM, NIfTI, MetaImage |
# | MRI | signal intensity: depends on sequence (T1, T2, FLAIR...), **arbitrary units** | 3D / 4D | DICOM, NIfTI |
# | PET | radiotracer concentration (Bq/mL, often turned into SUV), low resolution | 3D (4D dynamic) | DICOM, NIfTI |
# | Ultrasound | amplitude of the echo, operator and probe dependent | 2D (+ time) | DICOM cine, video |
#
# > 🧠 **Question:** a radiologist can say "this is 20 HU: probably fluid" on any CT scanner. Why can
# > nobody say the same for a number read in a T2-weighted MR image? What does it imply for pre-processing MR data for a neural network?

# %% [markdown] tags=["solution-only"]
# **Answer.** CT intensities are calibrated physical quantities (HU, anchored on water and air), MR intensities are
# not: they depend on the sequence, the field strength, the coil, the scanner vendor, the day. For MR we have to
# *normalise per image* (z-score, percentiles, histogram matching) before feeding a network; for CT we can use a
# **fixed HU window** and keep the physical meaning (we will do this in notebook 02).

# %% [markdown]
# ## 3. DICOM: one file, one slice, a lot of metadata
# **DICOM** is the format of hospital scanners and PACS. One file = one 2D image (a slice) + a
# header with hundreds of **tags**: `(group, element)` pairs with a name, a value representation and a value.
# A 3D scan is a *series*: many files that share a `SeriesInstanceUID`.
#
# We use [`pydicom`](https://pydicom.github.io/pydicom/stable/). `dcmread` returns a `Dataset` that you can
# access by keyword (`ds.Modality`) or by tag (`ds[0x0008, 0x0060]`).

# %%
ds = pydicom.dcmread(dicom_files[len(dicom_files) // 2])
print(ds)

# %%
for keyword in ["PatientID", "Modality", "Rows", "Columns", "PixelSpacing", "SliceThickness",
                "ImagePositionPatient", "ImageOrientationPatient", "RescaleSlope", "RescaleIntercept",
                "WindowCenter", "WindowWidth", "BitsStored", "PixelRepresentation"]:
    print(f"{keyword:26s} {ds.get(keyword, '<missing>')}")

print("\nSame tag, by number:", ds[0x0028, 0x0030])   # (0028,0030) = Pixel Spacing

# %% [markdown]
# The ones that matter for us:
#
# | Tag | Meaning |
# |---|---|
# | `PixelSpacing` | size of a pixel in mm, **as [row spacing, column spacing]** = [y, x] |
# | `SliceThickness` | thickness of the slice in mm (not always equal to the distance between slices!) |
# | `ImagePositionPatient` | world position (mm) of the **centre of the first pixel** of this slice |
# | `ImageOrientationPatient` | direction cosines of the image rows and columns in world space |
# | `RescaleSlope`, `RescaleIntercept` | stored value -> HU: `HU = stored * slope + intercept` |
# | `WindowCenter`, `WindowWidth` | suggested display window (radiologist-style) |
#
# > ⚠️ **Pitfall:** DICOM pixels are **stored values**, not HU. Scanners often store unsigned numbers
# > with an offset (typically `RescaleIntercept = -1024`), so a stored 24 means -1000 HU (air).
# > If you forget the rescale your "air" is a dark gray and everything downstream (windows, thresholds,
# > a network trained on HU) is wrong. Convert to a **float** array *before* multiplying: small integer types can overflow.

# %% [markdown]
# ### ✏️ Exercise 1.1 - from stored values to Hounsfield units
# Complete `dicom_to_hu(ds)`: read the slope and the intercept from the header and apply `HU = stored * slope + intercept`.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The header attributes are called `RescaleSlope` and `RescaleIntercept`. pydicom returns them as `DS` objects (decimal strings): wrap them in `float(...)`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)`
#
# </details>

# %%
def dicom_to_hu(ds):
    """Stored pixel values -> Hounsfield units (float32), array of shape (rows, columns)."""
    stored = ds.pixel_array.astype(np.float32)  # float first: no integer overflow
    #@solution
    slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
    #@student
    #| slope, intercept = ____, ____  # TODO: read both values from the header
    #@end
    return stored * slope + intercept


hu_slice = dicom_to_hu(ds)
print("stored values:", ds.pixel_array.dtype, ds.pixel_array.min(), "..", ds.pixel_array.max())
print("in HU        :", hu_slice.dtype, hu_slice.min(), "..", hu_slice.max())

# %% [markdown]
# Check cell for exercise 1.1.

# %%
assert hu_slice.shape == (ds.Rows, ds.Columns), "the shape of the array must not change"
assert hu_slice.dtype == np.float32, "return a float32 array"
assert hu_slice.min() < -500, "a chest slice always contains air (about -1000 HU): did you apply the intercept?"
assert hu_slice.max() < 4000, "HU values above 4000 do not exist in a CT: check the formula"
print("✅ Exercise 1.1 passed")

# %% [markdown]
# `pixel_array` has shape `(rows, columns)` = `(y, x)`, so it can be shown directly with `imshow`.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
axes[0].imshow(ds.pixel_array)
axes[0].set_title("stored values (no meaning!)")
axes[1].imshow(hu_slice, vmin=-1000, vmax=400)
axes[1].set_title("HU, displayed between -1000 and 400")
axes[2].hist(hu_slice.ravel(), bins=100, log=True)
axes[2].set(title="HU histogram (log counts)", xlabel="HU")
for ax in axes[:2]:
    ax.axis("off")
plt.tight_layout()

# %% [markdown]
# ### Privacy: the header is full of personal data
# DICOM headers can contain the patient's name, birth date, address of the hospital, physician names,
# accession numbers... (**PHI**, protected health information, covered by GDPR in the EU). Some can even be
# burned into the pixels (ultrasound and secondary captures!). Public datasets such as LUNA16 (derived from
# LIDC-IDRI on TCIA) are **de-identified**: those tags are removed or replaced. Before sharing medical images,
# always check *both* the header and the pixels. Our demo file only has a dummy name:

# %%
print("PatientName:", ds.PatientName, "| PatientID:", ds.PatientID)

# %% [markdown]
# ## 4. From a DICOM series to a 3D volume
# We read all the files of the series. To simulate what happens in real life (a PACS export, a zip file
# extracted in random order, a series merged from two folders) we **shuffle** the list on purpose.

# %%
slices = [pydicom.dcmread(f) for f in dicom_files]
shuffled = [slices[i] for i in np.random.default_rng(0).permutation(len(slices))]

print("z of the first 8 files, in the order we received them:")
print([float(s.ImagePositionPatient[2]) for s in shuffled[:8]])
print("InstanceNumber of the same files:", [int(s.InstanceNumber) for s in shuffled[:8]])

# %% [markdown]
# > ⚠️ **Pitfall:** the **file name** order is arbitrary, and `InstanceNumber` is only a counter chosen by
# > the scanner software (it can restart, be duplicated across series, or run in the opposite direction).
# > The reliable way to order slices is their **position in space**: `ImagePositionPatient`.
# > (For tilted acquisitions, project the position on the slice normal, the cross product of the
# > two vectors in `ImageOrientationPatient`. For axial CT, the z coordinate is enough.)
#
# ### ✏️ Exercise 1.2 - sort the slices and derive the voxel size along z
# 1. Sort `shuffled` by the z coordinate of `ImagePositionPatient` (the third value).
# 2. Compute `dz`, the distance in mm between consecutive slices, from the sorted z positions. Use the median of the differences (robust to a missing slice).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `sorted(list, key=function)` calls `function` on every element and sorts by the result. Here the function returns the z coordinate of a slice.
#
# </details>
#
# <details>
# <summary>💡 Hint 2</summary>
#
# `np.diff(z_positions)` gives the distances between consecutive sorted positions.
#
# </details>
#
# <details>
# <summary>💡 Hint 3 (almost the solution)</summary>
#
# `key=lambda s: float(s.ImagePositionPatient[2])` and `dz = float(np.median(np.diff(z_positions)))`
#
# </details>

# %%
#@solution
slices_sorted = sorted(shuffled, key=lambda s: float(s.ImagePositionPatient[2]))
#@student
#| slices_sorted = sorted(shuffled, key=lambda s: ____)  # TODO: the z coordinate of this slice
#@end
z_positions = np.array([float(s.ImagePositionPatient[2]) for s in slices_sorted])
#@solution
dz = float(np.median(np.diff(z_positions)))
#@student
#| dz = ____  # TODO: median distance between consecutive slices (mm)
#@end

# Stack the slices: (z, y, x), the same layout as SimpleITK arrays
volume_zyx = np.stack([dicom_to_hu(s) for s in slices_sorted])
ps_row, ps_col = (float(v) for v in slices_sorted[0].PixelSpacing)   # [row, column] = [y, x] spacing
spacing_xyz = (ps_col, ps_row, dz)
print("volume (z, y, x):", volume_zyx.shape, "| voxel size x, y, z (mm):", spacing_xyz,
      "| SliceThickness tag:", slices_sorted[0].SliceThickness)

# %%
assert np.all(np.diff(z_positions) > 0), "slices must be sorted from the lowest to the highest z"
assert dz > 0 and np.isclose(dz, np.diff(z_positions).mean(), rtol=0.2), "dz is the (positive) distance between slices in mm"
assert volume_zyx.shape[0] == len(dicom_files)
print("✅ Exercise 1.2 passed")

# %% [markdown]
# Anisotropic voxels are normal in CT: the in-plane pixel is typically 0.6-0.9 mm but slices can be 1-5 mm
# apart. (On real LUNA16 scans the course notebooks will show you the distribution tomorrow.)
# Note also that `SliceThickness` and the distance between slice centres (`dz`) can differ (overlapping or gapped reconstructions).

# %% [markdown]
# ## 5. Windowing: choose which HU range you want to see
# CT spans about 4000 HU but a screen (and your eye) can distinguish only a few hundred gray levels.
# Radiologists therefore view the image through a **window** defined by a *center* and a *width*:
# everything below `center - width/2` is black, everything above `center + width/2` is white, and the
# range in between is mapped linearly to gray.
#
# | Window | Center / width (HU) | Shows well |
# |---|---|---|
# | Lung | -600 / 1500 | airways, vessels, nodules, emphysema |
# | Mediastinum (soft tissue) | 40 / 400 | heart, vessels, lymph nodes, muscle |
# | Bone | 400 / 1800 | ribs, spine, cortex |
#
# ### ✏️ Exercise 1.3 - implement `apply_window`
# Return an array in `[0, 1]`: 0 below the lower edge, 1 above the upper edge, linear in between.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# First compute the lower edge: `low = center - width / 2`. Then rescale: `(hu - low) / width`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# Finish with `np.clip(..., 0, 1)`.
#
# </details>

# %%
def apply_window(hu, center, width):
    """Map HU to [0, 1] with a CT window (values outside the window saturate)."""
    low = center - width / 2
    #@solution
    return np.clip((hu - low) / width, 0.0, 1.0)
    #@student
    #| return ____  # TODO: linear rescale of [low, low + width] to [0, 1], then clip
    #@end


# %%
test = apply_window(np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0]), center=0, width=1000)
assert np.allclose(test, [0, 0, 0.5, 1, 1]), f"expected [0, 0, 0.5, 1, 1], got {test}"
assert apply_window(volume_zyx[0], 0, 400).min() >= 0 and apply_window(volume_zyx[0], 0, 400).max() <= 1
print("✅ Exercise 1.3 passed")

# %%
WINDOWS = {"lung": (-600, 1500), "mediastinum": (40, 400), "bone": (400, 1800)}
mid = volume_zyx[len(volume_zyx) // 2]

fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
axes[0].imshow(mid, vmin=mid.min(), vmax=mid.max())
axes[0].set_title("full range (no window)")
for ax, (name, (c, w)) in zip(axes[1:], WINDOWS.items()):
    ax.imshow(apply_window(mid, c, w), vmin=0, vmax=1)
    ax.set_title(f"{name}  (C={c}, W={w})")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
print("window stored in the DICOM header: center", ds.get("WindowCenter"), "width", ds.get("WindowWidth"))

# %% [markdown]
# ### 🔍 Explore
# * Try a **very narrow** window (e.g. center 40, width 50): what happens to the lungs? And to soft-tissue contrast?
# * With a window of width `W`, how many HU does one gray level (out of 256) cover? Compare W=1500 and W=400.
# * Try a window centered at -900 with width 200: which structure is now visible *inside* the lungs?
#
# > 🧠 **Question:** a neural network sees the numbers, not the screen. What are the pros and cons of
# > feeding it a lung window instead of the full HU range? (We will make this choice for our model in notebook 02.)

# %% [markdown] tags=["solution-only"]
# **Answers.** One gray level covers `W / 256` HU: about 5.9 HU for W=1500 and 1.6 HU for W=400, so a narrow window
# separates tissues with similar HU. Around -900 / width 200 one sees the fine texture of the parenchyma and the
# smallest vessels. A window *removes* information (everything outside saturates) but also removes irrelevant variation
# and rescales the input to [0, 1], which helps training; if the window is too tight we may lose the structures we need
# (for lungs, the vessels and nodules need the soft-tissue range, the airways need the very low range).

# %% [markdown]
# ## 6. MetaImage (`.mhd` + `.raw`): the LUNA16 native format
# MetaImage is the format of the ITK world. A **text header** (`.mhd`) describes a **raw binary file** (`.raw`).
# It is very simple, and it is how LUNA16 ships its scans. Let us look at the header, it is just text:

# %%
print(mhd_path.read_text())

# %% [markdown]
# [SimpleITK](https://simpleitk.readthedocs.io/) reads it and exposes geometry as `GetSize()`, `GetSpacing()`,
# `GetOrigin()` and `GetDirection()`, all in **(x, y, z)** order.

# %%
itk_img = sitk.ReadImage(str(mhd_path))
print("size (x, y, z):", itk_img.GetSize())
print("spacing (mm)  :", itk_img.GetSpacing())
print("origin (mm)   :", itk_img.GetOrigin())
print("direction     :", itk_img.GetDirection())

arr_zyx = sitk.GetArrayFromImage(itk_img)
print("numpy array shape:", arr_zyx.shape, arr_zyx.dtype)

# %% [markdown]
# > ⚠️ **Pitfall:** SimpleITK describes the image in **(x, y, z)** but `GetArrayFromImage` returns a numpy
# > array indexed **[z, y, x]**. Mixing the two orders is the most frequent bug when moving between ITK and numpy
# > (the shape is "reversed" and so is the spacing you need for plotting).
#
# ### ✏️ Exercise 1.4 - from `[z, y, x]` to `[x, y, z]`
# Our NIfTI files (next section) and the 2D course dataset use `[x, y, z]`. Reorder the axes of `arr_zyx` with `np.transpose`.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `np.transpose(a, axes)` lists, for each *new* axis, which *old* axis goes there. Old axes are 0=z, 1=y, 2=x.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# We want x first, then y, then z: `axes=(2, 1, 0)`.
#
# </details>

# %%
#@solution
vol_mhd = np.transpose(arr_zyx, (2, 1, 0))
#@student
#| vol_mhd = np.transpose(arr_zyx, (____, ____, ____))  # TODO: new axis order (x, y, z)
#@end
print("vol_mhd:", vol_mhd.shape, "| sitk size:", itk_img.GetSize())

# %%
assert vol_mhd.shape == itk_img.GetSize(), "the array shape must now match (x, y, z) as in SimpleITK"
assert np.array_equal(vol_mhd[3, 5, 1], arr_zyx[1, 5, 3]), "voxel (x=3, y=5, z=1) is arr_zyx[z=1, y=5, x=3]"
print("✅ Exercise 1.4 passed")

# %% [markdown]
# ## 7. NIfTI: the neuroimaging format, with an explicit affine
# **NIfTI** (`.nii`, `.nii.gz`) stores one array and a header with the **affine**: a 4x4 matrix that maps
# *voxel indices* `(i, j, k)` to *world coordinates* in mm. It is the format of our 2D course dataset
# (one 2D slice per file) and the preferred one for deep learning pipelines. We use [`nibabel`](https://nipy.org/nibabel/).

# %%
nii = nib.load(nifti_path)
vol = np.asanyarray(nii.dataobj)                       # keeps the stored dtype (int16); get_fdata() would give float64
lab = np.asanyarray(nib.load(label_path).dataobj)      # 0 background, 1 right lung, 2 left lung, 3 trachea
zooms = tuple(float(z) for z in nii.header.get_zooms()[:3])

print("shape:", nii.shape, "| dtype:", vol.dtype, "| voxel size (mm):", np.round(zooms, 3))
print("axis codes:", nib.aff2axcodes(nii.affine))
print("affine:\n", np.round(nii.affine, 3))

# %% [markdown]
# ### Voxel space vs world space, and orientation conventions
# * **Index space:** the array indices `(i, j, k)`. No units, no direction.
# * **World space:** physical coordinates in mm: `world = affine @ [i, j, k, 1]`. The three diagonal numbers are
#   the voxel sizes (with sign), the last column is the world position of voxel `(0, 0, 0)`.
#
# Which way do the world axes point? There are two conventions, and they cause endless confusion:
#
# | Convention | +x points to | +y points to | +z points to | Used by |
# |---|---|---|---|---|
# | **LPS+** | patient's **L**eft | **P**osterior | **S**uperior | DICOM, ITK/SimpleITK, LUNA16 |
# | **RAS+** | patient's **R**ight | **A**nterior | **S**uperior | NIfTI world space, nibabel, 3D Slicer, MONAI |
#
# They differ only by the sign of x and y. `nib.aff2axcodes` tells in which direction the **array axes** grow:
# `('L', 'P', 'S')` means that index `i` grows towards the patient's left, `j` towards posterior, `k` towards superior.
# **Our files are stored in LPS index order** (like the DICOM they come from), the affine records it
# with negative x/y entries, and nothing is lost. A different NIfTI could store the same patient as `('R','A','S')`:
# same anatomy, flipped array. This is why libraries offer `nib.as_closest_canonical` (and MONAI `Orientationd`).
#
# > ⚠️ **Pitfall:** `plt.imshow(arr)` draws axis 0 as **rows** (vertical). For our `[x, y]` slices this puts
# > the left-right axis vertically. We need `arr.T`.

# %%
slice_xy = vol[:, :, vol.shape[2] // 2]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (title, img, origin) in zip(axes, [("imshow(img)", slice_xy, "upper"),
                                           ("imshow(img.T)  <- the one we use", slice_xy.T, "upper"),
                                           ("imshow(img.T, origin='lower')", slice_xy.T, "lower")]):
    ax.imshow(img, cmap="gray", vmin=-1000, vmax=400, origin=origin)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
# anatomical letters on the radiological view: patient's right is on the left of the screen
axes[1].text(0.02, 0.5, "R", transform=axes[1].transAxes, color="yellow", fontsize=16, va="center")
axes[1].text(0.95, 0.5, "L", transform=axes[1].transAxes, color="yellow", fontsize=16, va="center")
axes[1].text(0.5, 0.96, "A", transform=axes[1].transAxes, color="yellow", fontsize=16, ha="center")
axes[1].text(0.5, 0.02, "P", transform=axes[1].transAxes, color="yellow", fontsize=16, ha="center")
plt.tight_layout()

# %% [markdown]
# With `img.T`, index `i` (x, towards the patient's **left**) runs to the right of the screen and index `j`
# (y, towards **posterior**) runs downwards: anterior on top, the spine at the bottom, and the patient's right
# on the left of the screen. This is the standard **radiological view**: we look at the patient from the feet.
#
# > 🧠 **Question:** why does the right lung have *low* x indices in our arrays? And what would you see
# > if you displayed the mask of the right lung (label 1) with the correct code `img.T`, but you had
# > accidentally loaded a file stored as `('R','A','S')`?

# %% [markdown] tags=["solution-only"]
# **Answer.** Index x grows towards the patient's left (LPS), so the patient's right side has the *smallest* x, and
# it appears on the left of the screen after `.T`. If the file were stored as RAS, x would grow towards the
# patient's right and, with the same `.T` recipe, left and right would be swapped on screen: the "right lung" label
# would be drawn on the screen-right side, and any left-right flip augmentation or L/R-based rule would silently be wrong.
# Always check `aff2axcodes` (and the anatomy!) when you get data from a new source.

# %% [markdown]
# ### ✏️ Exercise 1.5 - use the affine
# 1. Compute `size_mm`: the physical extent (mm) of the volume along x, y and z (number of voxels times voxel size).
# 2. Compute `world_center`: the world coordinate (mm, RAS) of the central voxel `ijk_center`, **by hand** with the affine
#    (matrix product with the homogeneous vector `[i, j, k, 1]`).
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `nii.shape` has the number of voxels per axis and `zooms` the voxel size in mm: multiply them element by element (`np.array(...) * np.array(...)`).
#
# </details>
#
# <details>
# <summary>💡 Hint 2</summary>
#
# Build the homogeneous vector with `np.append(ijk_center, 1)`, multiply with `affine @ vector` and keep the first three numbers.
#
# </details>
#
# <details>
# <summary>💡 Hint 3 (almost the solution)</summary>
#
# `world_center = (affine @ np.append(ijk_center, 1))[:3]`
#
# </details>

# %%
affine = nii.affine
ijk_center = np.array(vol.shape[:3]) // 2
#@solution
size_mm = np.array(vol.shape[:3]) * np.array(zooms)
world_center = (affine @ np.append(ijk_center, 1))[:3]
#@student
#| size_mm = ____         # TODO: physical size (mm) along x, y, z
#| world_center = ____    # TODO: world coordinate of the voxel ijk_center, using the affine
#@end
print("physical size (mm), x y z:", np.round(size_mm, 1))
print("world coordinate of the central voxel (RAS, mm):", np.round(world_center, 1))

# %%
assert np.asarray(size_mm).shape == (3,) and np.all(np.asarray(size_mm) > 0)
assert np.allclose(world_center, nib.affines.apply_affine(affine, ijk_center)), "compare with nibabel's apply_affine"
print("✅ Exercise 1.5 passed")

# %% [markdown]
# Now we put the three formats in relation. The DICOM position is LPS; to compare with the NIfTI (RAS) we flip
# the sign of x and y, then use the **inverse** affine to find the voxel where the first DICOM slice lives:

# %%
ipp_lps = np.array([float(v) for v in slices_sorted[0].ImagePositionPatient])
ipp_ras = ipp_lps * np.array([-1, -1, 1])
ijk_first_slice = np.linalg.inv(affine) @ np.append(ipp_ras, 1)
print("first DICOM slice (LPS mm):", ipp_lps, "-> NIfTI voxel index:", np.round(ijk_first_slice[:3], 2))

# %% [markdown]
# Integer indices mean the grids coincide. Finally, is it really the same scan in the three files?

# %%
volumes = {"DICOM": np.transpose(volume_zyx, (2, 1, 0)), "MetaImage": vol_mhd, "NIfTI": vol}
for name, v in volumes.items():
    print(f"{name:10s} shape {v.shape}  dtype {str(v.dtype):8s} HU range [{v.min():.0f}, {v.max():.0f}]")

if len({v.shape for v in volumes.values()}) == 1:
    print("identical voxel values:", {n: bool(np.allclose(v, vol)) for n, v in volumes.items()})
else:
    print("The shapes differ (e.g. the DICOM series is only a slab of the scan): compare the overlapping slices.")

# %% [markdown]
# ### 🔍 Explore
# * Load the volume with `nib.as_closest_canonical(nii)`: which axis codes do you get? Is the array flipped (compare `canon.get_fdata()[:, :, k]` with `vol[:, :, k]` after a flip)? Is the *world* position of a given anatomical point changed?
# * `pydicom.dcmread(f, stop_before_pixels=True)` reads only the header: how much faster is it for all the files? (Handy to index thousands of series.)
# * Print `nii.header` and find `sform_code`, `qform_code`, `xyzt_units`.
#
# > 📚 **Read more:** [nibabel: coordinate systems and affines](https://nipy.org/nibabel/coordinate_systems.html) |
# > [3D Slicer: coordinate systems](https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html)

# %% [markdown]
# ## 8. Three orthogonal views, with the right aspect ratio
# A 3D scan can be sliced in three ways: **axial** (from the feet, the one used for our 2D task), **coronal**
# (from the front) and **sagittal** (from the side). Because voxels are not cubes, we must tell `imshow` the
# shape of one pixel, otherwise the anatomy looks squashed: `aspect = pixel height (mm) / pixel width (mm)`.
#
# With `vol[x, y, z]` and `.T` the three views are:
#
# | View | 2D array | rows = | columns = | `origin` |
# |---|---|---|---|---|
# | axial | `vol[:, :, k].T` | y | x | `"upper"` |
# | coronal | `vol[:, j, :].T` | z | x | `"lower"` (superior on top) |
# | sagittal | `vol[i, :, :].T` | z | y | `"lower"` (anterior on the left) |
#
# ### ✏️ Exercise 1.6 - the aspect ratio of the coronal and sagittal views
# `sx, sy, sz` are the voxel sizes. For the axial view the pixel height is `sy` and the width `sx`, so `aspect_axial = sy / sx`. Fill in the other two.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# In the coronal view the rows run along z and the columns along x. What is the height and the width of one pixel in mm?
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# Coronal: height `sz`, width `sx`. Sagittal: height `sz`, width `sy`.
#
# </details>

# %%
sx, sy, sz = zooms
aspect_axial = sy / sx
#@solution
aspect_coronal = sz / sx
aspect_sagittal = sz / sy
#@student
#| aspect_coronal = ____    # TODO: pixel height / pixel width (mm)
#| aspect_sagittal = ____   # TODO
#@end
print(f"aspect  axial {aspect_axial:.2f} | coronal {aspect_coronal:.2f} | sagittal {aspect_sagittal:.2f}")

# %%
assert np.isclose(aspect_coronal * sx, sz) and np.isclose(aspect_sagittal * sy, sz), "aspect = height_mm / width_mm"
print("✅ Exercise 1.6 passed")

# %%
# Choose the slice with most lung, and cut through the centre of the right lung
lung_3d = np.isin(lab, (1, 2))
k_lung = int(np.argmax(lung_3d.sum(axis=(0, 1))))
right_xy = np.argwhere(lab[:, :, k_lung] == 1)
i_cut, j_cut = (int(c) for c in right_xy.mean(axis=0))

LABEL_COLORS = {1: "tab:red", 2: "tab:blue", 3: "gold"}   # right lung, left lung, trachea

def label_to_rgba(lab2d, alpha=0.45):
    """Colour a 2D label map (0 = transparent) so it can be drawn on top of the image."""
    rgba = np.zeros(lab2d.shape + (4,))
    for value, color in LABEL_COLORS.items():
        rgba[lab2d == value] = (*mcolors.to_rgb(color), alpha)
    return rgba

def show_view(ax, img2d, lab2d, aspect, origin, title, window=(-600, 1500)):
    ax.imshow(apply_window(img2d, *window), cmap="gray", vmin=0, vmax=1, aspect=aspect, origin=origin)
    if lab2d is not None:
        ax.imshow(label_to_rgba(lab2d), aspect=aspect, origin=origin)
    ax.set_title(title)
    ax.axis("off")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
show_view(axes[0], vol[:, :, k_lung].T, lab[:, :, k_lung].T, aspect_axial, "upper", f"axial (k={k_lung})")
show_view(axes[1], vol[:, j_cut, :].T, lab[:, j_cut, :].T, aspect_coronal, "lower", f"coronal (j={j_cut})")
show_view(axes[2], vol[i_cut, :, :].T, lab[i_cut, :, :].T, aspect_sagittal, "lower", f"sagittal (i={i_cut})")
plt.tight_layout()

# %% [markdown]
# Red = right lung, blue = left lung, yellow = trachea (labels of the course dataset).
#
# ### 🔍 Explore
# * Set `aspect=1` for the coronal view and compare: how much is the anatomy distorted when the slice spacing is large?
# * Change the window of `show_view` (`window=(40, 400)`): does the label still fit the lung borders?
# * Move `k_lung`, `i_cut`, `j_cut` and look at the labels along the scan: how does the lung change from the apex to the base?

# %% [markdown]
# ## 9. A classical baseline: threshold the lungs
# Lungs are full of air, so they are the darkest large structure of the body. A first idea: **everything
# below -400 HU is lung**. We take the axial slice with the most lung and look at its HU histogram.

# %%
slice_hu = vol[:, :, k_lung].astype(np.float32)
slice_ref = np.isin(lab[:, :, k_lung], (1, 2))        # reference mask: right + left lung (trachea excluded)

lo, hi = -1100, 1500
outside = int(((slice_hu < lo) | (slice_hu > hi)).sum())
plt.figure(figsize=(9, 3.5))
plt.hist(slice_hu.ravel(), bins=150, range=(lo, hi), log=True, color="gray")
plt.axvspan(lo, -400, color="tab:blue", alpha=0.15, label="below -400 HU")
plt.xlabel("HU"); plt.ylabel("pixels (log)"); plt.legend()
plt.title(f"HU histogram of the axial slice (not shown: {outside} pixels outside [{lo}, {hi}], e.g. scanner padding)")
plt.tight_layout()

# %% [markdown]
# ### ✏️ Exercise 1.7 - the threshold mask
# Complete `threshold_baseline`: it returns a boolean mask that is `True` where the HU value is **below** the threshold.
# (We deliberately do not do anything smarter yet.)
#
# <details>
# <summary>💡 Hint 1</summary>
#
# Comparing a numpy array with a number gives a boolean array: `a < 5`.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `mask = hu < threshold`
#
# </details>

# %%
def threshold_baseline(hu, threshold=-400):
    """Classical baseline: predict 'lung' wherever the HU value is below the threshold."""
    #@solution
    mask = hu < threshold
    #@student
    #| mask = ____  # TODO: boolean array, True where hu is below the threshold
    #@end
    return mask


pred = threshold_baseline(slice_hu)

# %%
assert pred.dtype == bool and pred.shape == slice_hu.shape, "return a boolean mask with the shape of the image"
assert pred[slice_hu < -900].all() and not pred[slice_hu > 0].any(), "very dark pixels must be selected, bright ones not"
print("✅ Exercise 1.7 passed")

# %% [markdown]
# ### ✏️ Exercise 1.8 - the Dice coefficient, by hand
# How good is the mask? The standard metric for segmentation is the **Dice coefficient** (also called F1 score for sets):
#
# $$\mathrm{Dice}(P, R) = \frac{2\,|P \cap R|}{|P| + |R|}$$
#
# where $P$ is the predicted mask and $R$ the reference. It is 1 for a perfect match and 0 for no overlap.
# Implement it with numpy. A tiny `eps` in numerator and denominator makes two empty masks give 1 instead of a division by zero.
#
# <details>
# <summary>💡 Hint 1</summary>
#
# The intersection of two boolean arrays is `a & b`; `.sum()` counts the `True` values.
#
# </details>
#
# <details>
# <summary>💡 Hint 2 (almost the solution)</summary>
#
# `intersection = (pred & ref).sum()` and `dice = (2 * intersection + eps) / (pred.sum() + ref.sum() + eps)`
#
# </details>

# %%
def dice_score(pred, ref, eps=1e-8):
    """Dice coefficient between two binary masks (any shape)."""
    pred, ref = np.asarray(pred, dtype=bool), np.asarray(ref, dtype=bool)
    #@solution
    intersection = (pred & ref).sum()
    dice = (2 * intersection + eps) / (pred.sum() + ref.sum() + eps)
    #@student
    #| intersection = ____  # TODO: number of pixels that are True in both masks
    #| dice = ____          # TODO: the formula above, with eps
    #@end
    return float(dice)


# %%
a, b = np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])
assert np.isclose(dice_score(a, a), 1.0) and np.isclose(dice_score(a, 1 - a), 0.0, atol=1e-6)
assert np.isclose(dice_score(a, b), 2 / 3, atol=1e-6), "2 * 1 / (2 + 1) = 0.667"
assert np.isclose(dice_score(a, b), dice_score(b, a))
print("✅ Exercise 1.8 passed")

# %%
print(f"Dice of the threshold baseline on this slice: {dice_score(pred, slice_ref):.3f}")

def error_overlay(pred, ref):
    """RGBA map: green = correct, red = false positive, blue = missed (false negative)."""
    rgba = np.zeros(pred.shape + (4,))
    rgba[pred & ref] = (0.1, 0.8, 0.1, 0.5)
    rgba[pred & ~ref] = (0.9, 0.1, 0.1, 0.7)
    rgba[~pred & ref] = (0.1, 0.3, 1.0, 0.7)
    return rgba

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
show_view(axes[0], slice_hu.T, None, 1, "upper", "image (lung window)")
show_view(axes[1], slice_hu.T, lab[:, :, k_lung].T, 1, "upper", "reference mask")
axes[2].imshow(apply_window(slice_hu.T, -600, 1500), cmap="gray", vmin=0, vmax=1)
axes[2].imshow(error_overlay(pred, slice_ref).transpose(1, 0, 2))
axes[2].set_title("threshold < -400: green ok | red false positive | blue missed")
axes[2].axis("off")
plt.tight_layout()

# %% [markdown]
# What goes wrong, on a **real** chest CT (the demo data are drawings, but the problems are the same):
# * **Air around the patient** is darker than -400 HU too: a huge red false positive region (and the padding outside the scanner field of view).
# * **Trachea and main bronchi** are air, too, but our reference excludes them (the dataset convention: trachea = background).
# * **Gas in the bowel or stomach** appears in the lower slices of a chest CT: more false positives.
# * **Vessels, airway walls and nodules** inside the lungs are brighter than -400 HU: holes in the mask (blue).
# * **Diseased lungs** (consolidation, pleural effusion, pneumonia, ground glass) are *not* dark: the threshold misses them exactly when the segmentation matters most.
#
# > 🧠 **Question:** a prediction made only of zeros would have, for a lung slice, about 85-90% *pixel accuracy*.
# > Why is Dice a better metric than accuracy for segmentation? (Compute both for the all-zeros mask below.)

# %%
empty = np.zeros_like(slice_ref)
print(f"all-zeros prediction -> accuracy {(empty == slice_ref).mean():.3f} | Dice {dice_score(empty, slice_ref):.3f}")

# %% [markdown] tags=["solution-only"]
# **Answer.** The background dominates the image, so accuracy is high even when the organ is missed completely. Dice
# only looks at the overlap with the (small) foreground and ignores true negatives, so the empty prediction scores 0.

# %% [markdown]
# ### 🔍 Explore
# * Sweep the threshold from -950 to -200 (cell below): is there a *good* value? Is it good for every slice?
# * Display the error map of a slice near the apex or the base of the lungs: what changes?
# * Re-compute the baseline after clipping `slice_hu` to a window: does it change anything? Why (not)?

# %%
thresholds = np.arange(-950, -150, 50)
scores = [dice_score(threshold_baseline(slice_hu, t), slice_ref) for t in thresholds]
n_eval = 6 if FAST else 20
z_eval = np.unique(np.linspace(0, vol.shape[2] - 1, n_eval).astype(int))
dice_z = {int(k): dice_score(threshold_baseline(vol[:, :, k].astype(np.float32)), np.isin(lab[:, :, k], (1, 2)))
          for k in z_eval if np.isin(lab[:, :, k], (1, 2)).any()}

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
axes[0].plot(thresholds, scores, "o-"); axes[0].set(xlabel="threshold (HU)", ylabel="Dice", title="Dice vs threshold (one slice)")
axes[1].plot(list(dice_z), list(dice_z.values()), "o-"); axes[1].set(xlabel="slice index k (inferior -> superior)", ylabel="Dice",
                                                                    title="Dice of the threshold < -400 along the scan")
for ax in axes:
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
plt.tight_layout()

# %% [markdown]
# So thresholding is quick but brittle: it depends on the slice level, on the patient and on the scanner, and it
# knows nothing about *anatomy*. This is what we want a learned model to do better.
#
# **We will re-use `threshold_baseline` and `dice_score` on day 3** as the "classical baseline" the U-Net has to beat
# (notebooks are self-contained, so copy the two functions when you need them).

# %% [markdown]
# ### 🚀 Challenge (additional) - rescue the classical pipeline with `scipy.ndimage`
# Improve `threshold_baseline` using only **morphology and connected components**, then measure the Dice again:
# 1. Label the connected components of the thresholded mask with `ndi.label`.
# 2. Drop the components that touch the image border: that is the air around the patient.
# 3. Keep only the `n_components` largest remaining ones (two lungs; this also removes the trachea and small bowel gas).
# 4. Fill the holes (vessels!) with `ndi.binary_fill_holes`.
#
# Is the new Dice better? On which slices is it still bad, and why? (The code runs as it is, it just returns the plain threshold until you improve it.)
#
# <details>
# <summary>💡 Hint 1</summary>
#
# `labeled, n = ndi.label(mask)` gives an integer image with one id per component. Border ids: `np.unique` of the first and last rows and columns of `labeled`. `ndi.sum_labels(mask, labeled, index=ids)` gives component sizes.
#
# </details>
#
# <details>
# <summary>💡 Hint 2</summary>
#
# `np.isin(labeled, kept_ids)` rebuilds a mask from a list of component ids.
#
# </details>

# %%
def improved_baseline(hu, threshold=-400, n_components=2):
    """Threshold + connected components + hole filling."""
    #@solution
    air = threshold_baseline(hu, threshold)
    labeled, n = ndi.label(air)
    if n == 0:
        return air
    border_ids = np.unique(np.concatenate([labeled[0], labeled[-1], labeled[:, 0], labeled[:, -1]]))
    ids = np.arange(1, n + 1)
    sizes = ndi.sum_labels(air, labeled, index=ids)
    kept = [i for i in ids[np.argsort(sizes)[::-1]] if i not in border_ids][:n_components]
    return ndi.binary_fill_holes(np.isin(labeled, kept))
    #@student
    #| mask = threshold_baseline(hu, threshold)
    #| # TODO: label components, drop the ones touching the border, keep the largest n_components, fill holes
    #| return mask
    #@end


pred_better = improved_baseline(slice_hu)
print(f"Dice: plain threshold {dice_score(pred, slice_ref):.3f} -> improved {dice_score(pred_better, slice_ref):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, p, t in zip(axes, (pred, pred_better), ("plain threshold", "improved")):
    ax.imshow(apply_window(slice_hu.T, -600, 1500), cmap="gray", vmin=0, vmax=1)
    ax.imshow(error_overlay(p, slice_ref).transpose(1, 0, 2))
    ax.set_title(t); ax.axis("off")
plt.tight_layout()

# %% [markdown] tags=["solution-only"]
# **Discussion.** Dice goes up a lot, because the air around the patient (the main error) disappears. Remaining problems on real CT:
# in the upper slices the trachea can be as large as the lungs (it may be kept instead of a lung); in the lower slices
# the two lungs may touch each other (the two lungs become *one* component, so the second kept component is a random small
# one) or they are connected to the bowel gas; pathologies (pleural effusion, consolidation) are not dark at all; patients with
# the arms or table in the scan, or a body that touches the border of the image, break the "touches the border" rule.
# Classical pipelines need many such ad-hoc rules, and each rule fails on some case: this is the motivation for learning from data.

# %% [markdown]
# ### 🚀 Challenge (additional) - lung area in cm²
# The dataset metadata have a column `lung_area_mm2`: the physical area of the lungs in the slice. With the voxel size
# you can compute it from any mask: `area = number of pixels * sx * sy`.
# Compute the area (in cm²) of the reference mask and of the improved prediction on this slice. Why is the area in *pixels*
# not a good quantity to compare between two patients? (Day 3 will use this for per-case analysis.)

# %%
#@solution
area_ref_cm2 = slice_ref.sum() * sx * sy / 100
area_pred_cm2 = pred_better.sum() * sx * sy / 100
#@student
#| area_ref_cm2 = 0.0   # TODO: pixels * sx * sy gives mm2; 1 cm2 = 100 mm2
#| area_pred_cm2 = 0.0  # TODO
#@end
print(f"lung area on this slice: reference {area_ref_cm2:.1f} cm2 | improved baseline {area_pred_cm2:.1f} cm2")

# %% [markdown] tags=["solution-only"]
# **Answer.** The same pixel count corresponds to different physical areas when the pixel size differs (0.6 vs 0.9 mm
# is a factor 2.25 in area). Areas and volumes are clinical quantities (e.g. for lung volume or emphysema
# quantification), so they must be computed in mm² or cm² using the spacing, never in pixels.

# %% [markdown]
# ### 🌶️ Going further (optional)
# * **Interactive viewer.** With `ipywidgets` (`pip install ipywidgets`) build a slider over `k` that redraws the axial view with the window as a dropdown: `ipywidgets.interact(show_slice, k=(0, vol.shape[2]-1), window=list(WINDOWS))`.
# * **De-identify a DICOM series.** With pydicom, blank `PatientName`, `PatientBirthDate`, `InstitutionName`, `ReferringPhysicianName`, generate new `StudyInstanceUID`/`SeriesInstanceUID` (consistent across the series!) and save with `ds.save_as`. What else would a real de-identification profile remove? (see the DICOM PS3.15 confidentiality profiles and `dicom-anonymizer`/`CTP`).
# * **DICOM to NIfTI with SimpleITK.** `reader = sitk.ImageSeriesReader(); reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(dicom_dir))); img = reader.Execute(); sitk.WriteImage(img, str(WORK_DIR / "from_dicom.nii.gz"))`. Then reload it with nibabel and check that `aff2axcodes` and the affine match our NIfTI (ITK handles the LPS -> RAS sign flip for you; `dcm2niix` is the standard command-line tool).
# * **Look at the data in a real viewer.** Download [3D Slicer](https://slicer.org) or [ITK-SNAP](https://www.itksnap.org/), open the NIfTI volume and its label, and find the same slice you plotted here. Which coordinates does the viewer show for the cursor (RAS or LPS)?

# %% [markdown]
# ## 10. Recap: what to remember
# * A CT slice is a **matrix of HU**: air -1000, lung about -800, water 0, soft tissue 20-80, bone > 300. MRI values are not calibrated.
# * Pixels are only part of the story: **spacing, position, orientation and metadata** matter (areas in mm², anisotropic voxels, laterality).
# * DICOM stores *stored values*: apply `slope * stored + intercept`; sort slices by **position**, not by file name.
# * SimpleITK arrays are `[z, y, x]` but its geometry is `(x, y, z)`; our NIfTI arrays are `[x, y, z]` in **LPS** index order, and `img.T` shows the radiological view.
# * The **affine** maps voxel indices to mm; **LPS** (DICOM/ITK) and **RAS** (NIfTI/Slicer/MONAI) differ by the sign of x and y. Always check the anatomy.
# * Windowing selects the HU range you look at (lung, mediastinum, bone): a display choice that becomes a modelling choice for a network.
# * **Dice** measures the overlap, accuracy is misleading when the background dominates. A plain threshold gives a poor baseline (outside air, trachea, vessels, pathology): this is why we learn the segmentation.
#
# **Next:** notebook 02 explores the course dataset (one slice per patient) and builds the MONAI pre-processing pipeline.
#
# > 📚 **Read more:**
# > * [pydicom user guide](https://pydicom.github.io/pydicom/stable/) and the [DICOM tag browser](https://dicom.innolitics.com/ciods)
# > * [SimpleITK notebooks](https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks) and [nibabel coordinate systems](https://nipy.org/nibabel/coordinate_systems.html)
# > * [Radiopaedia: Hounsfield unit](https://radiopaedia.org/articles/hounsfield-unit) and [CT windowing](https://radiopaedia.org/articles/windowing-ct)
# > * Setio et al., *Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge*, Medical Image Analysis 42 (2017) 1-13, [doi:10.1016/j.media.2017.06.015](https://doi.org/10.1016/j.media.2017.06.015) - [challenge site](https://luna16.grand-challenge.org/)
