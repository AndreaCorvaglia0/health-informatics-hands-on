# Day 5 - From model to web app (VS Code, GitHub, Streamlit, Docker)

Yesterday you built `inference.py`: one function that turns a raw CT slice into a lung mask. Today we wrap it in a
**web app**, test it, put it on **GitHub**, deploy it with **Streamlit Community Cloud**, and package it in a **Docker**
container. About 4.5 hours of class time.

> ⚠️ **Research and education only.** The app you build is **not a medical device**. Do not use it for diagnosis or
> for any clinical decision, and never upload real patient data to it (see the checklist at the end).

| Marker | Meaning |
|---|---|
| ✏️ | **Mandatory**: everybody does it. Hints are hidden under 💡. |
| 🚀 | Additional, for those who finish early. |
| 🌶️ | Optional and open-ended. |
| 🧠 | A question to think about. |

Folders in this repository:

```
app/
  starter/     <- YOU START HERE (copy it to your own project folder, see section 1)
  README.md    this guide
```

**Suggested schedule**

| Time | What |
|---|---|
| 0:00 - 0:35 | 1. Local setup in VS Code (venv, packages, unzip the kit) |
| 0:35 - 1:50 | 2. Run Streamlit and code steps 1-5 (mandatory) |
| 1:50 - 2:30 | Steps 6-7 (🚀) and 3. tests |
| 2:30 - 3:00 | 4. Git and GitHub |
| 3:00 - 3:40 | 5. Deploy on Streamlit Community Cloud |
| 3:40 - 4:20 | 6. Docker |
| 4:20 - 4:30 | Prepare the show and tell (section 7); the presentations are in the final session |

---

## 1. Local setup in VS Code

### 1.1 ✏️ Open the project folder
1. Copy the folder `app/starter/` to a place of your own (for example `~/lung-app`): this will become **your** Git repository.
2. In VS Code: **File > Open Folder...** and choose it. (Always open the *folder*, not a single file: VS Code then
   knows where your project, your virtual environment and your tests are.)
3. Open the integrated terminal: **Terminal > New Terminal** (shortcut `` Ctrl+` ``). It starts in the project folder.

### 1.2 ✏️ Create and activate a virtual environment
A **virtual environment** (venv) is a private copy of Python with its own packages, so that this project does not
interfere with others (and you can list exactly what it needs: `requirements.txt`).

```bash
python -m venv .venv
```

Activate it (do this in every new terminal):

| System | Command |
|---|---|
| Windows, PowerShell | `.venv\Scripts\Activate.ps1` (if it is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`) |
| Windows, cmd | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

The prompt now starts with `(.venv)`. Use Python **3.11 or 3.12** (`python --version`); on some systems the command is
`python3` or `py -3.12`.

<details>
<summary>🚀 The faster alternative: <code>uv</code></summary>

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager. If it is installed:

```bash
uv venv --python 3.12
# activate as in the table above, then:
uv pip install -r requirements.txt
```

It reads the same `requirements.txt`. Everything else in this guide is identical.

</details>

### 1.3 ✏️ Select the interpreter
Press `Ctrl+Shift+P` (macOS: `Cmd+Shift+P`), type **Python: Select Interpreter**, and choose the one inside `.venv`.
VS Code now runs, debugs and tests with it (bottom-right corner shows the Python version). Install the Microsoft
**Python** extension if VS Code asks.

### 1.4 ✏️ Install the packages
```bash
pip install -r requirements.txt
pip install pytest
```
Open `requirements.txt`: the first line `--extra-index-url https://download.pytorch.org/whl/cpu` makes pip choose the
**CPU-only build of PyTorch**. 🧠 *Why?* The default Linux wheel bundles CUDA libraries and is several GB; our web
host has no GPU anyway, and a free host has little disk and memory. The CPU wheel is around 200 MB. The installation
takes a few minutes.

### 1.5 ✏️ Unzip the starter kit from day 4
Copy `app_starter_kit.zip` (made in notebook 06) into the project folder and unzip it **there**
(Explorer: right click > *Extract All*, or `unzip app_starter_kit.zip`). You should now have:

```
models/<your-model>/{model.pt, config.json, MODEL_CARD.md}
samples/*.nii.gz, *_label.nii.gz, one .dcm
tests/test_inference.py
inference.py              (the same file as in the starter: it is OK to overwrite it)
```
If you could not finish notebook 06, ask a teacher for the kit.

---

## 2. Run the app and code it step by step

### 2.1 ✏️ Run it
```bash
streamlit run app.py
```
The browser opens `http://localhost:8501`. The starter already runs: it tells you which step to do next. Leave it
running; when you save `app.py`, Streamlit offers **Rerun** (or **Always rerun**) in the top-right corner.
Stop it with `Ctrl+C` in the terminal.

### 2.2 How Streamlit works (5 lines)
1. Your `app.py` is a normal Python script, run **from top to bottom**. Each `st.something(...)` draws an element.
2. Whenever the user touches **any** widget (a slider, an upload...) Streamlit runs the **whole script again**.
3. A widget call returns its current value: `t = st.slider("Threshold", 0.0, 1.0, 0.5)` gives you a number.
4. Everything computed in the script is therefore recomputed at each interaction, unless you **cache** it.
5. `@st.cache_resource` keeps *one shared object* (a loaded model) alive across reruns and users; `@st.cache_data`
   caches *return values* of functions. That is why STEP 3 exists.

### 2.3 The steps
Open `app.py` and follow the numbered `STEP` / `TODO` comments. **The app runs after every step**: save, look at the
browser, then continue. The helpers at the top (drawing masks, reading files) are given; the real logic is in
`inference.py`.

| Step | Level | What you do | You see |
|---|---|---|---|
| 1 | ✏️ | Title, notice, **file uploader** for `.nii/.nii.gz/.dcm` | an upload box |
| 2 | ✏️ | Read the slice (`inf.pick_slice`) and **show the CT** with `st.image` | the CT in radiological view |
| 3 | ✏️ | **Load the model once** with `@st.cache_resource`; show its info in the sidebar | model name, Dice |
| 4 | ✏️ | **Predict** with `inf.segment_slice`; three columns: CT, overlay, probability | the lung contour |
| 5 | ✏️ | **Lung area** in cm², inference time; **Dice and error map** vs ground truth | numbers + error map |
| 6 | 🚀 | Sidebar controls: window presets (lung/mediastinum/bone/custom), threshold, post-processing, opacity | a live-tunable app |
| 7 | 🚀 | **Model selector** (several folders in `models/`) and **download** the mask as NIfTI | download button |
| 8 | 🌶️ | **Compare two models** (Dice between them); **slice slider** for 3D volumes; model card in an expander | your own ideas |

<details>
<summary>💡 Hint: step 3, how do I know caching works?</summary>

The starter prints `loading model from ...` in the **terminal** each time the function body runs. Without the decorator
it prints at every click; with `@st.cache_resource` only once.

</details>

<details>
<summary>💡 Hint: step 5, where does the area come from?</summary>

`res.measurements["area_cm2"]` is computed in `inference.py` from the mask and the **pixel spacing** read from the file
(NIfTI header or DICOM `PixelSpacing`). If the spacing were wrong, the picture would look the same but the number
would not: always check units.

</details>

<details>
<summary>💡 Hint: the display window is not the model window</summary>

In step 6 the window presets change only how the image is **displayed** (`to_display`). The model always receives
the HU window stored in its `config.json`. Move the window slider and watch: the contour does not change.

</details>

> 🧠 **Question (step 5):** compare the predicted mask with the ground truth of a sample. Where are the red (false
> positive) and blue (false negative) pixels? Are they errors of the model, or of the reference (LUNA16 masks are
> *silver standard*, made by an algorithm)?

> ⚠️ **Pitfall:** the last version of this course's app re-implemented the preprocessing by hand (different resize,
> different normalisation) and silently gave worse masks. Here the app **never** preprocesses anything itself: it only
> calls `inf.segment_slice`. Keep it that way.

### 🌶️ Ideas for step 8
* A 3D volume (`samples/` or the `demo/volume` file): add the slice slider, then compute the area for **all** slices
  and plot the curve with `st.line_chart`.
* Show `MODEL_CARD.md` in an `st.expander`; warn when the predicted area is tiny ("is there a lung in this slice?").
* Classification or embedding add-ons from the optional notebooks.

---

## 3. Tests with `pytest`

A test is a small function that fails loudly if something that used to work breaks. They protect you from
**training-serving skew** and from "I only changed one line" accidents.

### 3.1 ✏️ Run the tests that came with the kit
```bash
python -m pytest -q
```
`tests/test_inference.py` loads the model in `models/` (or the folder in the environment variable `HI_MODEL_DIR`) and
checks shape, small mask for an image without lungs, determinism. All green? Good. If it says *skipped*, there is no
model in `models/`.

### 3.2 ✏️ Add a test of your own
Open `tests/test_inference.py` and add one test at the end, e.g. that `inf.preprocess(hu, cfg)` always returns values
in `[0, 1]`, or that `inf.measure` returns the right area for a square mask of known size.

<details>
<summary>💡 Hint</summary>

```python
def test_measure_known_area():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:10, :10] = 1                      # 100 pixels
    assert inf.measure(mask, (1.0, 1.0))["area_cm2"] == pytest.approx(1.0)   # 100 mm2 = 1 cm2
```

</details>

### 🚀 Test the app itself
Streamlit can run your script without a browser. The teachers' `tests/test_app.py` (ask for it when your app is finished) shows how
(`streamlit.testing.v1.AppTest`). Read it and write one test for your app: "the app starts without exception".

---

## 4. Git and GitHub (just what we need)

Git records the history of your files; **GitHub** hosts a copy online. Streamlit Community Cloud will read your
code from there, so the repository is the *source of the deployment*.

### 4.1 ✏️ What must NOT be committed
Open `.gitignore` (given). It excludes `.venv/` (huge, rebuildable from `requirements.txt`), `__pycache__/`, and
secrets (`.env`). Rules of thumb: **never commit** virtual environments, passwords/API keys, large datasets, and
**any real patient data or real DICOM** (headers contain names and IDs). Our small models and synthetic/public sample
slices are fine.

### 4.2 ✏️ Create the repository and commit
In the terminal, inside the project folder:

```bash
git init
git config user.name "Your Name"          # once per machine (add --global to set it everywhere)
git config user.email "you@example.com"
git status                                # what would be committed? .venv must NOT appear
git add .
git commit -m "Lung segmentation app: first version"
git ls-files                              # the files that are really in the repository
```
Alternative with a GUI: the **Source Control** panel in VS Code (the branch icon in the left bar): *Initialize
Repository*, write a message, click the ✓ *Commit* button.

### 4.3 ✏️ Push to GitHub
1. On github.com (log in or sign up): **New repository**. Name it e.g. `lung-segmentation-app`, **Public** (the simplest
   for Community Cloud), and do **not** add a README/.gitignore there (your folder already has files).
2. GitHub shows you the commands; they look like:

```bash
git branch -M main
git remote add origin https://github.com/<your-user>/lung-segmentation-app.git
git push -u origin main
```
GitHub does not accept account passwords on the command line: the first push opens a browser/VS Code sign-in or asks
for a *personal access token*. Refresh the repository page: your files are there.

### 🧠 About the model file
GitHub rejects any single file above **100 MB**. Our model is about 6-7 MB, so it can live in the repository. For
bigger models the options are: keep the model smaller, **Git LFS** (large file storage; check the current quota of your
GitHub plan), or upload the weights as a release asset / on a model hub and download them at start-up
(inside the `@st.cache_resource` loader).

### 🚀 Make a change and push it again
Change something visible (a title), then `git add . && git commit -m "..." && git push`. Watch the deployed app
update itself in section 5.

---

## 5. Deploy on Streamlit Community Cloud

Free hosting for public Streamlit apps, fed from GitHub. Facts below come from the Streamlit documentation
(<https://docs.streamlit.io/deploy/streamlit-community-cloud>) as checked on 18 Nov 2026; **menu labels and limits
may change**, so read what you see on screen.

### 5.1 ✏️ Deploy
1. Go to <https://share.streamlit.io> and sign in **with GitHub** (authorise access to your repositories).
2. Click **Create app** / **New app** (label may change) and choose *Deploy a public app from GitHub*.
3. Fill in: **repository** (`<your-user>/lung-segmentation-app`), **branch** (`main`), **main file path** (`app.py`).
4. Open **Advanced settings** and choose the **Python version** explicitly: **3.12** (or 3.11). It cannot be changed
   afterwards without deleting and redeploying the app.
5. Click **Deploy**. Watch the **build logs** ("Manage app" at the bottom-right, or the log panel): the platform
   installs `requirements.txt`, then starts `app.py`. The first build takes several minutes (PyTorch).
6. You get a public URL like `https://<something>.streamlit.app`. Open it, test a sample. 🎉

### 5.2 Things to know
* **Resources are limited and shared**: the documentation lists roughly 2.7 GB of memory and up to 2 CPU cores per
  app (approximate, subject to change). PyTorch + MONAI use a good part of that memory: load **one model**, never
  whole CT volumes, and keep `@st.cache_resource`.
* **Sleep**: apps without traffic go to sleep (the docs say after about 12 hours); visitors can wake them with a
  button and the first request is slower (imports + model loading).
* **Updates**: each `git push` to the deployed branch redeploys the app.

### 5.3 Common failures and what to look at

| Symptom in the build / app log | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'x'` | package missing from `requirements.txt` | add it, commit, push |
| `FileNotFoundError: models/...` or "No model found" | model folder not committed (`.gitignore`?) or wrong path | `git ls-files`; use paths relative to `app.py` (`Path(__file__).parent`) |
| Build is very slow / disk or memory error while installing torch | the large CUDA wheel was installed | keep the CPU `--extra-index-url` line in `requirements.txt` (see note below) |
| "This app has gone over its resource limits" | too much memory (several models, volumes in memory) | smaller inputs, one model at a time, caching |
| `torch`/`monai` has no wheel for this Python | wrong Python version in Advanced settings | redeploy with 3.11 or 3.12 |
| App shows the old version | pushed to another branch / not pushed | check the branch in the app settings and `git log` |

> The `--extra-index-url` line works with pip and uv on a laptop. Whether Community Cloud honours it is **not
> confirmed by its documentation**: if the build fails because of it, check the log and ask a teacher. Fallbacks: remove
> the line (the large wheel may still install, but it is slow), or use an ONNX model with `onnxruntime` instead of
> PyTorch (see notebook 06, 🚀 section).

---

## 6. Docker, practical

### 6.1 What it is (in 5 lines)
* An **image** is a frozen, self-contained recipe result: an operating system layer + Python + your packages + your
  code. A **container** is a running instance of an image.
* Same image = same behaviour on your laptop, on a colleague's laptop and on a server: no "works on my machine".
* Images are built from a **Dockerfile**, layer by layer; unchanged layers are cached.
* A container is isolated: it only sees what you put in it and the ports you open.
* 🧠 *Why do hospitals care?* **Reproducibility** (the exact software that was validated is what runs), **on-premise
  deployment** (patient data never leaves the hospital network: you bring the model to the data), easy rollback and
  audit of versions, and IT departments already know how to run containers.

### 6.2 ✏️ Read the Dockerfile, line by line
`Dockerfile` and `.dockerignore` are already in your project folder (next to `app.py`).

| Line | What it does |
|---|---|
| `FROM python:3.12-slim` | start from a small Debian image with Python 3.12 |
| `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 ...` | no `.pyc` files, logs shown immediately, pip does not keep a download cache (smaller image) |
| `RUN useradd ... appuser` | create a normal user: do not run as root inside the container |
| `WORKDIR /app` | the folder where the following commands run |
| `COPY requirements.txt .` + `RUN pip install -r requirements.txt` | install dependencies **before** copying the code: this slow layer stays cached until `requirements.txt` changes |
| `COPY --chown=appuser:appuser . .` + `USER appuser` | copy the code (minus what `.dockerignore` lists) and switch to the normal user |
| `EXPOSE 8501` | documents the port Streamlit listens on |
| `HEALTHCHECK ...` | Docker asks Streamlit's health endpoint if the app is alive |
| `CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", ...]` | what runs when the container starts. `0.0.0.0` = reachable from outside the container (`localhost` would not be) |

🧠 *Why is `COPY requirements.txt` before `COPY . .`?* Try changing one line of `app.py` and rebuilding: which steps
are re-run?

### 6.3 ✏️ Build and run
Install Docker Desktop (or use the lab machine) and start it. In the project folder:

```bash
docker build -t lung-app .
docker run --rm -p 8501:8501 lung-app
```
Open <http://localhost:8501>. `-p 8501:8501` maps *host port : container port*; `--rm` removes the container when it
stops (`Ctrl+C`). Useful: `docker images`, `docker ps`, `docker logs <container>`.

### 🚀 Go further
* Run a **classmate's image**: `docker save lung-app -o lung-app.tar` -> send the file -> `docker load -i lung-app.tar`
  -> `docker run ...`. What if their model is different? (Everything is inside the image.)
* **Push to a registry** (e.g. Docker Hub): `docker tag lung-app <user>/lung-app:1.0`, `docker login`,
  `docker push <user>/lung-app:1.0`.
* 🌶️ **Hugging Face Spaces with the Docker SDK**: needs `sdk: docker` and `app_port` in the README's YAML header, and
  the app must listen on that port (default 7860). *Availability for free accounts has been changing: check with a
  fresh account before relying on it.*

---

## 7. Show and tell (final session)

Each pair/student presents for about 5 minutes. **Checklist**

- [ ] A **public URL** (or a screen recording if the deploy failed) of the app, and the GitHub repository.
- [ ] The **model card** (`MODEL_CARD.md`): intended use, NOT intended use, metrics with *n*, limitations.
- [ ] **One failure case** that you found and can explain (which image, what went wrong, hypothesis why: out-of-domain
  slice? silver-standard label? orientation? thickness?).
- [ ] **One number** you trust and one you do not (and why).
- [ ] **What you would do next** (more data, other model, 3D, validation on local data, regulatory path).

### Responsible-use checklist
- [ ] **No real patient data** in the repository, in samples, or in uploads to a public app. Public apps are public.
- [ ] **DICOM headers contain PHI** (name, ID, birth date, institution): de-identify before sharing anything.
- [ ] The notice **"research/education only, not a medical device"** is visible in the app and in the model card.
- [ ] No secrets (tokens, passwords) in the code or in Git history.
- [ ] Know the **licence** of the data and of the model before redistributing them (LUNA16/LIDC-IDRI terms).
- [ ] If anyone wanted to use this clinically: local validation, risk analysis, regulatory approval, privacy
  assessment. We are far from that.

---

## Troubleshooting

| Problem | Try |
|---|---|
| `python` / `pip` not found | install Python 3.11/3.12; on Windows try `py -3.12`; on macOS/Linux `python3` |
| PowerShell refuses to activate the venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use cmd |
| VS Code runs the wrong Python | `Ctrl+Shift+P` > *Python: Select Interpreter* > the one in `.venv`; open a *new* terminal |
| `streamlit: command not found` | the venv is not active (no `(.venv)` in the prompt), or `pip install -r requirements.txt` failed: re-run it |
| `pip install` of torch is huge/slow | check that the first line of `requirements.txt` is `--extra-index-url https://download.pytorch.org/whl/cpu` |
| "No model found" in the app | unzip `app_starter_kit.zip` in the project root; each model needs `config.json` **and** `model.pt` |
| Error "weights_only" / "Unsupported global" when loading | the file is not a plain `state_dict`; re-export it in notebook 06 |
| The mask looks flipped or the area is absurd | wrong orientation or spacing: the app warns for NIfTI that are not LPS; check the file with notebook 01 |
| "values look like an 8-bit image" warning | you uploaded a PNG-derived or normalised file; the model needs Hounsfield units |
| Port 8501 already in use | stop the other Streamlit (`Ctrl+C`) or `streamlit run app.py --server.port 8502` |
| App reruns and the model reloads at every click | missing `@st.cache_resource` on the loader |
| `git push` rejected / asks for a password | use the browser/VS Code sign-in or a personal access token; never share it |
| `git add .` shows `.venv` | the `.gitignore` is missing or misspelled; `git rm -r --cached .venv` |
| File rejected by GitHub (> 100 MB) | do not commit big files; remove it from the commit (`git rm --cached`) and see section 4 |
| Streamlit Cloud build fails | read the log from the bottom up; see the table in section 5.3 |
| `docker: command not found` / daemon not running | install/start Docker Desktop and retry |
| Container starts but the browser shows nothing | forgot `-p 8501:8501`, or the server is not on `0.0.0.0` |
