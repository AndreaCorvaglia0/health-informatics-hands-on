# Technical setup (day 1)

Two environments are used in the course:

| | Where | For what | When |
|---|---|---|---|
| **Cloud notebook with a free GPU** | Kaggle (plan B: Google Colab) | notebooks 01-06, training | days 1-4 |
| **Your laptop** | VS Code + Python + Git (+ Docker) | the web app | day 5 (install before!) |

Legend: ✏️ mandatory - 🚀 additional - 🌶️ optional.

## Before day 1 (15 minutes, at home)

1. ✏️ **Kaggle account**: <https://www.kaggle.com> -> Register. Then *Settings -> Phone
   verification*: without a verified phone number Kaggle does not give you the GPU nor
   internet access inside notebooks. Do it at home: the SMS sometimes takes a while.
2. ✏️ **GitHub account**: <https://github.com>. You will need it on day 5 (and to sign in
   to Streamlit Community Cloud).
3. ✏️ **Google account** (you probably have one): plan B for notebooks (Colab).

## Day 1 in class - the notebook environment

### ✏️ Kaggle (primary)

1. Open the course notebook link given by the teachers and press **Copy & Edit**
   (or: *Create -> New Notebook -> File -> Import Notebook* and upload the `.ipynb`).
2. Right panel -> **Input -> Add Input** -> search the course dataset
   (the teachers give you the name) -> **+**. It appears under `/kaggle/input/...`.
3. Right panel -> **Session options**: *Internet* **on**; *Accelerator* **GPU** when
   the notebook says so (notebooks 01-02 do not need it: keep it off, the weekly GPU
   quota is limited to about 30 hours).
4. Run the first two code cells (setup + configuration). You should read
   `data: /kaggle/input/...` and no error.

<details>
<summary>💡 It says "Dataset not found"</summary>

The dataset is not attached to *this* notebook: repeat step 2, then re-run the
configuration cell. Every new notebook needs its own "Add Input".

</details>

<details>
<summary>💡 Where do my files go? Will I lose my trained model?</summary>

Everything you write goes to `/kaggle/working`. It survives while the session is
alive. To keep it: download it from the *Output* panel, or press **Save Version**
(the outputs of that version stay attached to the notebook and can be added as
*Input* to another notebook). From day 2 on you will produce a `*.zip` with your
model: download it every time.

</details>

### Plan B: Google Colab

Open `https://colab.research.google.com/github/<owner>/<repo>/blob/main/notebooks/<notebook>.ipynb`
(the teachers give you the exact link), *Runtime -> Change runtime type -> GPU*.
Upload the dataset zip in the Files panel (or mount Drive), unzip it, and set in a
cell at the top: `import os; os.environ["HI_DATA_DIR"] = "/content/luna16-lung-slices"`.
Colab deletes your files when the session ends: download your model zip.

### How the notebooks work

* Run cells top to bottom. ✏️ exercises have blanks `____` to replace; the cell right
  after an exercise checks your answer (`✅`).
* Stuck? Open the **💡 Hint** blocks under the exercise, one at a time. Still stuck
  after 5 minutes? Call a teacher: that is what we are here for.
* Done early? 🚀 challenges and 🌶️ ideas are there for you. They are not needed to continue.
* `FAST = True` in the configuration cell runs everything tiny and quick: use it to
  debug, set it back to `False` for real results.

## Before day 5 - your laptop

Install (ask for help on day 4 if something fails; do not arrive on day 5 without it):

1. ✏️ **Python 3.12** from <https://www.python.org/downloads/> (Windows: tick
   *"Add python.exe to PATH"*). Check in a terminal: `python --version`
   (macOS/Linux: `python3 --version`).
2. ✏️ **VS Code** from <https://code.visualstudio.com/> + the *Python* extension
   (Extensions panel -> search "Python" by Microsoft).
3. ✏️ **Git** from <https://git-scm.com/downloads>. Check: `git --version`. Then once:
   `git config --global user.name "Your Name"` and
   `git config --global user.email "you@example.com"`.
4. 🚀 **Docker Desktop** from <https://www.docker.com/products/docker-desktop/>
   (large download; on Windows it needs WSL 2). Check: `docker run hello-world`.
   If it does not work on your laptop you will pair with a classmate for that part.
5. ✏️ Sign in to **Streamlit Community Cloud** at <https://share.streamlit.io> with
   your GitHub account.

Self-check (terminal inside VS Code, *Terminal -> New Terminal*):

```bash
python --version      # 3.12.x  (python3 on macOS/Linux)
git --version
code --version
docker --version      # optional
```

Bring to day 5: the `app_starter_kit.zip` you create at the end of notebook 06.
