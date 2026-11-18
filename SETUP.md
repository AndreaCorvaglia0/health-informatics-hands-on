# Setup

There are three moments. Each one says exactly what to do and how to check that it worked.

| Part | When | Where | Time |
|---|---|---|---|
| A. Accounts | at home, before day 1 | browser | 15 min |
| B. Kaggle notebook | day 1, in class | browser | 20 min |
| C. Your laptop | at home, before day 5 | your computer | 45 min |

Days 1 to 4 run entirely in the browser, on Kaggle. Nothing to install.
Only day 5 (the web app) uses programs installed on your laptop.

---

## A. Accounts (before day 1)

### A1. Kaggle account

1. Go to https://www.kaggle.com and click **Register**. Use your university e-mail.
2. Confirm the e-mail address (Kaggle sends you a message).

### A2. Verify your phone number on Kaggle

Without this step Kaggle does not give you the GPU, and the course cannot be followed.

1. Click your picture (top right) and open **Settings**.
2. Find **Phone verification** and click **Verify**.
3. Enter your mobile number and the code you receive by SMS.

Check: in Settings, the phone number shows as verified.

### A3. GitHub account

1. Go to https://github.com and click **Sign up**.
2. Choose a username you would not mind showing in a CV.

You will use it on day 5.

---

## B. Kaggle notebook (day 1, in class)

We do this together. The steps are here so that you can repeat them for each notebook of the course.

### B1. Open a course notebook on Kaggle

1. Download the notebook file from the course material (for example `00_environment_check.ipynb`).
2. On Kaggle click **Create** (top left) and then **New Notebook**.
3. In the notebook menu choose **File** and then **Import Notebook**.
4. Upload the file you downloaded and click **Import**.

### B2. Attach the dataset

1. In the right panel open **Input** and click **Add Input**.
2. Search `luna16-lung-slices` (the teachers tell you the exact name).
3. Click the **+** next to the dataset.

Check: in the Input panel you see the dataset with the folders `images` and `labels`.

You must do this for every new notebook.

### B3. Settings of the notebook

In the right panel open **Session options** (or **Settings**):

1. **Internet**: On.
2. **Accelerator**: `None` for notebooks 00, 01 and 02. `GPU T4 x2` for notebooks 03, 04 and 05.

Kaggle gives about 30 hours of GPU per week. Turn the GPU on only when the notebook says so.

### B4. Run the environment check

1. Open `00_environment_check.ipynb` on Kaggle (steps B1 to B3).
2. Run the cells one by one with `Shift + Enter`.

Check: the notebook prints `Libraries: OK` and `Dataset: OK`, and shows a CT image at the end.
If you turned the GPU on, it also prints `GPU: OK`.

### B5. Two rules for working on Kaggle

- **Save**: Kaggle saves the notebook while you work. To keep a permanent copy click **Save Version**.
- **Files you create are temporary**. Everything written in `/kaggle/working` (for example your trained model)
  is deleted when the session ends. Download what you need from the **Output** panel before you close.

### If something goes wrong

| Problem | What to do |
|---|---|
| `No module named 'monai'` | Internet is Off. Turn it On (B3) and run the first cell again. |
| `/kaggle/input/luna16-lung-slices not found` | The dataset is not attached to this notebook (B2). |
| The Accelerator menu is disabled | The phone number is not verified (A2). |
| "You have used your GPU quota" | Work with a classmate until the quota resets, and tell a teacher. |
| The notebook was idle and everything is gone | The session expired. Run the cells again from the top. |

---

## C. Your laptop (before day 5)

Install four programs. After each one there is a check: do not skip it.
If a check fails, write down the message and ask a teacher on day 4. Do not arrive on day 5 without this part done.

### C1. Python 3.12

1. Download the installer of Python **3.12** from https://www.python.org/downloads/
2. Run it. **On Windows tick "Add python.exe to PATH"** on the first screen, then click Install Now.

Check: open a terminal (Windows: the program "PowerShell"; macOS: the program "Terminal") and type

```
python --version
```

On macOS type `python3 --version`. The answer must start with `Python 3.12`.

### C2. Visual Studio Code

1. Download it from https://code.visualstudio.com/ and install it.
2. Open it. In the left bar click **Extensions** (the icon with four squares).
3. Search `Python` and install the extension published by Microsoft.

Check: in the Extensions list, "Python" appears as installed.

### C3. Git

1. Windows: download it from https://git-scm.com/downloads and install it, leaving all the default options.
   macOS: type `git --version` in the Terminal; if Git is missing, macOS offers to install it. Accept.
2. Tell Git who you are. Type these two lines in the terminal, with your name and the e-mail of your GitHub account:

```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Check: `git --version` prints a version number.

### C4. Docker Desktop (optional)

Docker is a large download and does not work on every laptop. We show it in class in any case.

1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/ and install it.
2. Start it and wait until it says that the engine is running.

Check: `docker run hello-world` prints "Hello from Docker!".

### C5. Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Click **Continue with GitHub** and authorise it.

Check: you see the page "My apps" (empty).

### C6. What to bring on day 5

- Your laptop, with parts C1, C2 and C3 done and checked.
- The file `lung_segmenter.pth` that you downloaded from Kaggle at the end of STEP 11 of the project.
