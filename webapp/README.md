# Day 5 - The web app

Today the model leaves Kaggle. You build a small web page where anybody can upload a CT slice and see the
lungs found by **your** model, and you publish it on the internet.

| Part | What you do | Time |
|---|---|---|
| 1 | From Kaggle to your laptop: project folder and model | 15 min |
| 2 | Virtual environment and packages | 20 min |
| 3 | Run the app | 10 min |
| 4 | Complete the app (STEP 1 to 6 in `app.py`) | 60 min |
| 5 | Put the project on GitHub | 20 min |
| 6 | Publish the app | 20 min |
| 7 | Docker | 20 min |

Before you start: part C of `SETUP.md` must be done (Python, VS Code, Git).

Do the parts in order. Each one ends with a check.

---

## 1. From Kaggle to your laptop

1. Copy the folder `webapp` of the course material to your computer, for example on the Desktop.
   Rename it `lung-app`.
2. Copy your model `lung_segmenter.pth` into the folder `lung-app`.
   (You downloaded it from Kaggle at the end of STEP 11. If you do not have it, ask a teacher for theirs.)
3. Open VS Code. Choose **File -> Open Folder** and select `lung-app`.

Check: in the left panel of VS Code you see

```
lung-app
  samples/
  app.py
  Dockerfile
  lung_segmenter.pth
  README.md
  requirements.txt
```

## 2. Virtual environment and packages

A **virtual environment** is a private copy of Python for this project. The packages we install go there and
do not touch the rest of your computer. It is a folder called `.venv` inside the project.

1. In VS Code open the terminal: **Terminal -> New Terminal**. It opens at the bottom, already inside `lung-app`.
2. Create the environment:

   | Windows | macOS / Linux |
   |---|---|
   | `python -m venv .venv` | `python3 -m venv .venv` |

3. Activate it:

   | Windows | macOS / Linux |
   |---|---|
   | `.venv\Scripts\activate` | `source .venv/bin/activate` |

   Check: the line of the terminal now starts with `(.venv)`.

4. Install the packages listed in `requirements.txt`:

   ```
   pip install -r requirements.txt
   ```

   It takes a few minutes: PyTorch is about 200 MB.

Check: `pip list` shows `torch`, `monai` and `streamlit`.

Every time you open a new terminal you must activate the environment again (point 3).

## 3. Run the app

1. In the terminal, with `(.venv)` active:

   ```
   streamlit run app.py
   ```

2. The browser opens the address `http://localhost:8501`.

Check: you see an empty page. It is correct: the app does nothing yet.

Leave this terminal running. To stop the app, click in the terminal and press `Ctrl + C`.

**How Streamlit works.** An app is a normal Python file. Streamlit runs it from the first line to the last and
draws on the page what the `st.` commands ask for. Every time the user does something (uploads a file, moves a
slider), Streamlit runs **the whole file again** from the top.

The commands we use:

| Command | What it shows |
|---|---|
| `st.title("...")`, `st.write("...")`, `st.caption("...")` | text |
| `st.file_uploader("...")` | a button to upload a file |
| `st.image(array)` | an image |
| `st.columns(3)` | three columns side by side |
| `st.metric("name", "value")` | a number in large type |
| `st.info("...")`, `st.success("...")` | a coloured message |

## 4. Complete the app

Open `app.py` in VS Code. It contains six steps, each with numbered instructions in the comments.

For each step:

1. read the instructions;
2. write the code where it says `# your code here`;
3. save the file (`Ctrl + S`, on macOS `Cmd + S`);
4. look at the browser. Streamlit shows **Source file changed**: click **Always rerun**, once;
5. compare the page with "You should see" before you go to the next step.

If the page shows a red box, it is a Python error: read the last line of the message and the line number,
fix the file, save.

| Step | Result |
|---|---|
| 1 | The page has a title |
| 2 | Your model is loaded |
| 3 | The preprocessing of the notebook is in the app |
| 4 | A file can be uploaded and the model predicts |
| 5 | Image, mask and overlay are shown |
| 6 | The lung area is shown in cm² |

Check: upload `samples/case_0010.nii.gz`. You see three pictures and a lung area of about 360 cm².

## 5. Put the project on GitHub

GitHub keeps a copy of your project online. The service that publishes the app reads the files from there.

1. In the left bar of VS Code click **Source Control** (the icon with three dots joined by lines).
2. Click **Publish to GitHub**. The first time VS Code asks to sign in to GitHub: accept, the browser opens,
   authorise it.
3. Choose **Publish to GitHub public repository** and confirm the name `lung-app`.
4. VS Code shows the list of files that will be uploaded. Leave them all selected and confirm.

Check: go to `https://github.com/<your username>/lung-app`. You see `app.py`, `requirements.txt`,
`lung_segmenter.pth` and the folder `samples`. You do **not** see `.venv`: the file `.gitignore` tells Git to
leave it out, because it is large and everybody can rebuild it from `requirements.txt`.

When you change a file later: in Source Control write a short message in the box, click **Commit**, then
**Sync Changes**.

**Alternative, with the terminal.** Create an empty public repository called `lung-app` on github.com
(button **New**), then type:

```
git init
git add .
git commit -m "Lung segmentation app"
git branch -M main
git remote add origin https://github.com/<your username>/lung-app.git
git push -u origin main
```

## 6. Publish the app

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Fill the form:
   - Repository: `<your username>/lung-app`
   - Branch: `main`
   - Main file path: `app.py`
4. Open **Advanced settings** and choose Python **3.12**.
5. Click **Deploy**.

The first time it takes about five minutes: the service installs the packages of `requirements.txt` on its own
computer. You can follow what it does in the panel on the right.

Check: the app opens at an address that ends with `.streamlit.app`. Send the address to a classmate and ask
them to upload a file from `samples`.

**Never upload real patient data to a public app.** Use only the files in `samples`.

## 7. Docker

Your app works on your laptop because you installed the right Python and the right packages. On another
computer it may not. A **Docker image** is a package that contains the app together with everything it needs:
Linux, Python, the libraries, your model. A **container** is an image that is running. The same image runs in
the same way on any computer that has Docker. This is how software is installed in hospitals.

The recipe to build the image is the file `Dockerfile`. Open it: it has six instructions, each one explained
by a comment.

If Docker Desktop works on your laptop (part C4 of the setup):

1. Start Docker Desktop.
2. Stop the Streamlit app in the terminal (`Ctrl + C`).
3. Build the image. The final dot is part of the command:

   ```
   docker build -t lung-app .
   ```

4. Start a container from the image:

   ```
   docker run -p 8501:8501 lung-app
   ```

Check: open `http://localhost:8501` in the browser. It is the same app, but it is now running inside the
container, not in your `.venv`. Stop it with `Ctrl + C`.

If Docker does not work on your laptop, follow the teacher's demonstration.

---

## If you have finished

- Do the **EXTRA** step at the end of `app.py`: upload the true mask and show the Dice score.
- Add a slider for the transparency of the overlay (`st.slider`).
- Add a button to download the predicted mask (`st.download_button`).
- If you trained more than one model in the extra experiments, let the user choose which one to use
  (`st.selectbox`).

## If something goes wrong

| Problem | What to do |
|---|---|
| `python` is not recognised (Windows) | Python is not in the PATH: install it again and tick "Add python.exe to PATH". |
| `.venv\Scripts\activate` gives a red error about "execution policy" (Windows) | Type `Set-ExecutionPolicy -Scope Process RemoteSigned`, press Enter, then activate again. |
| `streamlit` is not recognised | The environment is not active: you do not see `(.venv)`. Activate it (part 2, point 3). |
| `FileNotFoundError: lung_segmenter.pth` | The model is not in the folder `lung-app`, or it has another name. |
| `Error(s) in loading state_dict` | The network in `app.py` is not the same as in the notebook: check `channels`, `strides` and the three numbers of STEP 2. |
| The prediction is empty or wrong | The preprocessing is not the same as in the training: compare STEP 3 with STEP 13 of the notebook. |
| The deploy fails while installing packages | Open the log, read the last lines, and call a teacher. |
| The published app says it is asleep | Normal after some hours without visitors: click the button to wake it up. |

## To go further

- Streamlit documentation: https://docs.streamlit.io/
- Python virtual environments: https://docs.python.org/3/library/venv.html
- GitHub, first steps: https://docs.github.com/en/get-started/start-your-journey/hello-world
- Git cheat sheet: https://education.github.com/git-cheat-sheet-education.pdf
- Docker, getting started: https://docs.docker.com/get-started/
- Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud
