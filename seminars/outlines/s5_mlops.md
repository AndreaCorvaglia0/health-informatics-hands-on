# Deck s5_mlops — "5 - From notebook to web app"
Day 5, short seminar (25-30 minutes) before the final part: run the app on the laptop, complete it, GitHub,
publish on Streamlit Community Cloud, Docker. Footer title: "From notebook to web app".
Thread: your model is a file trapped in a notebook -> what must leave the notebook with it (weights,
preparation, versions) -> keeping track of what you did -> an interface for people -> the same thing on
another computer: environments, then Docker with one metaphor -> a server: what changes -> today's plan.
Keep MLOps to what they will touch today. Name bigger tools in the notes only.

Facts:
- weights file: lung_segmenter.pth, saved with torch.save(model.state_dict(), ...). It holds the numbers only:
  to load it you must build the same UNet first, then model.load_state_dict(torch.load(path, map_location="cpu")), then model.eval()
- the app must prepare the image exactly as in training: Hounsfield window -1000..400 to 0..1, resize to 256 x 256
- requirements.txt pins versions (torch==2.5.1, monai==1.6.1, streamlit, nibabel); CPU build of PyTorch
- the app: Streamlit, one Python file app.py; upload a NIfTI slice, see slice, mask, overlay, lung area in cm2
  (361 cm2 on the sample), optional Dice if a reference mask is uploaded
- experiment table of notebook 04 (validation Dice): baseline 0.843; another seed 0.835; Dice + CE loss 0.828; augmentation 0.858
- our Dockerfile (comments removed):
    FROM python:3.12-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    EXPOSE 8501
    CMD ["streamlit", "run", "app.py"]
- commands: docker build -t lung-app .   then   docker run -p 8501:8501 lung-app
- deploy: Streamlit Community Cloud reads a public GitHub repository and runs app.py; free, CPU only, limited memory, the app sleeps when unused
- Docker metaphor to use everywhere, the kitchen: Dockerfile = the recipe (a text file); image = the meal cooked and
  frozen (built once, identical everywhere, can be copied); container = one portion heated and served (a running
  copy; you can serve many, and throwing one away does not touch the frozen meal)
- tools to name only in the notes: MLflow, Weights and Biases (experiment tracking); DVC (data versions)

Slides (17):
1. cover (cover) — "From notebook to web app". lead "Your model works. Now somebody else has to use it." Picture right: app_result (class fit, on a light box: style background #F3EFE6; keep proportions, do not crop).
2. trapped (accent statement) — "Today your model lives in a notebook, on Kaggle, and only you can run it."
3. leaves (cards three) — eyebrow "What must leave the notebook", title "A model is three things". "The weights" tag "lung_segmenter.pth" txt "the numbers learned in training"; "The preparation" tag "transforms" txt "the same steps applied to every new image"; "The environment" tag "requirements.txt" txt "the same versions of the libraries".
4. weights (cols, light) — title "Save and load the weights". `.code` (max 8 lines):
    # save (in the notebook)
    torch.save(model.state_dict(), "lung_segmenter.pth")

    # load (in the app)
    model = UNet(...)   # the same architecture
    model.load_state_dict(torch.load("lung_segmenter.pth"))
    model.eval()
   Right: "The file holds numbers, not the network. You rebuild the same U-Net, then fill it."
5. skew (img-row or cols) — title "Same model, wrong preparation". Left figs skew_input and skew_mask 440x440 caps "Image prepared in a different way", "What the model answers". Right or lead: "Nothing crashes. The answer is just wrong. The app must repeat the training steps exactly."
6. track (table, light) — title "Write down what you did". Reuse the experiment table: Setting | Validation Dice (4 rows). lead "For each run: the code, the data, the settings, the seed, the result. In a month you will not remember." Notes: tools that do this for you: MLflow, Weights and Biases.
7. divider — eyebrow "Part 2", h2 "An interface for people".
8. app (cols center) — title "A web app around the model". Left app_empty (fit, light box, about 640x512 keep proportions) cap "Before"; right app_result same size cap "After uploading a slice". 
9. streamlit (cols, light) — title "Streamlit: an app is a Python file". `.code`:
    import streamlit as st

    st.title("Lung segmentation")
    file = st.file_uploader("Upload a CT slice")
    if file is not None:
        mask = predict(file)
        st.image(mask)
   Right: "No HTML, no JavaScript. Every time the user does something, the file runs again from the top."
10. divider — eyebrow "Part 3", h2 "The same app on another computer".
11. works (cols center) — title "It works on my machine". Two cards: "Your laptop" txt "Python 3.12, your packages, your files" and "Another computer" txt "another Python, other versions, no model file". lead "First fix: a virtual environment and a requirements.txt. You do this today."
12. kitchen (flow three) — title "Docker, in a kitchen". Steps: n "Dockerfile" t "The recipe" d "A text file: start from Python, install the packages, copy the app" -> n "Image" t "The frozen meal" d "Cooked once, the same everywhere, easy to copy" -> n "Container" t "A portion, served" d "A running copy. Serve as many as you want".
13. dockerfile (cols, light) — title "Our recipe". Left `.code` with the 7 lines of the Dockerfile. Right two lines as `.code` too or text: "docker build -t lung-app ." cap "cook and freeze" and "docker run -p 8501:8501 lung-app" cap "serve". 
14. server (cols) — title "On a server: what changes". Left a .flow of three small steps: "Your laptop" -> "GitHub" -> "Streamlit Cloud". Right table or four short lines: "No GPU: predictions on CPU", "Little memory: a small model helps", "A public address: anyone can open it", "Never upload real patient data".
15. after (cards three) — title "After the deploy, the work continues". "Watch it" txt "does it still answer, and how fast"; "New data" txt "another hospital, another scanner: check again"; "New versions" txt "retrain, test, replace, keep the old one". Notes: this is what MLOps means in a company; a model used on patients is also a medical device and follows regulation.
16. today (flow, 5 steps max: merge) — eyebrow "Now", title "Today, step by step". "Laptop: download model, environment" -> "Run and complete the app" -> "GitHub" -> "Publish online" -> "Docker". lead "The guide is webapp/README.md. Follow it part by part."
17. end (accent statement) — "At the end of today, send the link of your app to somebody who has never seen a notebook."
