# Seminars

One slide deck for each seminar. The decks are a proposal: tell us what to change.
The first plan of each deck (thread, slide by slide, pictures, sources of the facts) is in `outlines/`; where a
deck and its outline differ, the deck is the current version.

The slides are in `seminars/slides/` of this repository, one PowerPoint (`.pptx`) and one PDF for each deck.
Open the PowerPoint to present: the speaker notes are in the notes of each slide.

| When | Deck | Slides | Time | File in `seminars/slides/` |
|---|---|---|---|---|
| Day 1, opening | 0 - Course presentation | 11 | 15 min | `0_course_presentation` |
| Day 1 | 1 - Deep learning in medicine | 15 | 30 min | `1_deep_learning_in_medicine` |
| Day 1 | 2 - Medical images and their files | 18 | 40 min | `2_medical_images_and_files` |
| Day 2 | 3 - From tasks to the U-Net | 21 | 45 min | `3_from_tasks_to_the_unet` |
| Day 3 | 4 - Training a model and judging it | 20 | 45 min | `4_training_and_judging_a_model` |
| Day 5 | 5 - From notebook to web app | 17 | 25 min | `5_from_notebook_to_web_app` |
| Optional | Extra - Images and clinical data together | 18 | 20 min | `extra_images_and_clinical_data` |

Each slide of the PowerPoint is one picture of the slide, so the text cannot be edited in PowerPoint. To change a
deck, ask Andrea: the original decks are in his Claude account, and after a change he exports the files again.
If you need a slide you can edit, add a new slide in your copy of the PowerPoint.

Day 4 has no seminar: the last part of deck 4 is a preview of day 4 (test set, failures, nodules). It shows the
worst cases of our run as a question and leaves the answer to the students (the answer is in the speaker notes).

## How the decks are made

- Each deck opens with a question or a picture, and most close with "Now, in the notebook": what the students do
  next with those ideas. The eyebrow of many slides names the notebook section or STEP it prepares.
- Every slide has speaker notes: what to say and why the next slide follows.
- The pictures come from our dataset and from the reference run of the solutions (CT slices, masks, curves,
  worst cases, the web app). The numbers on the slides are the ones of the solutions.
- Facts about the outside world have their source in the notes of the slide.
- Pictures that are not ours (retina, skin lesion, mammogram, chest X-ray, brain MRI, PET, CT scanner) come from Wikimedia
  Commons; author and licence are in the notes of their slide.

## To do before the course

- If you have them, add your own screenshots: a notebook on Kaggle (deck 0), the app online and the GitHub
  repository (deck 5).
- Deck 1, slide "Already in the clinic": the slide says "1,000+" devices on the FDA list. Check the current
  number on fda.gov (secondary sources report about 1,600 in 2026).
- The numbers of decks 3, 4 and 5 come from a CPU run of the solutions on the full dataset. If your run on
  Kaggle gives different values, change them (test Dice 0.978, worst case 0.17, nodule table).

## Animations

`animations/` has two GIFs that explain `strides` and `num_res_units` with a real nodule patch. Use them in seminar 3 and on day 4. See `animations/README.md`.
