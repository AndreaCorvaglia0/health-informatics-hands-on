# Seminars

One slide deck for each seminar. The decks are a proposal: open them, change the words, add your pictures.
The outline of each deck (thread, slide by slide, pictures, sources of the facts) is in `outlines/`.

| When | Deck | Slides | Time | Link |
|---|---|---|---|---|
| Day 1, opening | 0 - Course presentation | 11 | 15 min | https://claude.ai/artifact/1VLpvUBfAnVY39n3oENDce |
| Day 1 | 1 - Deep learning in medicine | 15 | 30 min | https://claude.ai/artifact/VVQMDhRbJZgzDhpwwbUJxs |
| Day 1 | 2 - Medical images and their files | 18 | 40 min | https://claude.ai/artifact/7WThw3RsDTrcChtGTGTdYA |
| Day 2 | 3 - From tasks to the U-Net | 20 | 45 min | https://claude.ai/artifact/H8ao2pqUGVG1jKoZ3ao9RG |
| Day 3 | 4 - Training a model and judging it | 20 | 45 min | https://claude.ai/artifact/42nQEMxCnKKPDFt1XfBLdg |
| Day 5 | 5 - From notebook to web app | 17 | 25 min | https://claude.ai/artifact/MAAJAamEVuQhhnZZgBUJVx |
| Optional | Extra - Images and clinical data together | 18 | 20 min | https://claude.ai/artifact/MPqQXuseyAdoeW5UP2tdZ2 |

Day 4 has no seminar: deck 4 also covers the test set and the nodules.

## How the decks are made

- Each deck opens with a question or a picture, and closes with "Now, in the notebook": what the students do
  next with those ideas. The eyebrow of many slides names the notebook section or STEP it prepares.
- Every slide has speaker notes: what to say and why the next slide follows.
- The pictures come from our dataset and from the reference run of the solutions (CT slices, masks, curves,
  worst cases, the web app). The numbers on the slides are the ones of the solutions.
- Facts about the outside world have their source in the notes of the slide.

## To do before the course

- Deck 1, slide "Classify": two empty frames for a retina photograph and a skin lesion photograph.
- Deck 2, slide "Different machines, different physics": two empty frames for an X-ray and a brain MRI.
- Deck 1, slide "Already in the clinic": the slide says "1,000+" devices on the FDA list. Check the current
  number on fda.gov (secondary sources report about 1,600 in 2026).
- The numbers of decks 3, 4 and 5 come from a CPU run of the solutions on the full dataset. If your run on
  Kaggle gives different values, change them (test Dice 0.978, worst case 0.17, nodule table).
