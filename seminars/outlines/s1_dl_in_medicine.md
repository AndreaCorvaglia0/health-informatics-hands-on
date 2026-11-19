# Deck s1_dl_in_medicine — "1 - Deep learning in medicine"
Day 1, right after the course presentation. 30-35 minutes. Footer title: "Deep learning in medicine".
Purpose: an overview that makes them want to start. Not a history lesson, not a list of papers.
Thread: one picture, many possible questions -> the kinds of answer a network can give -> where this is
already used -> why medicine is harder than cats and dogs -> so this is how a project is built (our week).

Facts you may use (source goes in the notes):
- Gulshan et al., JAMA 2016: a network grades diabetic retinopathy on retinal photographs.
- Esteva et al., Nature 2017: skin lesion classification trained on about 130 000 clinical images.
- Ardila et al., Nature Medicine 2019: lung cancer screening on low-dose chest CT.
- McKinney et al., Nature 2020: breast cancer screening on mammograms.
- Ronneberger et al., 2015: the U-Net, the model of this course.
- Wasserthal et al., Radiology: AI 2023: TotalSegmentator segments more than 100 structures in a CT scan.
- The FDA keeps a public list of AI-enabled medical devices; most of them are in radiology. The slide says
  "1,000+"; secondary sources report about 1,600 in 2026: check the current number on fda.gov before the lecture.
- Our own numbers: 888 patients; 5 reference lung masks in the dataset are wrong; the lung model reaches
  Dice 0.978 on patients never seen.

Slides (15):
1. cover (cover) — "Deep learning in medicine". lead: "What networks do today with clinical data, and why it is harder than it looks." Picture right: ct_lung.
2. question (plain) — eyebrow "One image", title "What would you ask about this image?" Big picture ct_lung (520x520) left; right column, four short questions appearing one by one (.rise): "Is something wrong?", "Where is it?", "How big is it?", "What will happen to the patient?". Notes: each question is a different task; the talk follows them.
3. idea (accent statement) — "A network learns the answer from examples: many images, each with the right answer." lead: "Nobody writes the rules. This is the difference with classic software."
4. data-kinds (cards, 3x2, light) — title "Medicine produces many kinds of data". Cards: Images (CT, MRI, X-ray, pathology slides), Signals (ECG, EEG), Text (reports, notes), Tables (lab values, clinical records), Sequences (genomics), Video (endoscopy, ultrasound). One line each. Notes: this week images; the extra module mixes images and tables.
5. divider — eyebrow "Part 1", h2 "Four kinds of answer".
6. classify (cols) — title "Classify: one answer per image". Left picture ct_lung 460x460 with cap "Healthy or not?"; right two cards: "Retina photographs: grading diabetic retinopathy" and "Skin photographs: benign or malignant lesion". Notes with sources (Gulshan 2016, Esteva 2017).
7. detect (cols) — title "Detect: find where it is". Left task_detect 460x460; right: nodule_1, nodule_3, nodule_5, nodule_2 in a row of four 180x180 with cap "Lung nodules: a few millimetres in a whole scan" and two short lines: screening for lung cancer on CT, screening mammograms. Notes: Ardila 2019, McKinney 2020. Nodules come back on day 4.
8. segment (cols) — title "Segment: one answer per pixel". Left ct_overlay 460x460; right three lines as cards.two or stack: "Measure: volumes, areas", "Plan: radiotherapy, surgery", "Follow: is the lesion growing?". lead: "This is what you build this week."
9. predict (plain) — title "Predict: what happens next". A .flow of three steps: "Image and clinical data" -> "Model" -> "Risk, survival, response to therapy". lead: "Here images are usually combined with other data." Notes: link to the optional module on multimodal models.
10. more (cards three, light) — title "And also". "Reconstruct: sharper images from faster or lower-dose scans", "Write: draft reports from images and notes", "Search: find similar past cases". Keep it one line each. Notes: say these exist, do not explain.
11. divider — eyebrow "Part 2", h2 "From papers to hospitals".
12. clinic (cols center) — left .big "1,000+" and .big-label "AI-enabled medical devices on the FDA list. Most are in radiology."; right three short lines: "They help, they do not decide", "Each does one narrow task", "A person signs the report". Notes: source FDA list.
13. hard (cards.two x2 = four cards) — title "Why medicine is harder". Cards: "Few labelled images" (txt: a label costs the time of an expert), "Labels are opinions" (txt: 5 of our 888 reference masks are wrong), "Every hospital is different" (txt: scanners, protocols, patients), "Mistakes cost" (txt: so models are tested on patients never seen). Notes: we will meet each of these during the week.
14. recipe (flow, 5 steps) — title "How a project is built". Steps: Data -> Preparation -> Model -> Evaluation -> Product, each with a 3-5 word description and the day ("days 1", "day 1-2", "day 2-3", "day 4", "day 5") in `.n`. lead: "The model is one step out of five."
15. next (accent) — eyebrow "Next", title "First, the images". lead: "Before training anything you must know what is inside a CT file. That is the next seminar." No "Now, in the notebook" here because another seminar follows.
