# Deck x1_multimodal_fusion — "Extra - Images and clinical data together"
Optional seminar (20-25 minutes) for the students who finished everything, before the notebook
`extra_multimodal/multimodal_fusion.ipynb`. Footer title: "Multimodal fusion (extra)".
Thread: a doctor never decides from the image alone -> so can a model use both? -> first problem: an image is
not a row of a table: two ways to turn it into numbers -> three places where the two kinds of data can meet
(early, late, intermediate) -> what we measured -> what is true outside this toy example -> the notebook.
Keep the theory light: no formulas, one picture or diagram per idea.

REAL NUMBERS: read the results of the solution notebook first and use ITS numbers (AUC of each model, fraction alive,
the example nodule, the first rows of the clinical table). Never use numbers from memory. Read also the notebook
`extra_multimodal/solutions/multimodal_fusion.ipynb` (markdown cells) to
use the same words and names as the notebook.

Pictures : fusion_patches, fusion_measure, fusion_pca, fusion_results (made from
the notebook), plus nodule_1..8, measure_nodule, ct_overlay, model_input.

Outside facts you may use (source in the notes):
- Vocabulary early / joint (intermediate) / late fusion: Huang et al., npj Digital Medicine 2020. Other authors
  use the words a little differently.
- Real example: lung cancer risk from screening CT plus clinical data (Radiology: Artificial Intelligence 2021,
  doi 10.1148/ryai.2021210032): AUC 0.88 with both, 0.86 image only, 0.69 clinical only, on about 23 500 patients.
- A review of 34 studies fusing health records and images (Mohsen et al., Scientific Reports 2022): early fusion
  was the most used (22 of 34), late fusion the least.
- MONAI DenseNet121 pretrained on ImageNet photographs gives 1024 numbers per image.
- Networks pretrained on medical images exist: RadImageNet (1.35 million CT, MRI and ultrasound images).
- PyRadiomics is the usual library for radiomics; in the notebook we compute five measurements by hand instead.
- "Alive at one year: yes/no" ignores patients lost before one year; real studies use survival analysis.

Slides (17):
1. cover (cover) — "Images and clinical data together". lead "An optional module on multimodal models." Picture right: fusion_patches is a wide strip, so use nodule_3 (or measure_nodule) as the cover picture.
2. doctor (cols center) — eyebrow "The starting point", title "Nobody decides from the image alone". Left measure_nodule 520px cap "A lung nodule". Right a small table (4 rows, 2 columns) with one invented patient taken from results.md: Age, Smoking (pack-years), Performance status, and a last row "Alive after one year?" with "?" in amber.
3. question (accent statement) — "Does a model that sees both predict better than a model that sees only one?"
4. modality (cards.four) — title "A modality is a kind of data". Image / Table / Text / Signal, one line each with a medical example. lead "Multimodal: a model that uses more than one."
5. data (cols, light) — title "The data of this module". Left: table with the first 4 rows of the clinical table from results.md (age, sex, pack_years, performance_status, alive_1y). Right two lines: "601 patients, one nodule each: real images." and, in amber (.hl2), "Clinical data and survival are invented. Nothing here has medical meaning."
6. divider — eyebrow "Part 1", h2 "An image as a row of numbers".
7. measure (cols center) — title "Way 1: measure it". Left fusion_measure (fit, keep proportions, about 600px high). Right three cards stacked or three lines: "Shape: area, diameter", "Intensity: mean and spread of the Hounsfield values", "Texture: how irregular it is (not in our notebook)". lead "This needs a mask. It is the idea of radiomics."
8. embed (flow three + picture) — title "Way 2: let a network describe it". Flow: "Nodule patch" -> "Network trained on a million photographs" -> "1024 numbers: the embedding". Under or beside it fusion_pca (fit) with cap "The embeddings of our nodules, squeezed to two numbers and coloured by diameter". If both do not fit, split into two slides (embed, embed-map).
9. divider — eyebrow "Part 2", h2 "Three places to join them".
10. early (flow) — title "Early fusion: join the inputs". Diagram with kit classes: two small steps stacked on the left ("Image numbers", "Clinical table") -> "One table" -> "One model" -> "Answer". lead "Simple. The model can use combinations of the two."
11. late (flow) — title "Late fusion: join the answers". Two rows: "Image numbers" -> "Model A" -> "Probability A"; "Clinical table" -> "Model B" -> "Probability B"; then both -> "Average". lead "Works even when one kind of data is missing for a patient."
12. intermediate (flow) — title "Intermediate fusion: join inside the network". "Image" -> "Image branch" and "Clinical table" -> "Table branch", both -> "Joined layers" -> "Answer"; a sentence: "Trained together: the image branch learns what is useful for this answer." lead "The most flexible. It needs the most data."
13. compare (table, light) — title "Which one?" Columns: (blank) | Early | Late | Intermediate. Rows: "What is joined" (inputs / answers / learned features); "Models" (one / one per modality / one network with branches); "Needs" (all data for every patient / can skip a missing modality / many patients); "In the notebook" (STEP 6 / STEP 7 / STEP 9). Check the step numbers against the notebook.
14. results (cols center) — title "What we measured". Left fusion_results (fit, at least 1000px wide). Right two short lines using the real numbers: the best single model, the best fusion. Notes: all the numbers from results.md.
15. careful (cards three) — title "Before you believe it". "Invented data" txt "the formula was built so that both kinds of data matter"; "Small test set" txt "about 180 patients: differences of 0.02 are noise"; "In a real study" txt with the real example: "lung screening CT plus clinical data: 0.88 against 0.86 with the image alone". Notes with the source.
16. beyond (cards three) — title "Where this goes". "Networks pretrained on medical images" txt "RadImageNet and others: better embeddings than everyday photographs"; "Images and text together" txt "models that read the scan and the report"; "Survival analysis" txt "when and not only whether: patients lost before one year still count".
17. now (plain) — eyebrow "Now, in the notebook", title "extra_multimodal/multimodal_fusion.ipynb". Flow of four: "Clinical model" -> "Image model" -> "Early and late fusion" -> "Embeddings, intermediate fusion". lead "CPU is enough. About two hours."
