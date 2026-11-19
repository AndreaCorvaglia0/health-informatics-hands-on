# Deck s3_segmentation_models — "3 - From tasks to the U-Net"
Day 2 seminar, before notebook 03 STEP 1-7 (files, split, transforms, DataLoader, model, loss, metric). 45 minutes.
Footer title: "Segmentation and the U-Net". Thread: three tasks on the same image -> segmentation = a label
per pixel -> why simple rules are not enough -> convolutions -> encoder -> the problem of lost position ->
decoder and skip connections = U-Net -> its output -> how we score it (Dice) and how we split the data ->
what is beyond the U-Net today. No formulas except Dice.

Facts and numbers:
- our model: MONAI UNet, 2D, 1 input channel, 2 output classes, channels (16, 32, 64, 128, 256), about 1.6 million parameters (say "under two million" on the slide)
- input 256 x 256; the encoder halves the size four times: 256 -> 128 -> 64 -> 32 -> 16
- split: 888 patients -> 621 training, 133 validation, 134 test
- Dice = 2 x overlap / (size of prediction + size of reference); 1 is perfect, 0 is no overlap
- lung area measured by the app on a sample slice: 361 cm2
- U-Net: Ronneberger, Fischer, Brox, MICCAI 2015. V-Net and the Dice loss: Milletari et al., 2016.
- nnU-Net: Isensee et al., Nature Methods 2021: a U-Net that configures itself from the dataset.
- TotalSegmentator: Wasserthal et al., Radiology: AI 2023: more than 100 anatomical structures in CT, built on nnU-Net.
- Transformers for segmentation: UNETR and Swin UNETR (Hatamizadeh et al., 2022).
- Segment Anything (Kirillov et al., 2023): one model, you point or draw a box, it segments. MedSAM (Ma et al., Nature Communications 2024): the same idea trained on medical images.
- "nnU-Net Revisited" (Isensee et al., MICCAI 2024): well-configured U-Nets still match or beat newer architectures on 3D medical benchmarks.

Slides (20):
1. cover (cover) — "From tasks to the U-Net". lead "What segmentation is, and the network that does it." Picture right: model_overlay.
2. tasks (img-row three) — eyebrow "One image, three tasks", title "What do you want to know?" ct_lung cap "Classification: one label for the image", task_detect cap "Detection: a box around each object", ct_overlay cap "Segmentation: a label for every pixel". 500x500 each.
3. segmentation (cols center) — title "Segmentation gives you shape and size". Left ct_labels 480x480. Right .big "361 cm2" big-label "lung area in one slice, measured from the mask" and one line "With a mask you can measure, plan and compare over time."
4. threshold (cols) — title "Why not a simple rule?" Left hu_histogram (fit, 800x360). Right: "Lungs are dark: keep the pixels below a threshold." then a fragment: "It also keeps the air outside the body, the trachea, the bowel. And it fails on diseased lungs." Notes: this is the extra exercise of notebook 01.
5. divider — eyebrow "Part 1", h2 "How a network looks at an image".
6. convolution (img-row three) — title "A filter slides over the image". prep_resized cap "The slice", conv_edges cap "A filter that reacts to vertical borders", conv_edges_h cap "A filter that reacts to horizontal borders". 500x500. lead "A filter is a small grid of numbers, 3 x 3."
7. learned (img-row four) — title "The network learns its own filters". model_input, feature_1, feature_2, feature_3 at 380x380, caps "Input", then "Three of the 16 maps of the first layer of our trained model". lead "Nobody chose these filters. Training did."
8. encoder (img-row five) — title "Going down: smaller image, wider view". pyramid_256, pyramid_128, pyramid_64, pyramid_32, pyramid_16 each 300x300 with caps "256", "128", "64", "32", "16". lead "Each step halves the size. The network sees more context and less detail."
9. problem (accent statement) — "At 16 x 16 the network knows there are lungs. It no longer knows where their border is."
10. unet (plain) — title "The U-Net: go down, come back up". unet_diagram (fit) 1160x560 centred. Notes: encoder, bottleneck, decoder; numbers are channels.
11. skip (cols center) — title "Skip connections bring the detail back". Left pyramid_16 280x280 cap "What: from the bottom" plus pyramid_256 280x280 cap "Where: from the top" side by side; right one sentence: "The decoder gets both: the coarse map from below and the sharp map from the same level of the encoder." 
12. output (img-row four) — title "What comes out". model_input cap "Input: 1 channel", model_prob cap "Probability of lung for each pixel", model_mask cap "Mask: the class with the highest score", model_overlay cap "On the image". 380x380. lead "Two scores per pixel: background and lung."
13. code (cols, light) — title "The U-Net in MONAI". `.code` block:
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
   Right three short lines mapping: "in_channels: one grey image", "out_channels: background, lung", "channels: the five levels of the U". (Notebook 03, STEP 6)
14. divider — eyebrow "Part 2", h2 "Is the mask right?"
15. dice (cols center) — title "Dice score: how much two masks overlap". Left good_compare 480x480 cap "Reference border and model border". Right the formula as large text (h3 size, two lines): "Dice = 2 x overlap / (prediction + reference)" and "1 = identical, 0 = no overlap". Notes: the Dice loss is 1 - Dice computed on probabilities; handbook section 5.
16. split (flow three, light) — title "Three sets of patients". Steps with `.n` = number: "621" t "Training" d "The network learns from these"; "133" t "Validation" d "You choose settings and the best epoch"; "134" t "Test" d "Used once, at the end". lead "A patient is in one set only." (STEP 3)
17. divider — eyebrow "Part 3", h2 "Beyond our U-Net".
18. frontier (cards.two, two rows = four cards) — title "What is used today". Cards: tag "2021" h3 "nnU-Net" txt "A U-Net that configures itself from the dataset"; tag "2023" h3 "TotalSegmentator" txt "More than 100 structures in a CT, ready to use"; tag "2022" h3 "Transformers" txt "UNETR, Swin UNETR: attention instead of some convolutions"; tag "2023-24" h3 "Promptable models" txt "Segment Anything, MedSAM: click a point, get a mask". Notes with the sources.
19. still (accent statement) — "Ten years later, a well-trained U-Net is still the baseline to beat." lead "This is why you learn it first." Notes: nnU-Net Revisited, MICCAI 2024.
20. now (plain) — eyebrow "Now, in the notebook", title "Notebook 03, STEP 1 to 7". A .flow of four steps: "Files and split" -> "Transforms and DataLoader" -> "The model" -> "Loss, optimizer, metric". lead "At the end of today your model is ready to train. Training is tomorrow."
