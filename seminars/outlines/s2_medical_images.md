# Deck s2_medical_images — "2 - Medical images and their files"
Day 1, second seminar, just before notebooks 01 (medical images) and 02 (MONAI transforms). 40 minutes.
Footer title: "Medical images". Thread: where the picture comes from -> what the numbers mean -> a scan is
a volume with a real size and a direction -> how it is stored (DICOM, NIfTI) -> what a network needs.
Every slide prepares a section of notebook 01 or 02; the notes say which one.

Real values to use (from the course files):
- demo scan: 512 x 512 pixels, 121 slices, pixel spacing 0.76 mm, slice thickness 2.5 mm
- DICOM header of one slice: Modality CT; Rows 512; Columns 512; PixelSpacing [0.76, 0.76];
  SliceThickness 2.5; ImagePositionPatient [-198.1, -195.0, -230.2]; RescaleSlope 1; RescaleIntercept -1024
- HU = stored value x RescaleSlope + RescaleIntercept
- Hounsfield units: air -1000, lung about -850 to -700, fat about -100, water 0, soft tissue about 20 to 80, bone from about 300 up
- our windows: lung window shows -1000..400; the network gets -1000..400 rescaled to 0..1 and resized to 256 x 256
- label values in our files: 0 background, 1 right lung, 2 left lung, 3 trachea
- dataset: 888 patients, one slice each

Slides (18):
1. cover (cover) — "Medical images and their files". lead: "What is inside a CT scan, and how to give it to a network." Picture right: view_coronal.
2. modalities (cards.four) — eyebrow "Where images come from", title "Different machines, different physics". Four cards, `.tag` = what it measures, h3 = name, `.txt` one line: X-ray ("X-ray absorption", "One projection, 2D"), CT ("X-ray absorption", "Many projections, a 3D volume"), MRI ("Magnetism of tissue", "3D, great for soft tissue, no radiation"), PET ("Radioactive tracer", "Shows function, not shape"). Under the cards: ct_soft (cap "CT, our modality") and two `.todo` frames with caps "Teacher: add an X-ray" and "Teacher: add a brain MRI".
3. ct-how (cols center) — title "How a CT scanner sees". Left: view_axial 480x480. Right .flow vertical is not available, so three stacked cards (small padding): "An X-ray tube turns around the patient", "Dense tissue stops more X-rays", "A computer rebuilds one slice at a time". Notes: no physics beyond this.
4. matrix (cols center) — title "A slice is a matrix of numbers". Left ct_lung 440x440, arrow, right pixel_values 440x440 with cap "The numbers at the border of a lung". lead "Not colours: measurements." (Notebook 01, section 2)
5. hu (plain) — title "The numbers have a unit". Picture hu_scale (fit) 1540x352 wide. lead: "Hounsfield units: air is -1000, water is 0. The same in every scanner." (01, section 3)
6. windowing (img-row of three) — title "One slice, three ways to look at it". ct_lung, ct_soft, ct_bone 500x500 each with caps "Lung window", "Soft tissue window", "Bone window". lead: "A screen has 256 greys, the scan has thousands of values: you choose which range to show." (01, section 4)
7. mask (img-row three) — title "A mask is also an image". ct_lung, ct_mask, ct_labels 500x500, caps "The slice", "Mask: 1 inside the lungs, 0 outside", "Labels: 1 right lung, 2 left lung, 3 trachea". (01, section 5)
8. divider — eyebrow "Part 2", h2 "A scan is a volume".
9. stack (cols center) — title "Many slices, one volume". Left slice_stack (fit, 640x480). Right .big "121" + big-label "slices in our demo scan, each 512 x 512 pixels". (01, section 9)
10. views (img-row three) — title "Three ways to cut it". view_axial 440x440 cap "Axial", view_coronal cap "Coronal", view_sagittal cap "Sagittal" (keep each file's proportions, height 440). 
11. spacing (cols) — title "Pixels have a size in millimetres". Left fig spacing_wrong (fit, 700x165) cap "Ignoring the spacing"; right fig spacing_right (fit, 568x440) cap "With the spacing: 0.76 mm in the slice, 2.5 mm between slices". lead: "Areas and volumes need the spacing. You will measure a lung area." (01, section 6, exercise 2)
12. orientation (cols center) — title "Images have a direction". orient_wrong and ct_lung side by side 440x440, caps "Same file, shown without care", "Shown the way radiologists expect". lead: "Left and right matter. The file says how the axes are oriented; your code must respect it." Notes: in our notebooks `plt.imshow(image.T)`.
13. divider — eyebrow "Part 3", h2 "How it is stored".
14. dicom (cols, light) — title "DICOM: what the scanner writes". Left three short lines: "One file per slice", "A header with tags, then the pixels", "The format of every hospital". Right `.code` block with the real header: Modality, Rows, Columns, PixelSpacing, SliceThickness, ImagePositionPatient, RescaleIntercept (one per line, `Tag: value`). (01, section 7)
15. dicom-work (flow three steps, light) — title "Reading a DICOM series". Steps: "Read all the files of the series" -> "Sort the slices by position" (d: "file names are not a safe order") -> "Convert to Hounsfield units" (d: "value x slope + intercept"). (01, section 7 and exercise 4)
16. nifti (table, light) — title "NIfTI: what researchers use". Table DICOM vs NIfTI, rows: Files ("one per slice" / "one per scan"), Header ("hundreds of tags, patient data" / "size, spacing, orientation"), Used ("hospitals, scanners" / "research, datasets, MONAI"), In this course ("one demo series" / "everything else"). lead "Converting to NIfTI also removes the patient's name." (01, section 8)
17. prepare (flow four steps with pictures) — title "What the network needs". Row of four figs: ct_raw 380x380 cap "As stored", prep_scaled cap "Intensities from 0 to 1", prep_resized cap "256 x 256 pixels", augment_2 cap "Small random changes for training". lead: "Each step is one MONAI transform." (notebook 02) If two images must overlap (two scans of the same patient) one more step is needed, registration: say it only in the notes.
18. now (accent) — eyebrow "Now, in the notebook", title "Open notebook 01". Two cards on the accent background are hard to read, so use a plain list of three lines in 42px (.lead): "01: open a slice, change the window, measure a lung, read DICOM and NIfTI", "02: build the pipeline that prepares images for the network", "Read each instruction, complete the cell, compare with \"You should see\"".
