# samples/

Small example files shown in the "Bundled sample" list of the app.

* CT slices: `name.nii.gz` (2D NIfTI, Hounsfield units) or `name.dcm` (one DICOM slice);
* optional ground truth next to a NIfTI slice: `name_label.nii.gz` (1 = right lung, 2 = left lung): the app uses it
  automatically to show the Dice and the error map; files ending in `_label` are not listed as samples.

`app_starter_kit.zip` (notebook 06) puts 3-5 test-split slices here. Use **only public or synthetic data**: never put
real patient data in a repository. DICOM headers can contain the patient name, ID and birth date.
