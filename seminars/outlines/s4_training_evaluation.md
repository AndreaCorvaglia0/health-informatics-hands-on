# Deck s4_training_evaluation — "4 - Training a model and judging it"
Day 3 seminar. It covers day 3 (notebook 03 STEP 8-11: training step, loop, curves, predictions) and day 4,
which has no seminar (STEP 12-13: test set, new image; then notebook 04 on nodules for who is ahead). 45 minutes.
Footer title: "Training and evaluation". Thread: the model is built but knows nothing -> one training step
-> repeat: epochs -> read the curves -> pick the best epoch with validation -> look at predictions -> the
test set, once -> look at the failures, not only the mean -> a harder target and how to experiment properly.

Numbers of our reference run (full dataset, 20 epochs, Adam, learning rate 0.001, batch 16):
- loss before training about 0.5-0.6; validation Dice of the best epoch 0.989
- test set: 134 patients, mean Dice 0.978, minimum 0.17. The three worst are 0.17, 0.68, 0.72.
- the three worst cases are errors in the reference masks, not errors of the model
- 5 of 888 reference masks are wrong (they are automatic masks, never checked by hand)
- nodules (notebook 04): baseline validation Dice 0.843; same settings with another seed 0.835 (so 0.008 is noise);
  Dice + cross-entropy loss 0.828; with augmentation 0.858; final test 0.853
- the five lines of a training step, exactly as in the notebook:
    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

Slides (20):
1. cover (cover) — "Training a model and judging it". lead "How a network learns, and how to know if you can trust it." Picture right: model_prob.
2. start (cols center) — eyebrow "Where we are", title "Your model exists. It knows nothing." Left model_input cap "Input", ct_mask cap "What we want", and a sentence "Its weights are random numbers. Training changes them, a little at a time."
3. step (cols, light) — title "One training step, five lines". Left `.code` with the five lines. Right five very short lines numbered 1-5: "forget the old gradients", "predict", "measure the error", "find how each weight should change", "change the weights a little". (STEP 8)
4. loop (flow four) — title "Then repeat". Steps: "Batch" d "16 images at a time" -> "Step" d "the five lines" -> "Epoch" d "all the training images once" -> "20 epochs" d "our training". lead "After every epoch you measure the Dice on the validation set." (STEP 9)
5. lr (cols center) — title "How big is each change?" Left learning_rate (fit, keep proportions, about 760 wide). Right: "The learning rate decides the size of the step." "Too small: slow. Too large: the loss jumps and never settles." "We use 0.001."
6. curves (cols) — title "The two curves to watch". Left curve_loss, right curve_dice (fit, each about 780 wide, keep proportions) caps "Training loss goes down" and "Validation Dice goes up, then flattens". (STEP 10)
7. overfit (cols center) — title "When to stop". Left overfitting (fit, about 800 wide). Right: "Training keeps improving, validation stops or gets worse: the network is memorising." then "So you keep the weights of the best validation epoch." 
8. save (cols, light) — title "Keep the best model". `.code`:
    if dice > best_dice:
        best_dice = dice
        torch.save(model.state_dict(), "lung_segmenter.pth")
   Right: "This file is your result. You will download it and use it in the web app on day 5."
9. look (img-row three) — title "Then look at what it does". good_image, good_truth, good_pred 500x500, caps "Slice", "Reference", "Prediction". lead "A number is not enough. Always look." (STEP 11)
10. divider — eyebrow "Day 4", h2 "Can you trust it?"
11. test (accent statement) — "The test set is opened once, at the end. If you use it to make choices, its number means nothing."
12. histogram (cols center) — title "134 patients never seen". Left test_histogram (fit, about 800 wide). Right .big "0.978" big-label "mean Dice on the test set". (STEP 12)
13. minimum (cols center) — title "The mean hides something". Left .big (class hl2 colour: use `class="big hl2"`) "0.17" big-label "the worst patient". Right: "One patient in 134. In a hospital that is a person." 
14. worst (picture grid) — title "The three worst patients". Grid 3 columns x 3 rows: slice, reference, prediction of worst1, worst2, worst3; Dice values (0.17, 0.68, 0.72) as `.pill` above the columns; at the right "Look at the reference masks."
15. lesson (accent statement) — "In all three, the model is right and the reference mask is wrong." lead "5 of our 888 reference masks have errors. Looking at failures tells you about your data."
16. divider — eyebrow "If you are ahead", h2 "A harder target: nodules".
17. nodules (cols) — title "A few millimetres wide". Left: grid 4x2 of nodule_1..8 at 190x190. Right dice_small_objects (fit, about 700 wide, keep proportions) cap "On small objects one pixel changes the Dice a lot". (Notebook 04)
18. experiments (table, light) — title "Experiments on nodules: our numbers". Table: Setting | Validation Dice. Rows: Baseline 0.843; Same, another seed 0.835; Dice + cross-entropy loss 0.828; With augmentation 0.858. lead "Two identical runs differ by 0.008. A smaller gain is noise."
19. rules (cards three) — title "How to run an experiment". "Start from a baseline" (txt "a result to compare with"), "Change one thing" (txt "or you will not know what helped"), "Know your noise" (txt "repeat with another seed"). 
20. now (plain) — eyebrow "Now, in the notebook", title "Notebook 03, STEP 8 to 11". .flow: "One training step" -> "The training loop" -> "Curves" -> "Predictions, download the model". lead "Tomorrow, with no seminar: the test set (STEP 12-13), then nodules if you are ahead."
