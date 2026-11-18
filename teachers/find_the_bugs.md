# Find the bugs

An extra for fast students (listed at the end of notebook 03). Give them this training loop:
it runs without errors, but it contains three mistakes. They must find them and explain what each one causes.

```python
for epoch in range(MAX_EPOCHS):

    model.train()
    epoch_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1, keepdim=True)
            dice_metric(y_pred=preds, y=labels)

    val_dice = dice_metric.aggregate().item()
    val_dices.append(val_dice)

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "lung_segmenter.pth")
```

## Solution

1. **`optimizer.zero_grad()` is missing.** The gradients of all the previous batches are added together:
   the updates become larger and larger and the training is unstable or does not converge.
2. **The validation uses `train_loader`.** The "validation" Dice is measured on the training images (with
   augmentation): it looks better than it is and says nothing about new patients.
3. **`dice_metric.reset()` is missing.** The metric keeps the values of all the previous epochs: the Dice
   printed at each epoch is an average over the whole history, so the curve is too smooth and too low.
