# models/

Each **sub-folder** of `models/` is one selectable model in the app. A model folder contains exactly:

```
models/
  unet-baseline/
    model.pt       # state_dict (torch.save(model.state_dict(), ...))
    config.json    # architecture + preprocessing (HU window, img_size) + metrics
```

How to fill it:

* unzip `app_starter_kit.zip` (made in notebook 06, day 4) in the app folder: it creates `models/<name>/` and `samples/`;
* to try another model (for example one from your experiments in notebook 05), copy its folder here: it appears in the
  sidebar of the app after a refresh;
* a model is about 6-7 MB, so it can be committed to Git (GitHub refuses single files above 100 MB).

The app (`inference.py`) reads the HU window and the network input size from `config.json`: do not hard-code them.
