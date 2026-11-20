# Animations

Two GIFs for seminar 2 (models) and for the lab of day 4, when students change the U-Net.

| File | What it shows |
|---|---|
| `stride.gif` | A 3 x 3 filter on a real 8 x 8 nodule crop. With stride 2 the output is 4 x 4, with stride 1 it stays 8 x 8. |
| `unet_stride_resunits.gif` | A nodule patch through our U-Net (`channels=(16, 32, 64, 128)`, `strides=(2, 2, 2)`): 64 → 32 → 16 → 8 and back to 64, with skip connections. Then what is inside one block with `num_res_units=2`, and the number of parameters for 0, 1 and 2 units. |

To rebuild them (needs numpy, matplotlib, nibabel, pandas, pillow):

```bash
python make_gifs.py /path/to/luna16-lung-slices/nodules
```
