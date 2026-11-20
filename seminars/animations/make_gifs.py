"""Two explanatory GIFs: what stride 2 does, and the path of a nodule patch through the U-Net (with residual blocks)."""
import io
import numpy as np, nibabel as nib, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp
from PIL import Image

D = "/kaggle/input/luna16-lung-slices/nodules"
BG, FG, MUTED, TEAL, AMBER, RED = "#FBFAF7", "#1F2A30", "#7A8890", "#159A8C", "#E08A1E", "#C8413B"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "figure.facecolor": BG, "axes.facecolor": BG})

meta = pd.read_csv(f"{D}/metadata.csv")
row = meta[meta.diameter_mm.between(9, 12)].iloc[3]
patch = np.asarray(nib.load(f"{D}/images/{row.nodule_id}.nii").dataobj).astype(float).T
mask = np.asarray(nib.load(f"{D}/labels/{row.nodule_id}.nii").dataobj).T > 0
img = np.clip((patch + 1000) / 1400, 0, 1)

def grab(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")

def save(frames, durations, name):
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(name, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)

def pool(a, n):
    k = a.shape[0] // n
    return a.reshape(n, k, n, k).mean(axis=(1, 3))

# ------------------------------------------------------------------ GIF 1: stride
cy, cx = [int(v) for v in np.argwhere(mask).mean(0)]
c0y, c0x = np.clip(cy - 16, 0, 32), np.clip(cx - 16, 0, 32)
small = pool(img[c0y:c0y + 32, c0x:c0x + 32], 8)            # 8 x 8 input
padded = np.pad(small, 1, mode="edge")
out2 = np.array([[padded[2*i:2*i+3, 2*j:2*j+3].mean() for j in range(4)] for i in range(4)])
out1 = np.array([[padded[i:i+3, j:j+3].mean() for j in range(8)] for i in range(8)])

def stride_frame(stride, step, final=False):
    fig = plt.figure(figsize=(9.6, 5.4))
    fig.text(0.04, 0.92, f"Stride {stride}: the 3 x 3 filter jumps {stride} pixel{'s' if stride > 1 else ''} at a time",
             fontsize=19, weight="bold")
    n_out = 8 // stride
    ax = fig.add_axes([0.05, 0.1, 0.38, 0.68]); ax.imshow(small, cmap="gray", vmin=0, vmax=1, extent=(0, 8, 8, 0))
    ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.grid(color="white", lw=1); ax.tick_params(length=0, labelbottom=False, labelleft=False)
    ax.set_title("input: 8 x 8 pixels", fontsize=14, color=MUTED, pad=8)
    bx = fig.add_axes([0.57, 0.1 + (0.68 - 0.68 * n_out / 8) / 2, 0.38 * n_out / 8, 0.68 * n_out / 8])
    shown = np.full((n_out, n_out), np.nan); out = out2 if stride == 2 else out1
    k = n_out * n_out if final else step + 1
    for t in range(k):
        shown[t // n_out, t % n_out] = out[t // n_out, t % n_out]
    bx.imshow(np.ma.masked_invalid(shown), cmap="gray", vmin=0, vmax=1, extent=(0, n_out, n_out, 0))
    bx.set_facecolor("#E9E6DF"); bx.set_xticks(range(n_out + 1)); bx.set_yticks(range(n_out + 1))
    bx.grid(color="white", lw=1); bx.tick_params(length=0, labelbottom=False, labelleft=False)
    bx.set_title(f"output: {n_out} x {n_out}", fontsize=14, color=MUTED, pad=8); bx.set_xlim(0, n_out); bx.set_ylim(n_out, 0)
    if not final:
        i, j = step // n_out, step % n_out
        ax.add_patch(mp.Rectangle((j * stride - 1, i * stride - 1), 3, 3, fill=False, ec=AMBER, lw=4, clip_on=False))
        bx.add_patch(mp.Rectangle((j, i), 1, 1, fill=False, ec=AMBER, lw=4))
        fig.add_artist(matplotlib.lines.Line2D([0.44, 0.555], [0.45, 0.45], color=AMBER, lw=2.5))
    fig.text(0.04, 0.025, ("same size: stride 1 keeps the image as it is" if stride == 1 else
                           "half the size in each direction: 8 x 8 becomes 4 x 4") if final else
             f"position {step + 1} of {n_out * n_out}", fontsize=15, color=FG if final else MUTED)
    return grab(fig)

frames, dur = [], []
for s in range(16):
    frames.append(stride_frame(2, s)); dur.append(420)
frames.append(stride_frame(2, 0, final=True)); dur.append(2200)
for s in range(0, 64, 4):
    frames.append(stride_frame(1, s)); dur.append(110)
frames.append(stride_frame(1, 0, final=True)); dur.append(2400)
save(frames, dur, "stride.gif")

# ------------------------------------------------------------------ GIF 2: the U-Net path and the residual block
levels = [(64, 1), (32, 16), (16, 32), (8, 64), (8, 128)]
pics = {64: img, 32: pool(img, 32), 16: pool(img, 16), 8: pool(img, 8)}
X_ENC = [1.0, 2.9, 4.6, 6.1]; X_BOT = 7.5; X_DEC = [9.0, 10.8, 12.9]; X_OUT = 13.4
TOP, STEP = 5.9, 1.3
def ypos(level): return TOP - level * STEP

def unet_frame(stage, note):
    fig = plt.figure(figsize=(11.2, 6.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14.6); ax.set_ylim(-0.9, 7.9); ax.axis("off")
    ax.text(0.3, 7.45, "A nodule patch through our U-Net: channels=(16, 32, 64, 128), strides=(2, 2, 2)", fontsize=16, weight="bold")
    def tile(x, lvl, size, ch, color, alpha=1.0):
        s = 0.62 + 1.0 * size / 64
        ax.imshow(pics[size], cmap="gray", vmin=0, vmax=1, extent=(x - s / 2, x + s / 2, ypos(lvl) - s / 2, ypos(lvl) + s / 2), alpha=alpha, zorder=2)
        ax.add_patch(mp.Rectangle((x - s / 2, ypos(lvl) - s / 2), s, s, fill=False, ec=color, lw=3, alpha=alpha, zorder=3))
        ax.text(x, ypos(lvl) - s / 2 - 0.12, f"{size} x {size}\n{ch} channel{'s' if ch > 1 else ''}", ha="center", va="top", fontsize=10.5, color=FG, alpha=alpha)
    def arrow(x0, y0, x1, y1, color, text=None, ls="-"):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, ls=ls, mutation_scale=16), zorder=1)
        if text: ax.text((x0 + x1) / 2 + 0.25, (y0 + y1) / 2 + 0.18, text, fontsize=10.5, color=color, ha="left")
    # input
    ax.text(X_ENC[0], TOP + 0.95, "input", ha="center", fontsize=11, color=MUTED)
    tile(X_ENC[0], 0, 64, 1, MUTED)
    enc = [(32, 16), (16, 32), (8, 64)]
    for k in range(min(stage, 3)):
        x0 = X_ENC[0] if k == 0 else X_ENC[k]
        arrow(x0 + 0.55, ypos(k) - 0.35, X_ENC[k + 1] - 0.5, ypos(k + 1) + 0.3, AMBER, "stride 2")
        tile(X_ENC[k + 1], k + 1, enc[k][0], enc[k][1], TEAL)
    if stage >= 4:
        arrow(X_ENC[3] + 0.45, ypos(3), X_BOT - 0.45, ypos(4) + 0.2, MUTED, "stride 1")
        tile(X_BOT, 4, 8, 128, AMBER)
        ax.text(X_BOT, ypos(4) + 0.55, "bottom", ha="center", fontsize=11, color=AMBER)
    dec = [(16, 32), (32, 16), (64, 2)]
    for k in range(max(0, stage - 4)):
        lvl = 2 - k
        src_x, src_y = (X_BOT, ypos(4)) if k == 0 else (X_DEC[k - 1], ypos(3 - k))
        arrow(src_x + 0.45, src_y + 0.3, X_DEC[k] - 0.45, ypos(lvl) - 0.3, RED)
        ax.text((src_x + X_DEC[k]) / 2 - 0.25, (src_y + ypos(lvl)) / 2 + 0.05, "x 2", fontsize=10.5, color=RED, ha="right")
        size, ch = dec[k]
        tile(X_DEC[k], lvl, size, ch, RED)
        if lvl >= 1:   # the decoder block of this size also receives the encoder output of the same size
            ax.annotate("", (X_DEC[k] - 0.62, ypos(lvl) + 0.05), (X_ENC[lvl] + 0.62, ypos(lvl) + 0.05),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, ls=(0, (4, 3)), mutation_scale=12), zorder=1)
    if stage >= 7:
        ax.text(X_DEC[2], TOP + 0.95, "output: 2 scores per pixel", ha="center", fontsize=11, color=RED)
        ax.text((X_ENC[1] + X_DEC[1]) / 2, ypos(1) + 0.22, "skip connections: the encoder output of the same size", fontsize=10.5, color=MUTED, ha="center")
    ax.text(0.3, -0.6, note, fontsize=14, color=FG)
    return grab(fig)

notes = ["The patch: 64 x 64 pixels, one channel (grey levels).",
         "Encoder block 1, stride 2: half the size (32 x 32), 16 channels.",
         "Encoder block 2, stride 2: 16 x 16, 32 channels.",
         "Encoder block 3, stride 2: 8 x 8, 64 channels. The whole patch is now in view.",
         "Bottom, stride 1: the size stays 8 x 8. Three strides of 2: 64 / 2 / 2 / 2 = 8.",
         "Decoder: each step doubles the size again (transposed convolution).",
         "The skip connections bring back the details lost on the way down.",
         "Back to 64 x 64: one prediction per pixel. The input must be divisible by 2 x 2 x 2 = 8."]
frames, dur = [], []
for st, n in enumerate(notes):
    frames.append(unet_frame(st, n)); dur.append(2300 if st < 7 else 3800)

# residual block, num_res_units=2
def res_frame(phase):
    fig = plt.figure(figsize=(11.2, 6.3)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14.6); ax.set_ylim(-0.6, 7.6); ax.axis("off")
    ax.text(0.3, 7.15, "Inside each block: num_res_units=2", fontsize=16, weight="bold")
    def box(x, y, w, h, text, color, on):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=color if on else "#ECE8E0", ec=color, lw=2.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color="white" if on else FG, weight="bold" if on else None)
    def arrow(p, q, color, on, ls="-"):
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", color=color if on else "#C9C3B8", lw=3 if on else 2, ls=ls, mutation_scale=18))
    ax.imshow(pics[32], cmap="gray", vmin=0, vmax=1, extent=(0.4, 1.9, 3.1, 4.6))
    ax.text(1.15, 2.8, "from the level above", ha="center", fontsize=10, color=MUTED)
    box(3.0, 3.35, 2.9, 1.0, "convolution 3 x 3\nstride 2", TEAL, phase in (1, 4, 5))
    box(7.0, 3.35, 2.9, 1.0, "convolution 3 x 3\nstride 1", TEAL, phase in (2, 4, 5))
    box(3.0, 0.9, 6.9, 0.9, "shortcut: one convolution, stride 2", AMBER, phase in (3, 4, 5))
    ax.add_patch(mp.Circle((11.1, 3.85), 0.42, fc=RED if phase >= 4 else "#ECE8E0", ec=RED, lw=2.5)); ax.text(11.1, 3.85, "+", fontsize=24, ha="center", va="center", color="white" if phase >= 4 else RED)
    arrow((1.95, 3.85), (2.95, 3.85), TEAL, phase >= 1); arrow((5.95, 3.85), (6.95, 3.85), TEAL, phase >= 2); arrow((9.95, 3.85), (10.65, 3.85), TEAL, phase >= 2)
    arrow((2.4, 3.1), (2.95, 1.4), AMBER, phase >= 3); arrow((9.95, 1.35), (10.95, 3.45), AMBER, phase >= 3)
    if phase >= 5:
        ax.imshow(pics[16], cmap="gray", vmin=0, vmax=1, extent=(12.0, 13.5, 3.1, 4.6)); arrow((11.55, 3.85), (11.95, 3.85), RED, True)
        ax.text(12.75, 2.8, "to the next level", ha="center", fontsize=10, color=MUTED)
    txt = {0: "Every encoder block is a small residual block. With num_res_units=0 it would be one convolution only.",
           1: "1. The first convolution has stride 2: it halves the size and changes the number of channels.",
           2: "2. The second convolution keeps the size: one more layer, more capacity.",
           3: "3. The shortcut takes the input around the two convolutions.",
           4: "4. The two paths are added: the convolutions learn what to ADD to the input (the idea of ResNet).",
           5: "Why: more layers, and the gradient also flows through the shortcut, so deeper networks train easily."}[phase]
    ax.text(0.3, -0.35, txt, fontsize=13.5, color=FG)
    ax.text(0.3, 6.35, "Parameters of the nodule model:   num_res_units=0: 162 000    =1: 206 000    =2: 402 000", fontsize=12.5, color=MUTED)
    return grab(fig)

for ph in range(6):
    frames.append(res_frame(ph)); dur.append(2600 if ph < 5 else 4200)
save(frames, dur, "unet_stride_resunits.gif")
print(row.nodule_id)
