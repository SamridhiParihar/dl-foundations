"""
Phase 7 — Visualizing CNN Learning
==================================

Run from the repo root:
    pixi run python 06-cnn/visualize_filters.py

What this script shows, in 4 steps:
    STEP 1  look at the raw numbers inside ONE filter          (printed)
    STEP 2  draw all 16 conv1 filters: trained vs untrained     (picture)
    STEP 3  push ONE image through conv1 and conv2              (printed shapes)
    STEP 4  draw what each filter "saw" in that image           (pictures)

Pictures are saved to 06-cnn/outputs/ and also opened in windows at the end.
"""

# this is generated file ...just for understanding and visualization of filters and feature maps 

import torch
from pathlib import Path
import matplotlib.pyplot as plt

from model import MySimpleCNN
from first_pytorch_training.checkpoint import load_checkpoint
from first_pytorch_training.dataset import test_loader


# =============================================================================
# STEP 0 — load the trained model  (your code, unchanged)
# =============================================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = MySimpleCNN()          # empty network: random weights

checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "cnn_fashion_mnist.pth"
)

model = load_checkpoint(model, checkpoint_path, DEVICE)   # random weights -> trained weights
model.eval()                                              # we are only looking, not training

print(model.conv1.weight.shape)
# prints: torch.Size([16, 1, 3, 3])
#                      │   │  └──┴── each filter is a 3×3 grid of numbers
#                      │   └──────── each filter looks at 1 input channel (grayscale image)
#                      └──────────── there are 16 filters  (out_channels=16 in model.py)

# Folder where all pictures will be saved: 06-cnn/outputs/
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)       # create it if missing; do nothing if it exists


# =============================================================================
# STEP 1 — a filter is just 9 numbers. Print one.
# =============================================================================

# model.conv1.weight is the tensor that training changed.
#   .detach() -> drop autograd tracking (we won't backprop through this)
#   .cpu()    -> copy from the Mac GPU (mps) to normal memory, so we can print/plot it
weights = model.conv1.weight.detach().cpu()      # shape (16, 1, 3, 3)

filter_0 = weights[0, 0]                         # filter number 0, channel 0  -> shape (3, 3)

print("\nSTEP 1 — the 9 numbers inside filter f0:")
print(filter_0)
# example of what you might see (your numbers will differ):
#   tensor([[-0.31,  0.05,  0.42],
#           [-0.28,  0.10,  0.39],
#           [-0.35, -0.02,  0.44]])
# Negative column on the left, positive column on the right
# -> this filter fires on "dark on the left, bright on the right" = a vertical edge.
# In STEP 2 these exact 9 numbers become 9 coloured squares:
#   negative -> blue,   zero -> white,   positive -> red.


# =============================================================================
# STEP 2 — draw all 16 filters as tiny 3×3 pictures
# =============================================================================

def plot_filters(model, title, filename):
    w = model.conv1.weight.detach().cpu()        # (16, 1, 3, 3) — same as STEP 1

    # Biggest absolute weight, e.g. 0.8 if weights range from -0.8 to +0.5.
    # Used so that the colour scale goes from -0.8 (blue) to +0.8 (red)
    # and white is ALWAYS exactly 0, in all 16 pictures.
    lim = w.abs().max()

    # 16 empty plotting areas arranged as 2 rows × 8 columns
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))

    # axes.flat walks the 2×8 grid as 16 plots in a row: plot 0, plot 1, ... plot 15
    # i = 0..15 picks the matching filter, ax = the plot it is drawn into
    for i, ax in enumerate(axes.flat):
        one_filter = w[i, 0]                     # (3, 3)  — 9 numbers, like filter_0 above
        ax.imshow(one_filter,                    # draw the 3×3 grid: one coloured square per number
                  cmap="bwr",                    # blue = negative, white = 0, red = positive
                  vmin=-lim, vmax=lim)           # same colour scale for every filter
        ax.set_title(f"f{i}", fontsize=8)        # label: f0, f1, ... f15
        ax.axis("off")                           # hide tick marks (0,1,2 mean nothing here)

    fig.suptitle(title)                          # one title above all 16 small plots
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    print(f"saved {OUT / filename}")


print("\nSTEP 2 — drawing filters")

# Your trained model: should show structure (red on one side, blue on the other).
plot_filters(model, "conv1 filters — TRAINED", "conv1_filters_trained.png")

# A brand-new MySimpleCNN() has never seen data: its weights are random.
# Same function, different model -> compare the two pictures side by side.
plot_filters(MySimpleCNN(), "conv1 filters — UNTRAINED (random)", "conv1_filters_untrained.png")


# =============================================================================
# STEP 3 — take ONE image and push it through conv1 and conv2 by hand
# =============================================================================

# test_loader gives batches. next(iter(...)) = "give me just the first batch".
images, labels = next(iter(test_loader))
print("\nSTEP 3 — one batch:", images.shape)
# prints: torch.Size([64, 1, 28, 28])  -> 64 images, 1 channel, 28×28 pixels

IMAGE_INDEX = 0                                   # change this to look at other images

# [IMAGE_INDEX : IMAGE_INDEX+1] keeps the batch dimension:
#   images[0]    -> (1, 28, 28)     conv1 would complain: no batch dimension
#   images[0:1]  -> (1, 1, 28, 28)  a "batch" containing exactly one image  ✓
x = images[IMAGE_INDEX:IMAGE_INDEX + 1].to(DEVICE)   # model is on DEVICE, so the image must be too

class_names = test_loader.dataset.classes             # ['T-shirt/top', 'Trouser', ..., 'Ankle boot']
true_label = class_names[labels[IMAGE_INDEX]]
print("image shape:", x.shape, "| true class:", true_label)

with torch.no_grad():                    # no gradients needed: we are only looking
    # Normally model(x) runs forward() and returns only the 10 final scores;
    # everything in between is thrown away.
    # Here we copy the first lines of forward() ourselves, so we can KEEP
    # the in-between results:
    #
    #     forward():   x = self.relu(self.conv1(x))
    #                  x = self.relu(self.conv2(x))
    #                  ... pool, flatten, linear
    #
    a1 = model.relu(model.conv1(x))      # (1, 16, 28, 28)  -> 16 feature maps
    a2 = model.relu(model.conv2(a1))     # (1, 32, 28, 28)  -> 32 feature maps

    prediction = class_names[model(x).argmax(dim=1).item()]   # full model, for the title

print("after conv1 + relu:", a1.shape)   # 16 filters  -> 16 maps, still 28×28 (padding=1)
print("after conv2 + relu:", a2.shape)   # 32 filters  -> 32 maps, still 28×28
print("model predicts:", prediction)

# Each feature map is a 28×28 grid: value at (row, col) = how strongly that
# filter's pattern appears at that spot in the image. ReLU made negatives 0,
# so: 0 = black = "pattern not here", bright = "pattern found here".
print("map 0 of conv1 — min/max:", a1[0, 0].min().item(), "/", a1[0, 0].max().item())


# =============================================================================
# STEP 4 — draw the input image and what the filters "saw" in it
# =============================================================================

# The input image.
# dataset.py normalised pixels with (p - 0.5) / 0.5, mapping 0..1 to -1..+1.
# To display it we undo that:  p * 0.5 + 0.5  maps -1..+1 back to 0..1.
img = images[IMAGE_INDEX, 0] * 0.5 + 0.5          # [image, channel] -> (28, 28)

fig = plt.figure(figsize=(3, 3))
plt.imshow(img, cmap="gray")                      # 0 = black, 1 = white
plt.title(f"input: {true_label}\npredicted: {prediction}")
plt.axis("off")
fig.savefig(OUT / "input_image.png", dpi=150, bbox_inches="tight")
print(f"\nSTEP 4 — saved {OUT / 'input_image.png'}")


def plot_feature_maps(maps, rows, cols, title, filename):
    maps = maps[0].cpu()        # drop the batch dim: (1, N, 28, 28) -> (N, 28, 28); back to CPU
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.4))
    for i, ax in enumerate(axes.flat):
        # maps[i] is ONE 28×28 feature map = filter i's answer for this image.
        # No vmin/vmax here: after ReLU everything is >= 0, and we let each map use
        # its own range so even weak responses are visible.
        ax.imshow(maps[i], cmap="viridis")        # dark purple = 0, yellow = strongest
        ax.set_title(f"{i}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    print(f"saved {OUT / filename}")


# conv1: 16 maps -> 2 rows × 8. Expect: still looks like the clothing, edges highlighted.
plot_feature_maps(a1, 2, 8, f"conv1 feature maps — {true_label}", "conv1_feature_maps.png")

# conv2: 32 maps -> 4 rows × 8. Expect: blobbier, more abstract (patterns of edges).
plot_feature_maps(a2, 4, 8, f"conv2 feature maps — {true_label}", "conv2_feature_maps.png")

# A completely dark map = that filter found nothing in this image.
# If a filter is dark for EVERY image, it is a "dead" filter.

plt.show()        # open all figures in windows (close them to end the script)
