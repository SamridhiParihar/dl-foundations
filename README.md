# dl-foundations

Deep learning from first principles, in NumPy and PyTorch — part of my Year One plan (RRC, IIIT Hyderabad).
Each chapter: concept → maths → small example → implementation → exercises → notes.

| # | Chapter | Plan days |
|---|---|---|
| 01 | [NumPy and Tensors](01-numpy-and-tensors/) | Day 001 |
| 02 | [Backpropagation by Hand](02-backprop-by-hand/) | Day 002 |
| 03 | [micrograd — an Autograd Engine](03-micrograd/) | Days 003–004 |
| 04 | [First PyTorch Training Pipeline](04-first-pytorch-training/) | Day 005 |

## Setup

```bash
pixi install
pixi run python -c "import torch; print(torch.backends.mps.is_available())"
```

`common/` holds small helpers reused across chapters. Datasets go in `data/` and run outputs in each chapter's `outputs/`; both are git-ignored.
