# 04 · First PyTorch Training Pipeline

> **Progress:** Phases 1–9 complete. This document records the FashionMNIST pipeline, its implementation choices, and the CPU vs MPS benchmark. The README milestone is documented here as Phase 10.

This chapter builds on the Autograd concepts in [Chapter 03 · micrograd](../03-micrograd/README.md) and the exercises in [exercises/autograd_exercises.py](exercises/autograd_exercises.py). It then applies those ideas to a complete FashionMNIST image-classification workflow: prepare the data, define a model, train and evaluate it, compare learning rates, save and reload weights, and measure CPU and Apple GPU training time.

## At a glance

- **Task:** classify grayscale clothing images into 10 FashionMNIST categories.
- **Model:** a fully connected multilayer perceptron (MLP).
- **Training:** mini-batches of 64, Adam, cross-entropy loss, learning rate 0.001.
- **Data split:** 54,000 training examples, 6,000 validation examples, and the official 10,000-example test set.
- **Benchmark:** five training epochs on CPU and MPS using the same model and configuration.
- **Measured benchmark result:** CPU 14.54 seconds; MPS 26.59 seconds; MPS speedup 0.55x. CPU was faster for this workload.

## How the pipeline fits together

A batch of images is transformed and loaded by a DataLoader. The model maps each image to ten class scores, or logits. Cross-entropy compares those scores with the integer labels. During training, Autograd computes gradients and Adam updates the model parameters. Validation and test evaluation run without gradient tracking. Checkpoints store the learned parameter values so a new model instance can restore them.

## Completed phases

### Phase 1 — Prepare FashionMNIST

Implemented in [dataset.py](dataset.py).

- TorchVision downloads and loads FashionMNIST into the repository's data directory.
- The official training partition has 60,000 examples; the official test partition has 10,000.
- The 60,000 training examples are split into 54,000 training and 6,000 validation examples with a fixed random-split seed of 42.
- ToTensor converts each 28 × 28 grayscale image to a floating-point tensor. Normalize with mean 0.5 and standard deviation 0.5 maps the usual pixel range from approximately [0, 1] to [-1, 1].
- A batch contains up to 64 images with shape [64, 1, 28, 28] and integer class labels.
- Training batches are shuffled. Validation and test batches are not. The training DataLoader uses a seeded generator for repeatable ordering.

**Why this matters:** training examples are used to update weights, validation examples help compare training choices, and the test set is kept for a final evaluation. The code uses the test set separately from the training and validation loaders.

### Phase 2 — Define the MLP

Implemented in [model.py](model.py).

The model flattens each image and applies these layers:

| Layer | Shape |
|---|---:|
| Flatten image | 784 features |
| Linear + ReLU | 784 → 128 |
| Linear + ReLU | 128 → 64 |
| Linear output | 64 → 10 logits |

The final layer returns raw logits; it does not apply softmax. CrossEntropyLoss consumes logits directly and combines the softmax calculation with the classification loss. The architecture has 109,386 trainable parameters, calculated from the layer dimensions and biases.

### Phase 3 — Train one epoch at a time

Implemented in [training.py](training.py).

For every training batch, the loop moves images and labels to the selected device, runs a forward pass, computes cross-entropy loss, clears the previous gradients, calls backward, and steps the Adam optimizer. The reported epoch loss is weighted by the number of examples in each batch, so a smaller final batch is handled correctly.

The default training configuration is:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs in the training and validation example | 5 |

Clearing gradients before each backward pass matters because PyTorch accumulates gradients by default. The optimizer update is separate from the gradient calculation: backward computes the gradients, and optimizer.step changes the parameters.

### Phase 4 — Evaluate on validation data

Implemented in [evaluation.py](evaluation.py), and called after each training epoch by [training.py](training.py).

Evaluation switches the model to evaluation mode and uses torch.no_grad so it does not build a backward graph. It calculates average loss and classification accuracy. Predicted labels come from the index of the largest logit.

model.eval() changes behavior for modules that have train/eval-specific behavior; torch.no_grad() disables gradient tracking. They solve different problems and are used together here.

### Phase 5 — Plot training and validation loss

Implemented in [plotting.py](plotting.py).

The training script plots training loss and validation loss across epochs. Keeping both curves visible helps reveal whether training loss is improving and whether validation behavior follows it. The current code displays plots interactively with Matplotlib; it does not save the figures to files.

No specific training or validation loss values are recorded in the available run notes, so this document does not invent them.

### Phase 6 — Compare learning rates

Implemented in [learning_rate_experiments.py](learning_rate_experiments.py), with curves created by [plotting.py](plotting.py).

The experiment compares learning rates 0.1, 0.01, and 0.001 for five epochs each. Each trial creates a fresh model and optimizer. The code resets the model seed and training-loader generator to 42 before each trial so the runs start with the same initialization and batch order.

**Expected behavior, not a recorded result:** a larger learning rate makes larger parameter updates. It may reduce loss quickly, but can also make training unstable; a smaller rate often changes weights more cautiously and may need more epochs. The actual per-epoch values and plotted curves are not preserved in the available run notes, so no winner is claimed here.

### Phase 7 — Save, reload, and verify a checkpoint

Implemented in [checkpoint.py](checkpoint.py) and [verify_checkpoint.py](verify_checkpoint.py).

The checkpoint code saves the model state dictionary: the learned parameter tensors, not the Python model class itself. To load it, the same model architecture is instantiated and the saved state is mapped to the requested device. The project contains checkpoint files under checkpoints/.

The verification script trains a model, records logits for a test batch, saves the weights, loads them into a fresh model, and checks that the restored logits are close to the original logits. It also counts agreement between predicted classes. This is the intended check; the exact agreement percentage is not included in the available run notes.

### Phase 8 — Evaluate the saved model on the test set

Implemented in [test_evaluation.py](test_evaluation.py), using the held-out test loader from [dataset.py](dataset.py).

This step loads a checkpoint and reports test loss and accuracy. Test data is for a final performance estimate, not for selecting the learning rate or tuning the model. The exact test loss and accuracy from the user's run are not present in the available conversation or project notes, so they are left unreported rather than guessed.

### Phase 9 — Benchmark CPU vs MPS

Implemented in [benchmark.py](benchmark.py).

The device check succeeded on the user's M1 Pro Mac:

~~~text
CPU device: cpu
MPS device: mps
MPS available: True
~~~

The benchmark measures training time, not accuracy. It uses the same FashionMNIST training loader, MLP, batch size, optimizer, learning rate, and number of epochs on both devices. Each device gets a disposable warm-up model and a separate fresh model for the timed run. The random seed and batch order are reset so both timed runs start comparably. MPS is synchronized around each timed epoch because GPU work can otherwise finish asynchronously after the timer stops.

| Epoch | CPU | MPS |
|---:|---:|---:|
| 1 | 2.98 s | 5.44 s |
| 2 | 2.83 s | 5.25 s |
| 3 | 2.98 s | 5.28 s |
| 4 | 2.88 s | 5.33 s |
| 5 | 2.88 s | 5.28 s |
| **Total reported** | **14.54 s** | **26.59 s** |

**MPS speedup = CPU total ÷ MPS total = 14.54 ÷ 26.59 = 0.55x.**

A speedup below 1.0x means MPS took longer. In this run, CPU completed the five epochs in 14.54 seconds while MPS took 26.59 seconds; MPS took about 1.83 times as long. The epoch values are rounded to two decimal places, while totals come from the underlying timings, so adding the displayed rows may differ from the reported totals by 0.01 seconds.

**Interpretation:** MPS was slower for this small FashionMNIST MLP workload. This is a measured result for this machine and configuration, not a general claim that MPS is slower than CPU. A likely explanation is that this small model and batch size do not provide enough computation to offset GPU dispatch and data-transfer overhead. The benchmark did not separately measure those overheads, so that explanation is an inference.

## What is measured and what is expected

| Type | Recorded information |
|---|---|
| **Measured and supplied from the run** | CPU/MPS availability; five epoch times per device; reported totals; 0.55x speedup. |
| **Confirmed by the project code** | Dataset split, preprocessing, MLP architecture, optimizer, evaluation logic, learning-rate values, checkpoint workflow, and benchmark controls. |
| **Expected behavior, not a recorded metric** | How the learning-rate curves compare; the final validation/test loss and accuracy; the checkpoint agreement percentage; why this MPS run was slower. |

The implementation produces training, validation, test, and checkpoint-verification metrics when run, but those numeric outputs are not present in the retained run notes. The benchmark table above is the only set of model-run measurements reproduced here.

## Reproduce the project scripts

Run commands from the repository root so the data directory and checkpoint paths resolve as intended. FashionMNIST is downloaded on the first run if it is not already present.

~~~bash
pixi run python 04-first-pytorch-training/dataset.py
pixi run python 04-first-pytorch-training/training.py
pixi run python 04-first-pytorch-training/learning_rate_experiments.py
pixi run python 04-first-pytorch-training/verify_checkpoint.py
pixi run python 04-first-pytorch-training/test_evaluation.py
pixi run python 04-first-pytorch-training/benchmark.py
~~~

The benchmark intentionally raises an error if MPS is unavailable; it does not silently fall back to CPU and accidentally compare CPU against itself.

## Project files

| File | Purpose |
|---|---|
| [dataset.py](dataset.py) | FashionMNIST download, transforms, split, and DataLoaders |
| [model.py](model.py) | MLP definition |
| [training.py](training.py) | Training and validation orchestration |
| [evaluation.py](evaluation.py) | Loss and accuracy calculation |
| [plotting.py](plotting.py) | Training/validation and learning-rate plots |
| [learning_rate_experiments.py](learning_rate_experiments.py) | Controlled learning-rate trials |
| [checkpoint.py](checkpoint.py) | Save and load model state dictionaries |
| [verify_checkpoint.py](verify_checkpoint.py) | Compare predictions before and after checkpoint reload |
| [test_evaluation.py](test_evaluation.py) | Evaluate a saved checkpoint on the test set |
| [benchmark.py](benchmark.py) | CPU vs MPS training-time benchmark |
| [exercises/autograd_exercises.py](exercises/autograd_exercises.py) | Autograd questions and worked code |

## Key concepts carried forward from Autograd

PyTorch records tensor operations in a dynamic graph when gradient tracking is enabled. backward applies the chain rule and accumulates gradients in leaf tensors. Gradients need to be cleared before the next optimizer update. torch.no_grad is useful for evaluation because evaluation does not need gradients. detach cuts a tensor's gradient history; detach alone still shares storage with the original tensor, while detach().clone() makes an independent copy.

These ideas connect to Chapter 03 · micrograd: both systems record operations and propagate derivatives in reverse. PyTorch provides this behavior for tensors, batches, and hardware devices, while micrograd makes the scalar operations easier to inspect.

