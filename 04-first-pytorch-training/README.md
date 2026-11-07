# 04 · First PyTorch Training Pipeline

> Plan: Day 005

This chapter begins by connecting PyTorch Autograd to the small reverse-mode
engine built in [Chapter 03: micrograd](../03-micrograd/README.md). A second,
larger exercise will use those ideas in a handwritten FashionMNIST training
pipeline; it is planned below and has not started.

## What this chapter explains

PyTorch records tensor operations in a dynamic computation graph when gradient
tracking is enabled. Calling backward on a scalar loss applies the chain rule
through that graph and accumulates derivatives in gradient buffers. This
chapter focuses first on the details that make that process predictable:
requires_grad, leaf and non-leaf tensors, grad_fn, backward(), grad,
gradient accumulation and clearing, torch.no_grad(), and detach().

The first exercise uses small scalar and vector computations so the graph and
its gradients can be inspected directly. The later exercise will apply the
same ideas to an MLP trained on FashionMNIST.

## The idea, intuitively

During the forward pass, PyTorch computes values and records the operations
needed to differentiate them. The result is a graph that can be traversed from
a loss back to the tensors that contributed to it. Each operation knows how to
pass a gradient to its inputs; the chain rule combines contributions when a
tensor influences the loss along multiple paths.

A tensor with requires_grad=True asks PyTorch to track operations involving
it. User-created inputs and parameters are typically leaves. Results of tracked
operations are non-leaves. After loss.backward(), PyTorch normally stores grad
on leaf tensors; call retain_grad() before backward if you also need to inspect
a non-leaf's gradient.

For a scalar loss, backward() starts with an upstream gradient of one. For a
non-scalar output, provide an upstream gradient of matching shape; this
computes a vector-Jacobian product. A backward pass frees graph data by
default, so a later pass should build a fresh forward graph unless retaining
the graph is specifically needed.

## Theory-to-Micrograd mapping

| Concept | PyTorch | Corresponding idea in micrograd |
|---|---|---|
| requires_grad | Enables recording of operations that can contribute to gradients. | Every Value operation records its parents and backward rule; there is no opt-in flag. |
| Leaf and non-leaf tensors | A tracked input or parameter is usually a leaf. An operation result is non-leaf. PyTorch populates grad on leaves by default; retain_grad() can retain a non-leaf gradient. | Value objects all have a grad field, so the implementation does not enforce this leaf-only rule. |
| backward() | Starts reverse-mode differentiation at the output and applies the chain rule through the recorded graph. | Value.backward() builds a topological order, seeds the output gradient, and calls each node's _backward rule in reverse order. |
| grad | Holds the gradient accumulated for a tensor. It is normally read from leaf parameters after backward. | Value.grad is the gradient slot updated by each operation's backward closure. |
| Accumulation and clearing | Repeated backward calls add into existing grad values. Clear gradients before a new training step so old contributions do not carry over. | Backward closures use += so all paths contribute. The micrograd training loop clears parameter gradients before the next pass. |
| grad_fn | Refers to the backward function associated with an operation result. | _op labels the operation and _backward stores its propagation rule; PyTorch packages this behavior in its autograd graph. |
| torch.no_grad() | Temporarily stops graph recording inside a context, useful for inference or computations that should not be differentiated. | The current micrograd implementation has no context for turning graph construction off. |
| detach() | Returns a tensor disconnected from the current autograd history. It shares storage with the original tensor. Use detach().clone() when you also need independent storage. | A new Value made from another value's numeric data can break history, but micrograd's scalar values do not model shared tensor storage. |

This comparison is about the concepts, not identical internals. PyTorch supports
tensor operations, devices, and graph behaviors that the small scalar
micrograd engine does not implement.

## Exercise 1: Autograd questions

Open [exercises/autograd_exercises.py](exercises/autograd_exercises.py). It
contains four integrated question prompts and blank answer spaces only. Add
your own code under the prompts and keep written observations as comments.
The questions cover graph tracking and metadata, leaf and non-leaf gradients,
backward propagation, accumulation and clearing, no-grad behavior, detaching,
and shared storage versus cloning.

After adding your answers, run the file from the repository root:

    pixi run python 04-first-pytorch-training/exercises/autograd_exercises.py

Use the official [PyTorch Autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
as the reading for this exercise. Compare the PyTorch behavior you observe with
the Value, backward(), and grad implementation in
[Chapter 03's micrograd code](../03-micrograd/micrograd/micrograd.py).

## Exercise 2: FashionMNIST training pipeline — planned

**Status: Planned for later; not implemented yet.**

In the same chapter, independently write an MLP for FashionMNIST and train it
on MPS. This exercise will include:

- A training and validation split.
- Accuracy and loss curves.
- Saving and loading a model checkpoint.
- Epoch timing on CPU versus MPS.
- Experiments with learning rates 1e-1, 1e-2, and 1e-3, with a
  description of each resulting curve.

We will work through this exercise after completing the Autograd questions.

## References

- [Automatic Differentiation with torch.autograd — PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Chapter 03: micrograd](../03-micrograd/README.md)
