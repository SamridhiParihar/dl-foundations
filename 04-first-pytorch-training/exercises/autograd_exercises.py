import torch
import math


# ============================================================
# PyTorch Foundations
# Chapter 04: Autograd



# ============================================================
# Q1. TRACKING A COMPUTATION
# ============================================================
#
# Create a scalar leaf tensor x with value 2.0 that tracks gradients.
# Build these scalar operations from x:
#
#     u = x * x
#     v = u + 3 * x
#     loss = v * v
#
# Before the backward pass:
# - Print requires_grad, is_leaf, and grad_fn for x, u, v, and loss.
# - Call retain_grad() on u.
#
# Call backward() on loss, then inspect x.grad and u.grad.
# Which tensors are leaves and which are non-leaves? Why?
# What does grad_fn tell you about each operation result?
# Work out d(loss)/dx by hand and compare it with x.grad.
# What would you expect u.grad to contain if you had not called retain_grad()?
#
# Your answer:


x = torch.tensor([2.0], requires_grad=True)
u = x*x
u.retain_grad()
v = u + 3*x 
loss = v*v

print("Question 1 : Answers  -------------------------------------")
out = x 
print(f"x:\nRequires grad : {out.requires_grad}\nIs it leaf : {out.is_leaf}\ngrad_fn: {out.grad_fn}")
out = u 
print(f"u:\nRequires grad : {out.requires_grad}\nIs it leaf : {out.is_leaf}\ngrad_fn: {out.grad_fn}")
out = v
print(f"v:\nRequires grad : {out.requires_grad}\nIs it leaf : {out.is_leaf}\ngrad_fn: {out.grad_fn}")
out = loss 
print(f"loss:\nRequires grad : {out.requires_grad}\nIs it leaf : {out.is_leaf}\ngrad_fn: {out.grad_fn}")
# x is leaf tensor and rest aren't because those are intermediate nodes
# grad_fn is a reference to the backward-function object associated with the operation that produced a non-leaf tensor.

loss.backward()
# d(loss)/dx = (d(loss)/dv)*(dv/dx) + (d(loss)/dv)*(dv/du)*(du/dx) = 4*x^3 + 18*x^2 + 18*x = 140.0
print(f"x grad: {x.grad}") # should be 140.0
print(f"u grad : {u.grad}") # if we didn't retain it ..we should  get None along with message below
# message : 
# UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.
#  Its .grad attribute won't be populated during autograd.backward(). 
# If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on
#  the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf 
# Tensor instead. 







# ------------------------------------------------------------


# ============================================================
# Q2. GRADIENT ACCUMULATION AND CLEARING
# ============================================================
#
# Create a fresh scalar leaf tensor x with value 2.0 and gradients enabled.
#
# 1. Build loss_a = x squared, call backward(), and record x.grad.
# 2. Build a new graph for loss_b = 3 * x, call backward(), and record x.grad again.
# 3. Predict each contribution by hand. Why does the second value differ from
#    the gradient contributed by loss_b alone?
# 4. Clear x.grad, build a fresh graph for loss_c = x squared, and call
#    backward() again.
# 5. Show that the final gradient contains only the new pass's contribution.
#
# Which operation or assignment did you use to clear the gradient?
# Why should each new backward pass use a fresh graph here?
#
# Your answer:
print("\nQuestion 2 : Answers  -------------------------------------")

x = torch.tensor([2.0],requires_grad=True)
loss_a = x*x
loss_a.backward()
x_a_grad = x.grad # 4.0
print(f"x_a_grad: {x_a_grad}")
loss_b = 3*x
loss_b.backward()
x_b_grad = x.grad # this should be summation of x_a_grad and due to loss_b (4.0+3.0=7.0)because of += and we didn't make it zero
print(f"x_b_grad: {x_b_grad}")

x.grad.zero_()
loss_c = x*x
loss_c.backward()
x_c_grad = x.grad # 4.0 (since grad is cleared and we calculate just the loss contribute)
print(f"x_c_grad: {x_c_grad}")
# I used grad.zero_() for in place operation to clear the gradient of x
# each new backward pass uses a fresh graph because we didn't do like retain_graph in backward() method






# ------------------------------------------------------------


# ============================================================
# Q3. torch.no_grad() AND detach()
# ============================================================
#
# Create a scalar leaf tensor x with gradients enabled and build y = x * x
# with gradient tracking enabled.
#
# Inside a torch.no_grad() context, build z = x * x. After leaving the context,
# build w = z * 2.
#
# Print requires_grad and grad_fn for y, z, and w. What changed inside the
# context, and did leaving the context change the history of z?
#
# Now create a detached tensor from y, then build a new scalar expression from
# that detached tensor. Inspect requires_grad and grad_fn for both tensors.
# Try calling backward() on the new expression and record what happens.
#
# Does detaching y change x or y? Explain which path can still connect to x.
#
# Your answer:
print("\nQuestion 3 : Answers  -------------------------------------")

x = torch.tensor([2.0], requires_grad= True)
y = x*x

with torch.no_grad():
    z = x*x
w = z*2

out = y 
print(f"y:\nrequires_grad: {out.requires_grad}\ngrad_fn: {out.grad_fn}")
out = z 
print(f"z:\nrequires_grad: {out.requires_grad}\ngrad_fn: {out.grad_fn}")
out = w
print(f"w:\nrequires_grad: {out.requires_grad}\ngrad_fn: {out.grad_fn}")
# Because z was born inside the gradient-disabled block, it remains permanently detached from the
# autograd network even when you return to normal execution space outside the block.


z_new = y.detach()
w_new = z_new*2
out = z_new
print(f"z_new:\nrequires_grad: {out.requires_grad}\ngrad_fn: {out.grad_fn}")
out = w_new
print(f"w_new:\nrequires_grad: {out.requires_grad}\ngrad_fn: {out.grad_fn}")

# Calling .detach() on a tensor does not change x or y at all. Both tensors remain completely 
# intact in memory, and their numerical values do not alter.
# The original y path is the only one that still connects to x.








# ------------------------------------------------------------


# ============================================================
# Q4. detach(): SHARED STORAGE AND clone()
# ============================================================
#
# Create a small tensor with gradients enabled. Make:
# - a detached view of it with detach()
# - a detached independent copy with detach().clone()
#
# For the original tensor, detached view, and detached copy, inspect:
# requires_grad, is_leaf, and grad_fn.
#
# Change one element of the detached view in place. Print all three tensors.
# Which values changed? What does this show about their storage?
#
# Recreate the tensors, change one element of the detached copy in place, and
# print them again. Which values changed this time?
#
# Explain why detach() cuts gradient history but does not by itself make an
# independent copy, and what clone() adds in this example.
#
# Your answer:


print("\nQuestion 4 : Answers  -------------------------------------")

orig = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

view_t = orig.detach()
copy_t = orig.detach().clone()

print("--- Initial Tracking Metadata ---")
print(f"orig   |\n  Requires grad : {orig.requires_grad}\n  Is leaf : {orig.is_leaf}\n  grad_fn : {orig.grad_fn}")
print(f"view_t |\n  Requires grad : {view_t.requires_grad}\n  Is leaf : {view_t.is_leaf}\n  grad_fn : {view_t.grad_fn}")
print(f"copy_t |\n  Requires grad : {copy_t.requires_grad}\n  Is leaf : {copy_t.is_leaf}\n  grad_fn : {copy_t.grad_fn}")

print("\n--- Experiment 1: Changing Detached View in Place ---")
view_t[0] = 99.0
print(f"Original tensor : {orig}")
print(f"Detached view   : {view_t}")
print(f"Detached copy   : {copy_t}")

orig = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
view_t = orig.detach()
copy_t = orig.detach().clone()

print("\n--- Experiment 2: Changing Detached Copy in Place ---")
copy_t[0] = 55.0
print(f"Original tensor : {orig}")
print(f"Detached view   : {view_t}")
print(f"Detached copy   : {copy_t}")






