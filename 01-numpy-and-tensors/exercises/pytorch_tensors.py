
# ============================================================
# PyTorch Foundations
# Chapter 03: Tensors
#
# Topics:
# 1. Tensor creation and dimensions
# 2. Shape and dtype
# 3. Devices (CPU / MPS)
# 4. NumPy bridge and shared memory
# 5. In-place operations
#
# Instructions:
# - Solve every question independently.
# - Write code below each question.
# - Print outputs wherever required.
# - Add your own observations in comments.
# - Do not skip questions.
# ============================================================

import torch
import numpy as np


# ============================================================
# PHASE 1: TENSOR CREATION AND DIMENSIONS
# ============================================================

# Q1. Create the following tensors:
#
# a. A scalar containing 25.
# b. A 1D tensor containing [10, 20, 30, 40].
# c. A 2D tensor containing:
#       [[1, 2, 3],
#        [4, 5, 6]]
# d. A 3D tensor containing 2 matrices, each of shape (2, 3).
#
# Print all four tensors.
#
# Answer:

scalar = torch.tensor(25)
data = [10,20,30,40]
vector = torch.tensor(data)
data = [[1, 2, 3],
        [4, 5, 6]]
matrix = torch.tensor(data)
tensor_3d = torch.ones((2,2,3))

print("Question 1 answers :----------------------------")
print(f"sc1: {scalar}")
print(f"sc2: {vector}")
print(f"sc3: {matrix}")
print(f"sc4: {tensor_3d}")

# ------------------------------------------------------------

# Q2. Create the following tensors using PyTorch functions:
#
# a. A tensor of shape (3, 4) filled with zeros.
# b. A tensor of shape (2, 3) filled with ones.
# c. A tensor of shape (3, 3) containing random values
#    between 0 and 1.
# d. A tensor of shape (2, 4) containing normally
#    distributed random values.
#
# Print all tensors.
#
# Answer:


tensor_zeors2D = torch.zeros((3,4))
tensor_ones2D = torch.ones((2,3))
tensor_rand2D = torch.rand((2,4))
tensor_randn2D = torch.randn((2,4))
print("\nQuestion 2 answers :----------------------------")
print(f"t1: {tensor_zeors2D} ")
print(f"t2: {tensor_ones2D} ")
print(f"t3: {tensor_rand2D} ")
print(f"t4: {tensor_randn2D} ")


# ------------------------------------------------------------

# Q3. Create a tensor X with shape (3, 4).
#
# a. Create another tensor using ones_like(X).
# b. Create another tensor using zeros_like(X).
#
# Print all three tensors.
#
# Question:
# What properties do the newly created tensors inherit
# from X?
#
# Answer:

X = torch.rand((3,4))
tensor_ones2D = torch.ones_like(X)
tensor_zeros2D = torch.zeros_like(X)
print("\nQuestion 3 answers :----------------------------")
print(f"t1: {tensor_ones2D}")
print(f"t2: {tensor_zeros2D}")
print(f"t3: {X}")
print(X.device)
# the newly created tensors inherit the property shape, and by default dtype, and device from the X


# ------------------------------------------------------------

# Q4. Consider a tensor with the following shape:
#
#       (4, 3, 2)
#
# Without running any code, predict:
#
# a. How many dimensions does it have?
# b. How many total elements does it contain?
#
# Now create a tensor with this shape and verify your
# predictions using PyTorch.
#
# Answer:
# 3 dimensions , (4*3*2=24) 
print("\nQuestion 4 answers :----------------------------")
X = torch.rand((4,3,2))
print(f"shape : {X.shape}")

# ------------------------------------------------------------

# Q5. Create a tensor containing 12 elements.
#
# Reshape it into:
# a. (3, 4)
# b. (2, 6)
# c. (2, 2, 3)
#
# Print the shapes and values.
#
# Question:
# Why must the total number of elements remain the same
# when reshaping a tensor?
#
# Answer:
X = [0,1,2,3,4,5,6,7,8,9,10,11]
tensor_x = torch.tensor(X)
tensor_a = tensor_x.reshape((3,4))
tensor_b = tensor_x.reshape((2,6))
tensor_c = tensor_x.reshape((2,2,3))
print("\nQuestion 5 answers :----------------------------")
print(f"a shape : {tensor_a.shape}")
print(tensor_a)
print(f"b shape : {tensor_b.shape}")
print(tensor_b)
print(f"c shape : {tensor_c.shape}")
print(tensor_c)
# because total no of elements remains same i.e, 12 


# ============================================================
# PHASE 2: SHAPE, DTYPE AND TENSOR ATTRIBUTES
# ============================================================

# Q6. Create three tensors:
#
# a. [1, 2, 3, 4]
# b. [1.0, 2.0, 3.0, 4.0]
# c. [1, 2, 3, 4] explicitly using float64.
#
# Print their:
# - Values
# - Shape
# - Dtype
# - Number of dimensions
# - Number of elements
#
# Answer:
print("\nQuestion 6 answers :----------------------------")
tensor_a = torch.tensor([1,2,3,4])
tensor_b = torch.tensor([1.0, 2.0, 3.0 , 4.0])
tensor_c = torch.tensor(
    [1,2,3,4],
    dtype = torch.float32
)

print(f"a shape : {tensor_a.shape}")
print(f"a data type : {tensor_a.dtype}")
print(f"a dimesnion : {tensor_a.ndim}")
print(f"a no of ele : {tensor_a.numel()}")
print(tensor_a)

print(f"b shape : {tensor_b.shape}")
print(f"b data type : {tensor_b.dtype}")
print(f"b dimesnion : {tensor_b.ndim}")
print(f"b no of ele : {tensor_b.numel()}")
print(tensor_b)

print(f"c shape : {tensor_c.shape}")
print(f"c data type : {tensor_c.dtype}")
print(f"c dimesnion : {tensor_c.ndim}")
print(f"c no of ele : {tensor_c.numel()}")
print(tensor_c)

# ------------------------------------------------------------

# Q7. Create an integer tensor:
#
#       [10, 20, 30, 40]
#
# Convert it into:
# a. float32
# b. float64
#
# Print the dtype before and after conversion.
#
# Question:
# Does changing dtype change the tensor's shape?
#
# Answer:

print("\nQuestion 7 answers :----------------------------")
X = torch.tensor([10,20,30,40])
a = X.to(torch.float32)
b = X.to(torch.float64)
print("Before : ")
print("X:", X, "| Type:", X.dtype)
print("After : ")
print("a:", a, "| Type:", a.dtype)
print("b:", b, "| Type:", b.dtype)
# no it doesn't change the tensor's shape


# ------------------------------------------------------------

# Q8. Create a float32 tensor containing 1 million elements.
#
# Calculate the memory occupied by its data in bytes.
#
# Repeat the same exercise using float64.
#
# Question:
# a. How much memory does each tensor occupy?
# b. What is the difference in memory consumption?
# c. Why does dtype selection matter in deep learning?
#
# Hint: Consider the number of bytes occupied by each
# datatype and the number of elements.
#
# Answer:
print("\nQuestion 8 answers :----------------------------")
zeros_tensor32 = torch.zeros(1000000, dtype=torch.float32)
zeros_tensor64 = torch.zeros(1000000, dtype=torch.float64)
print(f"a ele memeory size : {(zeros_tensor32.element_size())/1e6} mb")
print(f"a ele memeory size : {(zeros_tensor64.element_size())/1e6} mb")
print(f"a memeory size : {(zeros_tensor32.numel()*zeros_tensor32.element_size())/1e6} mb")
print(f"b memeory size : {(zeros_tensor64.numel()*zeros_tensor64.element_size())/1e6} mb")
# a occupy 4mb whereas b occupy 8mb
# twice
# in dl precision vs efficiency tradeoff is there ig



# ------------------------------------------------------------

# Q9. Create a tensor X with shape (2, 3).
#
# Print:
# - X.shape
# - X.dtype
# - X.device
# - X.ndim
# - X.numel()
#
# Question:
# Explain what each of these attributes represents.
#
# Answer:
X = torch.randn(2, 3)
print("\nQuestion 9 answers :----------------------------")
print("X.shape :", X.shape) # shape of tensor
print("X.dtype :", X.dtype) # data type of tensor
print("X.device:", X.device) # tensor stored on which device 
print("X.ndim  :", X.ndim) # dimension of tensor
print("X.elements  :", X.numel()) # no of elements in tensor




# ============================================================
# PHASE 3: DEVICES - CPU, CUDA AND MPS
# ============================================================

# Q10. Create a tensor on the CPU.
#
# Print its device.
#
# Now check:
# - Is CUDA available?
# - Is MPS available?
#
# Print the results.
#
# Answer:
X = torch.randn(2, 3)
print("\nQuestion 10 answers :----------------------------")
print(f"X device : {X.device}")

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()

print(f"CUDA Available: {cuda_available}")
print(f"MPS Available:  {mps_available}")




# ------------------------------------------------------------

# Q11. Create a tensor containing:
#
#       [10, 20, 30, 40]
#
# Move the tensor to MPS if available.
# Otherwise, keep it on CPU.
#
# Print:
# - Tensor values
# - Tensor device
#
# Answer:

X = torch.tensor([10,20,30,40,50])
mps_available = torch.backends.mps.is_available()

if mps_available : 
    X = X.to(device = 'mps')
else : 
    X = X.to(device = 'cpu')
print("\nQuestion 11 answers :----------------------------")
print(X)
print(X.device)


# ------------------------------------------------------------

# Q12. Create two tensors:
#
# a. One on CPU.
# b. One on MPS (if available).
#
# Try performing addition between them.
#
# Observe what happens.
#
# Question:
# Why can tensors on different devices cause an error?
# 
# Now move both tensors to the same device and perform
# the addition successfully.
#
# Answer:
X = torch.rand((3,4) , device = 'cpu')
Y = torch.rand((3,4) , device = 'mps')
# Z = X + Y
# error : RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
# because chips can only compute data that lives in their own dedicated memory
print("\nQuestion 12 answers :----------------------------")
X = X.to(device='mps')
Z = X+Y
print(Z)
# ------------------------------------------------------------

# Q13. Create a tensor on CPU.
#
# Convert its datatype to float64.
#
# Then move it to MPS if available.
#
# Print its final:
# - Device
# - Dtype
# - Shape
#
# Question:
# Can .to() be used for both device and dtype conversion?
#
# Answer:


x = torch.tensor([1, 2, 3], device="cpu")

device = "mps" if torch.backends.mps.is_available() else "cpu"
# x = x.to(dtype=torch.float64, device=device)
#TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
print("\nQuestion 13 answers :----------------------------")
print(f"Device: {x.device}")
print(f"Dtype:  {x.dtype}")
print(f"Shape:  {x.shape}")

# yes to can be used 


# ------------------------------------------------------------

# Q14. Create a tensor X on CPU.
#
# Call X.to("mps") without assigning the result back to X.
#
# If MPS is available, inspect X.device.
#
# Now assign the result back to X and inspect again.
#
# Question:
# What did you observe?
# Why is assigning the result of .to() important?
#
# Answer:

print("\nQuestion 14 answers :----------------------------")
X = torch.tensor([1.0, 2.0, 3.0], device="cpu")
if torch.backends.mps.is_available():
    X.to("mps")
    print(f"Device after unassigned .to(): {X.device}")  

if torch.backends.mps.is_available():
    X = X.to("mps")
    print(f"Device after assigned .to():   {X.device}")  
# it's not inplace operation so until we assign it to a new variable it will be on cpu itself 


# ============================================================
# PHASE 4: NUMPY BRIDGE AND SHARED MEMORY
# ============================================================

# Q15. Create a NumPy array:
#
#       [10, 20, 30, 40]
#
# Convert it into a PyTorch tensor using:
# torch.from_numpy()
#
# Print both objects.
#
# Also print their respective datatypes.
#
# Answer:

np_arr = np.array([10, 20, 30, 40])
pt_tensor = torch.from_numpy(np_arr)

print("\nQuestion 15 answers :----------------------------")
print(f"NumPy Array: {np_arr}, Dtype: {np_arr.dtype}")
print(f"PyTorch Tensor: {pt_tensor}, Dtype: {pt_tensor.dtype}")

# ------------------------------------------------------------

# Q16. MEMORY SHARING: NUMPY -> PYTORCH
#
# Using the NumPy array and PyTorch tensor from Q15:
#
# a. Modify the first element of the NumPy array to 999.
# b. Print the PyTorch tensor.
#
# Question:
# Did the PyTorch tensor also change?
# Why?
#
# Answer:

np_arr[0] = 999
print("\nQuestion 16 answers :----------------------------")
print(f"Modified NumPy Array: {np_arr}")
print(f"PyTorch Tensor:        {pt_tensor}")
# Observation: Yes, the PyTorch tensor also changed.
# Reason: torch.from_numpy() creates a wrapper that shares the exact same underlying memory block in RAM.


# ------------------------------------------------------------

# Q17. MEMORY SHARING: PYTORCH -> NUMPY
#
# Create a fresh NumPy array and convert it into a
# PyTorch tensor using torch.from_numpy().
#
# a. Modify the second element of the PyTorch tensor.
# b. Print the NumPy array.
#
# Question:
# Did the NumPy array reflect the modification?
#
# Answer:

np_arr_fresh = np.array([100, 200, 300])
pt_tensor_fresh = torch.from_numpy(np_arr_fresh)
pt_tensor_fresh[1] = 888

print("\nQuestion 17 answers :----------------------------")
print(f"Modified PyTorch Tensor: {pt_tensor_fresh}")
print(f"NumPy Array:             {np_arr_fresh}")
# Observation: Yes, the NumPy array reflected the modification because the memory link works bi-directionally.


# ------------------------------------------------------------

# Q18. Compare torch.tensor() and torch.from_numpy().
#
# a. Create a NumPy array.
# b. Convert it into a PyTorch tensor using torch.tensor().
# c. Modify the original NumPy array.
# d. Print the PyTorch tensor.
#
# Repeat the same process using torch.from_numpy().
#
# Question:
# What difference did you observe?
# Which operation shares memory and which creates a copy?
#
# Answer:

arr_a = np.array([1, 2, 3])
tensor_a = torch.tensor(arr_a)
arr_a[0] = 99

arr_b = np.array([1, 2, 3])
tensor_b = torch.from_numpy(arr_b)
arr_b[0] = 99

print("\nQuestion 18 answers :----------------------------")
print(f"Using torch.tensor():     {tensor_a}")
print(f"Using torch.from_numpy(): {tensor_b}")
# Observation: Modifying the source array does not alter the torch.tensor() output, but it does alter the torch.from_numpy() output.
# Conclusion: torch.tensor() creates a completely isolated memory copy, whereas torch.from_numpy() shares the active memory space.


# ------------------------------------------------------------

# Q19. Convert a CPU PyTorch tensor into a NumPy array
# using .numpy().
#
# a. Print both objects.
# b. Modify one element of the NumPy array.
# c. Print the original PyTorch tensor.
#
# Question:
# Does the tensor reflect the NumPy modification?
# Explain why.
#
# Answer:

pt_cpu = torch.tensor([5, 10, 15])
np_from_pt = pt_cpu.numpy()
np_from_pt[0] = 555

print("\nQuestion 19 answers :----------------------------")
print(f"Modified NumPy Bridge: {np_from_pt}")
print(f"Original PyTorch CPU:  {pt_cpu}")
# Observation: Yes, the tensor reflects the modification.
# Reason: The .numpy() method creates a direct link back to the CPU memory block, allowing shared mutations.


# ------------------------------------------------------------

# Q20. NUMPY AND GPU DEVICES
#
# Create a PyTorch tensor on MPS if available.
#
# Try converting it directly into a NumPy array.
#
# Observe the result.
#
# Now move it back to CPU and convert it into NumPy.
#
# Question:
# Why does the CPU conversion matter here?
#
# Answer:

print("\nQuestion 20 answers :----------------------------")
if torch.backends.mps.is_available():
    pt_mps = torch.tensor([1.0, 2.0, 3.0], device="mps")
    try:
        np_failed = pt_mps.numpy()
    except TypeError as e:
        print(f"Direct conversion error caught: {e}")
    
    pt_moved_back = pt_mps.to("cpu").numpy()
    print(f"Successful array after moving to CPU: {pt_moved_back}")
else:
    print("MPS not available to test hardware barrier.")
# Reason: NumPy is strictly a CPU-bound framework and cannot directly access data sitting inside GPU VRAM. 
# The tensor must be brought back across the system bus to Main RAM before NumPy can parse it.


# ============================================================
# PHASE 5: IN-PLACE OPERATIONS
# ============================================================

# Q21. Create a tensor:
#
#       [1, 2, 3, 4]
#
# Perform the following:
#
# a. Add 5 using a normal addition operation.
# b. Print the original tensor.
# c. Print the resulting tensor.
#
# Question:
# Did the original tensor's values change?
#
# Answer:

t_orig = torch.tensor([1, 2, 3, 4])
t_res = t_orig + 5

print("\nQuestion 21 answers :----------------------------")
print(f"Original Tensor:  {t_orig}")
print(f"Resulting Tensor: {t_res}")
# Observation: No, the original tensor's values did not change because standard addition creates an out-of-place clone.


# ------------------------------------------------------------

# Q22. Create another tensor:
#
#       [1, 2, 3, 4]
#
# Use add_() to add 5 to every element.
#
# Print the tensor before and after the operation.
#
# Question:
# What is different compared to Q21?
#
# Answer:

t_inplace = torch.tensor([1, 2, 3, 4])
print("\nQuestion 22 answers :----------------------------")
print(f"Before add_(): {t_inplace}")
t_inplace.add_(5)
print(f"After add_():  {t_inplace}")
# Observation: The internal values of the original tensor variable were altered directly in its current memory slot.


# ------------------------------------------------------------

# Q23. Explore additional in-place operations.
#
# Create a tensor:
#
#       [2, 4, 6, 8]
#
# Perform:
#
# a. Multiply every element by 2 using mul_().
# b. Set every element to zero using zero_().
#
# Print the tensor after each operation.
#
# Answer:

t_ops = torch.tensor([2, 4, 6, 8])
print("\nQuestion 23 answers :----------------------------")
t_ops.mul_(2)
print(f"After mul_(2): {t_ops}")
t_ops.zero_()
print(f"After zero_(): {t_ops}")


# ------------------------------------------------------------

# Q24. UNDERSTANDING REFERENCES AND IN-PLACE MODIFICATION
#
# Create a tensor X.
#
# Assign another variable Y = X.
#
# Now modify X using an in-place operation.
#
# Print both X and Y.
#
# Question:
# Did Y also reflect the modification?
# Why?
#
# Answer:

X_ref = torch.tensor([10, 20, 30])
Y_ref = X_ref
X_ref.add_(10)

print("\nQuestion 24 answers :----------------------------")
print(f"Tensor X: {X_ref}")
print(f"Tensor Y: {Y_ref}")
# Observation: Yes, Y reflected the modification.
# Reason: Assigning Y = X does not copy data; it creates a second variable reference pointing to the exact same location in memory.


# ------------------------------------------------------------

# Q25. IN-PLACE OPERATIONS AND AUTOGRAD
#
# Read about PyTorch autograd.
#
# Consider the following mathematical operation:
#
#       y = x * x
#
# If x = 3:
#
# a. What is the value of y?
# b. What is dy/dx?
#
# Now create a PyTorch tensor with requires_grad=True.
#
# Perform the mathematical operation.
# Calculate the gradient using backward().
#
# Question:
# What gradient did PyTorch calculate?
#
# Next, read about why modifying tensors in-place can
# cause problems with autograd.
#
# Do not deliberately overwrite tensors needed for
# gradient calculation without understanding the error.
#
# Answer:

print("\nQuestion 25 answers :----------------------------")
# Analytical Answers: 
# a. y = 3 * 3 = 9
# b. dy/dx = 2 * x = 2 * 3 = 6

x_grad = torch.tensor([3.0], requires_grad=True)
y_grad = x_grad * x_grad
y_grad.backward()

print(f"PyTorch Calculated Gradient (x.grad): {x_grad.grad.item()}")
# Observation: PyTorch calculated exactly 6.0, matching the analytical calculus calculation.
# Autograd Warning: In-place changes overwrite the numbers that the tracking engine needs to evaluate chain rule graph steps during .backward().
