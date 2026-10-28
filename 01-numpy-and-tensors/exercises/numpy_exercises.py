"""NumPy and tensors — exercises.

Complete the numbered prompts in order. Write your reasoning and code in the
answer spaces. Predict shapes/results before running code where requested.
Do not erase the prompts; add your work beneath each one.
"""

import time
import numpy as np


# 1. Shape, axes, and dtype
# Create a float32 array representing 4 examples with 3 features each.
# Print its shape, ndim, size, and dtype. In a comment, state what each axis means.
# Answer:

arr = np.array([
               [1,2,3],
               [4,5,6],
               [7,8,9],
               [10,11,12] 
               ])

print(f"shape: {arr.shape}, ndim: {arr.ndim} , size: {arr.size}, dtype: {arr.dtype}")

# 2. Indexing, slicing, and reshape
# Make an array containing the integers 1 through 12, reshape it to (3, 4),
# then select the second column and the top-left 2-by-2 block.
# Print each selected result and its shape.
# Answer:
X = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
matrix = X.reshape(3,4)
print(matrix)


# 3. Views and copies
# Create base = np.arange(6). Make one slice view and one independent copy.
# Use np.shares_memory to check each relationship. Change one element in the
# view and one in the copy, then print base and explain what changed.
# Answer:

base = np.arange(6)

base_view = base[1:4]
base_copy = base.copy()

shares_with_view = np.shares_memory(base, base_view)
shares_with_copy = np.shares_memory(base, base_copy)

print(f"Does base share memory with base_view? {shares_with_view}")
print(f"Does base share memory with base_copy? {shares_with_copy}\n")

base_view[0] = 99   
base_copy[0] = -55  

print("Modified base array:")
print(base)



# 4. Boolean masks
# Given the sensor readings below, create a mask for values in [0, 2),
# select those readings, and then cap values above 2.0 in a separate copy.
readings = np.array([-0.4, 0.0, 0.7, 2.0, 2.6, 4.1])
# Answer:
mask = (readings >= 0) & (readings < 2)
readings_filtered = readings[mask]
print(readings_filtered)

capped_readings = readings.copy()
capped_readings[capped_readings>2] = 2

print(capped_readings)
# 5. Broadcasting predictions — do this section BEFORE running the expressions.
# For each pair below, predict whether addition succeeds and, if it does,
# write the result shape. Record all six predictions in comments first.
# (a) (2, 3) + (3,)       prediction: (2,3)
# (b) (2, 3) + (2, 1)     prediction: (2,3)
# (c) (4, 1, 3) + (3,)    prediction: (4,1,3)
# (d) (2, 3) + (2,)       prediction: incompatible dimensions
# (e) (3, 1) + (1, 4)     prediction: (3,4)
# (f) (2, 1, 3) + (4, 3)  prediction: (2,4,3)
# After writing all six predictions above, create arrays with these shapes,
# run each addition, and compare the actual outcomes with your predictions.
# Answer:


pairs = [
    ("a", (2, 3), (3,)),
    ("b", (2, 3), (2, 1)),
    ("c", (4, 1, 3), (3,)),
    ("d", (2, 3), (2,)),
    ("e", (3, 1), (1, 4)),
    ("f", (2, 1, 3), (4, 3))
]

for label, shape1, shape2 in pairs:
    # Initialize arrays with ones for the designated shapes
    arr1 = np.ones(shape1)
    arr2 = np.ones(shape2)
    
    try:
        result = arr1 + arr2
        print(f"({label}) {shape1} + {shape2} -> Succeeds! Result Shape: {result.shape}")
    except ValueError:
        print(f"({label}) {shape1} + {shape2} -> Fails! Incompatible dimensions.")



# 6. Loop versus vectorized timing
# Create at least 200,000 float32 values. Compute their squares once with a
# Python loop and once with a NumPy expression. Check the outputs agree.
# Time only the computations (not array setup), repeat each measurement several
# times with time.perf_counter, and compare the median times in milliseconds.
# Briefly explain why the timings may vary across machines or array sizes.
# Answer:

# 1. Create 200,000 float32 values
n_elements = 200000
base_array = np.random.rand(n_elements).astype(np.float32)

# Initialize timing storage
loop_times = []
vector_times = []
n_repeats = 5

# --- Python Loop Timing Run ---
for _ in range(n_repeats):
    # Time ONLY the computation, loop through as a native list/iterator
    data_list = base_array.tolist()
    
    start_time = time.perf_counter()
    loop_result = [x ** 2 for x in data_list]
    end_time = time.perf_counter()
    
    loop_times.append((end_time - start_time) * 1000) # Convert to ms

# --- NumPy Vectorized Timing Run ---
for _ in range(n_repeats):
    start_time = time.perf_counter()
    vector_result = base_array ** 2
    end_time = time.perf_counter()
    
    vector_times.append((end_time - start_time) * 1000) # Convert to ms

# 2. Check that the outputs agree
assert np.allclose(loop_result, vector_result), "Error: Results do not match!"

# 3. Calculate and display median execution times
median_loop = np.median(loop_times)
median_vector = np.median(vector_times)
speedup = median_loop / median_vector

print(f"Validation Check: PASS (Outputs agree)")
print(f"Median Loop Time:      {median_loop:.3f} ms")
print(f"Median NumPy Time:     {median_vector:.3f} ms")
print(f"NumPy Vectorization Speedup: {speedup:.1f}x faster")



# 7. Matrix multiplication shape reasoning
# Without running code, find the result shape of A @ B when A has shape (5, 3)
# and B has shape (3, 7). Then decide whether shapes (5, 3) and (4, 7) can be
# multiplied with @, and explain which dimensions determine your answer.
# Verify your answers with small NumPy arrays.
# Answer: first is possible and res shape is : (5,7) because inner dimensions are same
# second is not possible 

# Case 1: Valid Multiplication (5, 3) @ (3, 7)
A1 = np.ones((5, 3))
B1 = np.ones((3, 7))

result1 = A1 @ B1
print(f"Case 1: (5, 3) @ (3, 7) succeeds! Resulting shape: {result1.shape}")

# Case 2: Invalid Multiplication (5, 3) @ (4, 7)
A2 = np.ones((5, 3))
B2 = np.ones((4, 7))

print("\nCase 2: (5, 3) @ (4, 7) simulation:")
try:
    result2 = A2 @ B2
except ValueError as e:
    print(f"Caught expected error: {e}")





# 8. Dense layer: batch size four
# Construct X with shape (4, 3), W with shape (3, 2), and b with shape (2,).
# Compute Y = X @ W + b. Print the shapes of X, W, b, and Y; explain why the
# bias broadcasts over the four examples. Confirm Y has shape (4, 2).
# Answer:

X = np.random.randn(4, 3)
W = np.random.randn(3, 2)
b = np.random.randn(2,)

Y = X @ W + b

print("--- Tensor Shapes Summary ---")
print(f"Input Matrix (X) Shape : {X.shape}      (Batch Size: 4, Features: 3)")
print(f"Weight Matrix (W) Shape: {W.shape}      (In-Features: 3, Out-Features: 2)")
print(f"Bias Vector (b) Shape  : {b.shape}       (Out-Features: 2)")
print(f"Output Matrix (Y) Shape: {Y.shape}      (Batch Size: 4, Out-Features: 2)")

assert Y.shape == (4, 2), f"Expected shape (4, 2), but got {Y.shape}"
print("\nValidation Status: PASS (Y matches the target shape of (4, 2))")


# 9. NumPy and PyTorch connection
# In your own words, list two array ideas shared by NumPy and PyTorch, and one
# PyTorch training feature that NumPy arrays do not provide. (No PyTorch code
# is required.)
# Answer:
# 1. Both NumPy and PyTorch arrays/tensors have shapes and data types (dtype).
# 2. Both support indexing, slicing, broadcasting, and matrix multiplication.
# 3. PyTorch provides automatic differentiation (autograd), which calculates
#    gradients needed to update neural network weights during training.
#    NumPy does not provide built-in automatic differentiation.
