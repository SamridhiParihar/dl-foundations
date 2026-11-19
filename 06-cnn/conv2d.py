import torch 
import torch.nn.functional as F
from manual_conv2d import conv2d
import numpy as np

image = [
    [1,2,3,4,5],
    [6,7,8,9,10],
    [11,12,13,14,15],
    [16,17,18,19,20],
    [21,22,23,24,25]
]

kernel = [
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
]

input_tensor = torch.tensor(
    image,
    dtype=torch.float32
).reshape(1,1,5,5)

kernel_tensor = torch.tensor(
    kernel,
    dtype=torch.float32
).reshape(1,1,3,3)

torch_result = F.conv2d(
    input_tensor,
    kernel_tensor,
    stride=1,
    padding=0
)

manual_result = conv2d(
    np.array(image),
    np.array(kernel),
    stride=1,
    padding=0
)

is_same = torch.allclose(torch_result.squeeze(), torch.tensor(manual_result, dtype=torch.float32), atol=1e-5)
print("Results match:", is_same)

