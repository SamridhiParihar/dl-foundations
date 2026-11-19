import numpy as np

def conv2d(image, kernel, padding, stride):
    w_img = image.shape[1]
    h_img = image.shape[0]
    w_krnl = kernel.shape[1]
    h_krnl = kernel.shape[0]
    p = padding
    s = stride

    out_w = (w_img - w_krnl + 2 * p) // s + 1
    out_h = (h_img - h_krnl + 2 * p) // s + 1

    res = [[0 for _ in range(out_w)] for _ in range(out_h)]

    for out_i in range(out_h):
        for out_j in range(out_w):
            i = out_i * s - p
            j = out_j * s - p

            total = 0
            for ki in range(h_krnl):
                for kj in range(w_krnl):
                    val = 0 if i + ki < 0 or j + kj < 0 or i + ki >= h_img or j + kj >= w_img else image[i + ki][j + kj]
                    total += val * kernel[ki][kj]

            res[out_i][out_j] = total

    return res
