import statistics
import time
import torch 

available = torch.backends.mps.is_available()

size = 2048
a = torch.randn(size,size)
b = torch.randn(size,size)

def time_matmul(device, warmups=3, runs=10):
    x = a.to(device)
    y = b.to(device)

    def synchronize():
        if device == "mps":
            torch.mps.synchronize()

    for _ in range(warmups):
        result = x @ y
        synchronize()

    samples = []

    for _ in range(runs) :
        synchronize()
        start = time.perf_counter()

        result = x @ y

        synchronize()
        samples.append(time.perf_counter() - start)

    return statistics.median(samples) * 1000


cpu_ms = time_matmul("cpu")
print(f"CPU median: {cpu_ms:.2f} ms")


if available:
    mps_ms = time_matmul("mps")
    print(f"MPS median: {mps_ms:.2f} ms")
    print(f"CPU/MPS speed ratio: {cpu_ms / mps_ms:.2f}x")
else:
    print("MPS benchmark skipped: MPS is unavailable on this system.")