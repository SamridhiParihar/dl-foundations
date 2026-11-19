import torch
import time
from torch import nn
from first_pytorch_training.model import MyNeuralNetwork
from first_pytorch_training.training import train_one_epoch
from first_pytorch_training.dataset import train_loader


def benchmark_device(device, epochs=5):
    device = torch.device(device)

    def synchronize():
        if device.type == "mps":
            torch.mps.synchronize()

    # Warm up the device with a disposable model and optimizer.
    torch.manual_seed(42)
    train_loader.generator.manual_seed(42)

    warmup_model = MyNeuralNetwork().to(device)
    warmup_loss_fn = nn.CrossEntropyLoss()
    warmup_optimizer = torch.optim.Adam(
        warmup_model.parameters(),
        lr=1e-3,
    )

    train_one_epoch(
        warmup_model,
        train_loader,
        warmup_loss_fn,
        warmup_optimizer,
        device,
    )
    synchronize()

    # Start the timed run with the same initialization and batch order
    # for each device.
    torch.manual_seed(42)
    train_loader.generator.manual_seed(42)

    model = MyNeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_times = []

    for epoch in range(epochs):
        synchronize()
        start_time = time.perf_counter()

        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        

        synchronize()
        elapsed = time.perf_counter() - start_time
        epoch_times.append(elapsed)

        print(f"{device} epoch {epoch + 1}/{epochs}: {elapsed:.2f} seconds")

    return epoch_times


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine")

    cpu_device = torch.device("cpu")
    mps_device = torch.device("mps")
    epochs = 5

    print("Benchmarking CPU...")
    cpu_times = benchmark_device(cpu_device, epochs=epochs)

    print("\nBenchmarking MPS...")
    mps_times = benchmark_device(mps_device, epochs=epochs)

    cpu_total = sum(cpu_times)
    mps_total = sum(mps_times)
    speedup = cpu_total / mps_total

    print("\nResults")
    for epoch, (cpu_time, mps_time) in enumerate(
        zip(cpu_times, mps_times), start=1
    ):
        print(
            f"Epoch {epoch}: "
            f"CPU {cpu_time:.2f}s | MPS {mps_time:.2f}s"
        )

    print(f"CPU total: {cpu_total:.2f}s")
    print(f"MPS total: {mps_total:.2f}s")
    print(f"MPS speedup (CPU total / MPS total): {speedup:.2f}x")


if __name__ == "__main__":
    main()