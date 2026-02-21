from __future__ import annotations

from collections.abc import Iterable

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def configure_device(device: str) -> tuple:
    """Parse a device string and return ``(torch_device, use_gpu)``.

    Returns:
        A ``(torch_device, use_gpu)`` tuple where *torch_device* is a
        ``torch.device`` (or ``None`` for CPU-only) and *use_gpu* is a bool.
    """
    device = str(device).strip()
    device_key = device.lower()
    if device_key == "gpu":
        device = "cuda"
        device_key = "cuda"
    if device_key == "cpu":
        return None, False
    if torch is None:
        raise RuntimeError(
            f"Torch is required for device '{device}', but it is not installed."
        )
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    use_gpu = torch_device.type != "cpu"
    return torch_device, use_gpu


def iter_batches(items: Iterable, batch_size: int):
    """Yield fixed-size batches from *items*."""
    if batch_size <= 0:
        batch_size = 1
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def torch_generator_for_index(
    torch_device: "torch.device",
    seed: int | None,
    base_rng: np.random.Generator,
    index: int,
) -> "torch.Generator":
    """Create a seeded ``torch.Generator`` for the given sample *index*."""
    gen = torch.Generator(device=torch_device)
    if seed is not None:
        seed_seq = np.random.SeedSequence([seed, index])
        seed_val = int(seed_seq.generate_state(1, dtype=np.uint64)[0])
        seed_val = seed_val & 0x7FFFFFFFFFFFFFFF
    else:
        seed_val = int(base_rng.integers(0, np.iinfo(np.int64).max))
    gen.manual_seed(seed_val)
    return gen
