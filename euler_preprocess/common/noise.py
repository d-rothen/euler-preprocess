from __future__ import annotations

import math

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_noise(
    height: int, width: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"Perlin scale must be > 0, got {scale}")
    grid_w = int(math.ceil(width / scale)) + 1
    grid_h = int(math.ceil(height / scale)) + 1

    angles = rng.uniform(0, 2 * math.pi, size=(grid_h, grid_w))
    grads = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    x = np.arange(width, dtype=np.float32) / scale
    y = np.arange(height, dtype=np.float32) / scale
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    xf = x - x0
    yf = y - y0

    x0g = x0[None, :]
    x1g = x1[None, :]
    y0g = y0[:, None]
    y1g = y1[:, None]
    xf_g = xf[None, :]
    yf_g = yf[:, None]
    u = fade(xf_g)
    v = fade(yf_g)

    g00 = grads[y0g, x0g]
    g10 = grads[y0g, x1g]
    g01 = grads[y1g, x0g]
    g11 = grads[y1g, x1g]

    d00 = g00[..., 0] * xf_g + g00[..., 1] * yf_g
    d10 = g10[..., 0] * (xf_g - 1) + g10[..., 1] * yf_g
    d01 = g01[..., 0] * xf_g + g01[..., 1] * (yf_g - 1)
    d11 = g11[..., 0] * (xf_g - 1) + g11[..., 1] * (yf_g - 1)

    nx0 = lerp(d00, d10, u)
    nx1 = lerp(d01, d11, u)
    nxy = lerp(nx0, nx1, v)

    min_val = float(nxy.min())
    max_val = float(nxy.max())
    if math.isclose(max_val, min_val):
        return np.zeros_like(nxy, dtype=np.float32)
    nxy = (nxy - min_val) / (max_val - min_val)
    return nxy.astype(np.float32)


def perlin_fbm(
    height: int, width: int, scales: list[int], rng: np.random.Generator
) -> np.ndarray:
    total = np.zeros((height, width), dtype=np.float32)
    weight_sum = 0.0
    for scale in scales:
        noise = perlin_noise(height, width, scale, rng)
        weight = math.log2(scale) ** 2 if scale > 1 else 1.0
        total += noise * weight
        weight_sum += weight
    if weight_sum <= 0:
        return np.zeros((height, width), dtype=np.float32)
    total /= weight_sum
    return np.clip(total, 0.0, 1.0)


def perlin_noise_torch(
    height: int,
    width: int,
    scale: float,
    rng: "torch.Generator",
    device: "torch.device",
    batch_size: int = 1,
):
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"Perlin scale must be > 0, got {scale}")
    grid_w = int(math.ceil(width / scale)) + 1
    grid_h = int(math.ceil(height / scale)) + 1

    angles = torch.rand(
        (batch_size, grid_h, grid_w), device=device, generator=rng, dtype=torch.float32
    )
    angles = angles * (2.0 * math.pi)
    grads = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    x = torch.arange(width, device=device, dtype=torch.float32) / scale
    y = torch.arange(height, device=device, dtype=torch.float32) / scale
    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    xf = x - x0
    yf = y - y0

    x0g = x0[None, :]
    x1g = x1[None, :]
    y0g = y0[:, None]
    y1g = y1[:, None]
    xf_g = xf[None, :]
    yf_g = yf[:, None]
    u = fade(xf_g)
    v = fade(yf_g)

    g00 = grads[:, y0g, x0g]
    g10 = grads[:, y0g, x1g]
    g01 = grads[:, y1g, x0g]
    g11 = grads[:, y1g, x1g]

    d00 = g00[..., 0] * xf_g + g00[..., 1] * yf_g
    d10 = g10[..., 0] * (xf_g - 1) + g10[..., 1] * yf_g
    d01 = g01[..., 0] * xf_g + g01[..., 1] * (yf_g - 1)
    d11 = g11[..., 0] * (xf_g - 1) + g11[..., 1] * (yf_g - 1)

    nx0 = lerp(d00, d10, u)
    nx1 = lerp(d01, d11, u)
    nxy = lerp(nx0, nx1, v)

    min_val = nxy.amin(dim=(1, 2), keepdim=True)
    max_val = nxy.amax(dim=(1, 2), keepdim=True)
    denom = max_val - min_val
    nxy = (nxy - min_val) / denom.clamp_min(1e-8)
    nxy = torch.where(denom <= 1e-8, torch.zeros_like(nxy), nxy)

    if batch_size == 1:
        return nxy[0]
    return nxy


def perlin_fbm_torch(
    height: int,
    width: int,
    scales: list[int],
    rng: "torch.Generator",
    device: "torch.device",
    batch_size: int = 1,
):
    total = torch.zeros((batch_size, height, width), device=device, dtype=torch.float32)
    weight_sum = 0.0
    for scale in scales:
        noise = perlin_noise_torch(height, width, scale, rng, device, batch_size)
        if noise.ndim == 2:
            noise = noise.unsqueeze(0)
        weight = math.log2(scale) ** 2 if scale > 1 else 1.0
        total += noise * weight
        weight_sum += weight
    if weight_sum <= 0:
        total = torch.zeros(
            (batch_size, height, width), device=device, dtype=torch.float32
        )
    else:
        total = total / weight_sum
    total = torch.clamp(total, 0.0, 1.0)
    if batch_size == 1:
        return total[0]
    return total
