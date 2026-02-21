from __future__ import annotations

import numpy as np


def sample_value(spec, rng: np.random.Generator):
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, list):
        return [sample_value(item, rng) for item in spec]
    if isinstance(spec, dict):
        if "dist" not in spec:
            if "value" in spec:
                return sample_value(spec["value"], rng)
            return {k: sample_value(v, rng) for k, v in spec.items()}
        dist = spec["dist"]
        if dist == "constant":
            return sample_value(spec.get("value", 0.0), rng)
        if dist == "uniform":
            return float(rng.uniform(spec["min"], spec["max"]))
        if dist == "normal":
            val = float(rng.normal(spec["mean"], spec["std"]))
            if "min" in spec or "max" in spec:
                val = float(
                    np.clip(val, spec.get("min", -np.inf), spec.get("max", np.inf))
                )
            return val
        if dist == "lognormal":
            val = float(rng.lognormal(spec["mean"], spec["sigma"]))
            if "min" in spec or "max" in spec:
                val = float(
                    np.clip(val, spec.get("min", -np.inf), spec.get("max", np.inf))
                )
            return val
        if dist == "choice":
            values = spec["values"]
            weights = spec.get("weights")
            idx = int(rng.choice(len(values), p=weights))
            return sample_value(values[idx], rng)
        raise ValueError(f"Unsupported dist: {dist}")
    if isinstance(spec, str):
        return spec
    raise ValueError(f"Unsupported spec type: {type(spec)}")


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def format_value(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"
