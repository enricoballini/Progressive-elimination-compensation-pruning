import numpy as np
import jax.numpy as jnp


def global_masks(importances, existing, ratio):
    flat_importance = np.concatenate(
        [np.asarray(imp)[np.asarray(ex) > 0] for imp, ex in zip(importances, existing) if imp is not None]
    )
    num_pruned = int(round(ratio * flat_importance.size))
    keep_flat = np.ones(flat_importance.size, dtype=np.float32)
    keep_flat[np.argsort(flat_importance, kind="stable")[:num_pruned]] = 0.0

    masks, start = [], 0
    for imp, ex in zip(importances, existing):
        if imp is None:
            masks.append(None)
            continue
        ex = np.asarray(ex) > 0
        mask = np.zeros(ex.shape, dtype=np.float32)
        mask[ex] = keep_flat[start : start + ex.sum()]
        start += ex.sum()
        masks.append(jnp.asarray(mask))
    return masks


def pruning_ratio(masks, existing):
    kept = sum(float(m.sum()) for m in masks if m is not None)
    total = sum(float(e.sum()) for e in existing if e is not None)
    return 1.0 - kept / total


def removed_weights(params, masks, existing):
    return [
        p["W"] * (ex - mask) if p else None
        for p, mask, ex in zip(params, masks, existing)
    ]


def apply_masks(params, masks):
    return [
        {"W": p["W"] * mask, "b": p["b"]} if p else {}
        for p, mask in zip(params, masks)
    ]


def apply_masks_and_compensation(params, masks, existing, compensations):
    new_params = []
    for p, mask, ex, comp in zip(params, masks, existing, compensations):
        if not p:
            new_params.append({})
            continue
        removed = ex - mask
        other_axes = tuple(range(1, p["W"].ndim))
        bias_shift = (removed * comp).sum(axis=other_axes)
        new_params.append({"W": p["W"] * mask, "b": p["b"] + bias_shift})
    return new_params
