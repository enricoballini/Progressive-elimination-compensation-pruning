import jax.numpy as jnp

from . import importance as imp
from . import pruning


def prune_with_scores(
    params, existing, scores, ratios, joint=None, joint_samples=None, gauss_newton_iterations=5
):
    pruned = []
    for ratio in ratios:
        masks = pruning.global_masks(scores, existing, ratio)
        params_pruned = pruning.apply_masks(params, masks)
        if joint is not None:
            params_pruned = joint.gauss_newton(
                params_pruned, params, joint_samples, num_iterations=gauss_newton_iterations
            )
        pruned.append((masks, params_pruned))
    return pruned


def elimination_compensation(
    params, existing, second_moments, samples, ratios,
    joint=None, joint_samples=None, gauss_newton_iterations=5,
    regularisation=imp.COMPLEX_REGULARISATION,
):
    moments = second_moments(params, samples)
    importances, compensations = imp.ec_importance(params, moments, regularisation)
    pruned = []
    for ratio in ratios:
        masks = pruning.global_masks(importances, existing, ratio)
        params_pruned = pruning.apply_masks_and_compensation(params, masks, existing, compensations)
        if joint is not None:
            params_pruned = joint.gauss_newton(
                params_pruned, params, joint_samples, num_iterations=gauss_newton_iterations
            )
        pruned.append((masks, params_pruned))
    return pruned


def progressive_elimination_compensation(
    params, existing, second_moments, samples, ratios,
    joint=None, joint_samples=None, steps_between_ratios=2, gauss_newton_iterations=2,
    regularisation=imp.COMPLEX_REGULARISATION, ranking=None, with_residual_term=False,
):
    current = params
    masks = existing
    pruned = []
    previous_ratio = 0.0
    for ratio in ratios:
        for step in range(1, steps_between_ratios + 1):
            fraction = step / steps_between_ratios
            kept = (1.0 - previous_ratio) * ((1.0 - ratio) / (1.0 - previous_ratio)) ** fraction

            moments = second_moments(current, samples)
            importances, compensations = imp.ec_importance(current, moments, regularisation)
            if ranking is not None:
                importances = ranking(current, moments)
            if with_residual_term:
                gradient = joint.discrepancy_gradient(current, params, joint_samples)
                importances = [
                    importance
                    + jnp.real(imp.bias_like_weight(g["b"], p["W"]) * shift)
                    - jnp.real(g["W"] * p["W"])
                    if importance is not None else None
                    for importance, shift, p, g in zip(importances, compensations, current, gradient)
                ]
            importances = [
                jnp.where(mask > 0, importance, -jnp.inf) if importance is not None else None
                for importance, mask in zip(importances, masks)
            ]
            masks = pruning.global_masks(importances, existing, 1.0 - kept)
            current = pruning.apply_masks_and_compensation(current, masks, existing, compensations)
            if joint is not None:
                current = joint.gauss_newton(
                    current, params, joint_samples, num_iterations=gauss_newton_iterations
                )
        previous_ratio = ratio
        pruned.append((masks, current))
    return pruned
