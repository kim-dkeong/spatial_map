"""Cue selectivity index (SI) = (mean_L - mean_R) / (mean_L + mean_R) for dF/F spatial maps.

maps: (n_neurons, n_trials, n_bins)   e.g. smooth_maps
SI > 0: Left-preferring, SI < 0: Right-preferring
"""

import numpy as np


def select_trials(groups, use=("L_o", "R_o")):
    """groups: per-trial labels like 'L_o', 'R_o', 'R_x', ...
    returns keep (bool, n_trials) and is_left (bool, keep.sum())"""
    groups = np.asarray(groups)
    keep = np.isin(groups, use)
    is_left = groups[keep] == use[0]
    return keep, is_left


def calculate_selectivity(maps, is_left, eps=1e-6):
    """SI per neuron and position bin.
    maps: (n_neurons, n_trials, n_bins), is_left: (n_trials,) bool
    returns (n_neurons, n_bins)"""
    mean_l = np.nanmean(maps[:, is_left, :], axis=-2)
    mean_r = np.nanmean(maps[:, ~is_left, :], axis=-2)

    # dF/F can be negative: shift each neuron by a common offset so values are >= 0
    offset = np.minimum(np.nanmin(np.minimum(mean_l, mean_r), axis=-1, keepdims=True), 0)

    l, r = mean_l - offset, mean_r - offset

    denom = l + r
    return (l - r) / (denom + eps)


def window_selectivity_perm(maps, is_left, centers_cm, window=(0, 30), n_perm=1000, seed=0, eps=1e-6):
    """Window-averaged SI per neuron + permutation test (shuffling L/R labels).

    maps: (n_neurons, n_trials, n_bins), is_left: (n_trials,) bool (L_o/R_o trials only)
    returns si (n_neurons,), p (n_neurons,), null (n_neurons, n_perm)
    p is two-sided: (1 + #{|null| >= |si|}) / (n_perm + 1)
    """
    rng = np.random.default_rng(seed)
    in_win = (centers_cm >= window[0]) & (centers_cm < window[1])
    v = np.nanmean(maps[:, :, in_win], axis=-1)                    # (n_neurons, n_trials)

    valid = np.isfinite(v)
    v = v - np.minimum(np.nanmin(v, axis=1, keepdims=True), 0)    # dF/F offset (label-independent)
    v = np.where(valid, v, 0.0)

    def si_from_masks(masks):                                      # masks: (k, n_trials) float, 1 = Left
        nl = valid @ masks.T
        nr = valid.sum(1, keepdims=True) - nl
        mean_l = (v @ masks.T) / np.maximum(nl, 1)
        mean_r = (v.sum(1, keepdims=True) - v @ masks.T) / np.maximum(nr, 1)
        denom = mean_l + mean_r
        return np.where(denom > eps, (mean_l - mean_r) / np.maximum(denom, eps), np.nan)

    n_tr, n_left = is_left.size, int(is_left.sum())
    masks = np.zeros((n_perm, n_tr))
    for i in range(n_perm):
        masks[i, rng.choice(n_tr, n_left, replace=False)] = 1.0    # L/R label shuffling

    si = si_from_masks(is_left.astype(float)[None])[:, 0]
    null = si_from_masks(masks)                                    # (n_neurons, n_perm)
    p = (1 + (np.abs(null) >= np.abs(si)[:, None]).sum(1)) / (n_perm + 1)
    return si, p, null