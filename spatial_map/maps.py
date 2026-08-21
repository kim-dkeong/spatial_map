import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix


def spatial_map(sig, F, valid, z_range=(-3.2, 8.8), bin_size=0.4, fs=15.0,
                col="z_rel"):
    
    n_bins = int(round((z_range[1] - z_range[0]) / bin_size))
    edges = np.linspace(z_range[0], z_range[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if len(F) != sig.shape[1]:
        raise ValueError(
            f"길이가 안 맞습니다: F 는 {len(F)} 행, sig 는 {sig.shape[1]} 열.\n"
            f"  build_frame_table(..., n_img_frames=?) 에 AI1 pulse 개수가 아니라\n"
            f"  **dF/F 의 frame 수**(sig.shape[1]) 를 넣으세요.\n"
            f"  AI1 pulse 는 보통 dF/F frame 보다 1~2 개 많습니다 "
            f"(녹화 종료 시점에 잘린 마지막 pulse 등).")
    if len(valid) != sig.shape[1]:
        raise ValueError(f"valid 길이 {len(valid)} != sig frame 수 {sig.shape[1]}")

    pos = F[col].to_numpy()
    tr = F["trial"].to_numpy()
    m = valid & np.isfinite(pos) & (tr >= 0) & (pos >= edges[0]) & (pos < edges[-1])

    ids = np.unique(tr[m])
    lut = {t: i for i, t in enumerate(ids)}
    bi = np.clip(np.digitize(pos[m], edges) - 1, 0, n_bins - 1)
    li = np.array([lut[t] for t in tr[m]], int)
    flat = li * n_bins + bi

    occ = np.bincount(flat, minlength=len(ids) * n_bins).astype(float)
    B = csr_matrix((np.ones(len(flat)), (np.arange(len(flat)), flat)),
                   shape=(len(flat), len(ids) * n_bins))
    S = np.asarray(sig[:, m] @ B)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = S / occ[None, :]
    M[:, occ == 0] = np.nan
    maps = M.reshape(sig.shape[0], len(ids), n_bins)
    occ_s = (occ / fs).reshape(len(ids), n_bins)
    return maps, occ_s, edges, centers, ids


def zscore(sig, mask=None, robust=True):
  
    ref = sig if mask is None else sig[:, mask]
    if robust:
        c = np.nanmedian(ref, 1, keepdims=True)
        s = 1.4826 * np.nanmedian(np.abs(ref - c), 1, keepdims=True)
    else:
        c, s = np.nanmean(ref, 1, keepdims=True), np.nanstd(ref, 1, keepdims=True)
    s[s == 0] = np.nan
    return (sig - c) / s


def smooth(maps, sigma_bins=1.2):
    """NaN-aware: g(x·mask)/g(mask)"""
    if sigma_bins <= 0:
        return maps
    W = np.isfinite(maps).astype(float)
    num = gaussian_filter1d(np.nan_to_num(maps), sigma_bins, axis=-1, mode="nearest")
    den = gaussian_filter1d(W, sigma_bins, axis=-1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < 1e-6] = np.nan
    return out
