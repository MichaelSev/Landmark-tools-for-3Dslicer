import numpy as np
from scipy.optimize import minimize

def align_shapes_3d(target_points, source_points, allow_reflection=False):
    """
    Align target_points -> source_points with a similarity transform (scale, rotation, translation).

    Parameters
    ----------
    target_points : (N,3) or flat array
        The points you want to move.
    source_points : (N,3) or flat array
        The reference points you want to align to.
    allow_reflection : bool
        If False (default), reflections are prevented. Set True if you want to allow mirrored solutions.

    Returns
    -------
    aligned_target : (N,3)
        The target_points after alignment into source space.
    transform_params : dict
        Contains rotation matrix, scale, translation, centroids, and RMS error.
    """
    def to_nx3(vec):
        a = np.asarray(vec, dtype=float)
        if a.ndim == 1:
            if a.size % 3 != 0:
                raise ValueError("Point arrays must be length multiple of 3.")
            return np.column_stack([a[0::3], a[1::3], a[2::3]])
        elif a.ndim == 2 and a.shape[1] == 3:
            return a
        else:
            raise ValueError("Points must be (N,3) or 1D flattened with length multiple of 3.")

    X = to_nx3(target_points)
    Y = to_nx3(source_points)

    n = X.shape[0]
    centroid_X = X.mean(axis=0)
    centroid_Y = Y.mean(axis=0)

    Xc = X - centroid_X
    Yc = Y - centroid_Y

    # covariance
    cov = (Yc.T @ Xc) / n

    # SVD
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        if not allow_reflection:
            S[2, 2] = -1.0

    R = U @ S @ Vt
    scale = np.sum(D * np.diag(S)) / (np.sum(Xc ** 2) / n)
    t = centroid_Y - scale * (centroid_X @ R.T)

    aligned_target = scale * (X @ R.T) + t

    rms = np.sqrt(np.mean(np.sum((aligned_target - Y) ** 2, axis=1)))

    transform_params = {
        "rotation_matrix": R,
        "scale": float(scale),
        "translation": tuple(t.tolist()),
        "centroid_target": centroid_X,
        "centroid_source": centroid_Y,
        "rms_error": float(rms),
    }

    return aligned_target, transform_params
