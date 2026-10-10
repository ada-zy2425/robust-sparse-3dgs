"""Training-camera-only frame alignment and pose diagnostics."""
import numpy as np
from .core import WorkflowError


def fit_similarity(source, target):
    x, y = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise WorkflowError("Similarity requires >=3 paired 3D camera centers")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise WorkflowError("Non-finite camera centers")
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x-mx, y-my
    singular = np.linalg.svd(xc, compute_uv=False)
    target_singular = np.linalg.svd(yc, compute_uv=False)
    if singular[0] < 1e-10 or singular[1]/singular[0] < 1e-5 or target_singular[1]/max(target_singular[0],1e-30) < 1e-5:
        raise WorkflowError("Degenerate/collinear camera centers; use an explicit calibrated frame or revise capture")
    u, d, vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(sign) @ vt
    scale = float((d*sign).sum() / (xc*xc).sum() * len(x))
    if not np.isfinite(scale) or scale <= 0:
        raise WorkflowError("Invalid similarity scale")
    translation = my - scale * rotation @ mx
    predicted = scale * (rotation @ x.T).T + translation
    residuals = np.linalg.norm(predicted-y, axis=1)
    return scale, rotation, translation, residuals


def reference_to_model(c2w, scale, rotation, translation):
    c2w = np.asarray(c2w, dtype=np.float64)
    result = np.eye(4)
    result[:3,:3] = rotation.T @ c2w[:3,:3]
    result[:3,3] = rotation.T @ (c2w[:3,3]-translation) / scale
    return result


def rotation_error_degrees(a, b):
    relative = np.asarray(a) @ np.asarray(b).T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))


def perturb_training_poses(poses, seed, rotation_degrees, translation_fraction):
    """Fixed total rotation angle about random axes; center displacement fixed
    in units of the training camera radius. First training camera is anchored."""
    poses = np.asarray(poses).copy()
    rng = np.random.default_rng(seed)
    centers = poses[:,:3,3]
    radius = float(np.max(np.linalg.norm(centers-centers.mean(0),axis=1)))
    angle = np.deg2rad(rotation_degrees)
    for i in range(1,len(poses)):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        a,b,c = axis
        cross = np.array([[0,-c,b],[c,0,-a],[-b,a,0]])
        r = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
        poses[i,:3,:3] = r @ poses[i,:3,:3]
        direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
        poses[i,:3,3] += direction * translation_fraction * radius
    return poses
