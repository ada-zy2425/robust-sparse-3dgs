#!/usr/bin/env python3
"""Compute geometric pre/post-BA deltas without modifying immutable pair artifacts."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bgs.core import WorkflowError,atomic_json,file_hash,read_json


def pose_matrix(image):
    import numpy as np
    pose=image.cam_from_world;pose=pose() if callable(pose) else pose
    return np.asarray(pose.matrix(),dtype=float)


def reprojection(reconstruction):
    import numpy as np
    errors=[];behind=0
    for point in reconstruction.points3D.values():
        xyz=np.asarray(point.xyz,dtype=float)
        for element in point.track.elements:
            image=reconstruction.images[element.image_id];camera=reconstruction.cameras[image.camera_id]
            cam=pose_matrix(image)[:,:3]@xyz+pose_matrix(image)[:,3]
            if cam[2]<=0:behind+=1;continue
            uv=np.asarray(camera.calibration_matrix(),dtype=float)@cam;uv=uv[:2]/uv[2]
            observed=np.asarray(image.points2D[element.point2D_idx].xy,dtype=float)
            errors.append(float(np.linalg.norm(uv-observed)))
    a=np.asarray(errors,dtype=float)
    if not len(a):raise WorkflowError('No valid observations for reprojection analysis')
    return {'valid_observations':len(a),'behind_camera_observations':behind,
        'mean_px':float(a.mean()),'rmse_px':float(np.sqrt(np.mean(a*a))),
        'median_px':float(np.median(a)),'p95_px':float(np.quantile(a,.95)),'max_px':float(a.max())}


def deltas(pre,post):
    import numpy as np
    common_images=sorted(set(pre.images)&set(post.images));centers=[];rotations=[];intrinsics=[]
    for image_id in common_images:
        a=pose_matrix(pre.images[image_id]);b=pose_matrix(post.images[image_id])
        ca=-a[:,:3].T@a[:,3];cb=-b[:,:3].T@b[:,3];centers.append(float(np.linalg.norm(ca-cb)))
        rel=b[:,:3]@a[:,:3].T
        rotations.append(float(np.degrees(np.arccos(np.clip((np.trace(rel)-1)/2,-1,1)))))
        pa=np.asarray(pre.cameras[pre.images[image_id].camera_id].params,dtype=float)
        pb=np.asarray(post.cameras[post.images[image_id].camera_id].params,dtype=float)
        intrinsics.extend(np.abs(pa-pb).tolist())
    common_points=sorted(set(pre.points3D)&set(post.points3D))
    movement=np.asarray([np.linalg.norm(np.asarray(pre.points3D[i].xyz)-np.asarray(post.points3D[i].xyz))
                         for i in common_points],dtype=float)
    def summary(values):
        a=np.asarray(values,dtype=float)
        return {'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),
                'max':float(a.max())}
    return {'common_images':len(common_images),'common_points3D':len(common_points),
        'camera_center_shift':summary(centers),'camera_rotation_degrees':summary(rotations),
        'camera_parameter_absolute_change':summary(intrinsics),'point3D_shift':summary(movement)}


def main():
    import pycolmap
    p=argparse.ArgumentParser();p.add_argument('--pair-root',required=True);args=p.parse_args()
    root=Path(args.pair_root).resolve();out=root/'geometry_analysis.json'
    if out.exists():raise WorkflowError('Refusing to overwrite BA geometry analysis')
    pair=read_json(root/'pair_receipt.json');pre=pycolmap.Reconstruction(str(root/'pre_ba'))
    post=pycolmap.Reconstruction(str(root/'post_ba'))
    atomic_json(out,{'schema':1,'pair_receipt_sha256':file_hash(root/'pair_receipt.json'),
        'pre_ba':reprojection(pre),'post_ba':reprojection(post),'pre_post_delta':deltas(pre,post),
        'ba_seconds':pair['ba_seconds'],'limit':'Joint camera/point BA effect; camera and point contributions are not separated.'})
    print(out)


if __name__=='__main__':
    try:main()
    except (WorkflowError,FileNotFoundError,KeyError,ValueError) as e:raise SystemExit(f'BLOCKED: {e}')
