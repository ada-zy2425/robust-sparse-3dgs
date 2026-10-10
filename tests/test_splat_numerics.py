import unittest

import numpy as np
import torch

from bgs.core import WorkflowError
from bgs.splat_numerics import first_nonfinite_splat,guarded_log_scales,stable_exact_deduplicate


def distances(points):
    return torch.sort(torch.cdist(points,points),dim=1).values[:,:4]


class SplatNumericsTests(unittest.TestCase):
    def test_normal_cloud_is_unchanged(self):
        points=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]],dtype=torch.float32)
        d=distances(points);got,receipt=guarded_log_scales(points,d,2.,1.)
        expected=torch.log(torch.sqrt((d[:,1:]**2).mean(-1))).unsqueeze(-1).repeat(1,3)
        self.assertTrue(torch.equal(got,expected));self.assertEqual(receipt['clamped_point_count'],0)

    def test_degenerate_cluster_gets_positive_dtype_scene_floor(self):
        points=torch.tensor([[0.,0.,0.]]*4+[[1.,0.,0.],[0.,1.,0.]],dtype=torch.float32)
        got,receipt=guarded_log_scales(points,distances(points),2.,1.)
        self.assertTrue(torch.isfinite(got).all());self.assertTrue((torch.exp(got)>0).all())
        self.assertEqual(receipt['zero_raw_scale_points'],4);self.assertEqual(receipt['clamped_point_count'],4)
        self.assertFalse(receipt['point_count_changed']);self.assertFalse(receipt['colors_changed'])

    def test_floor_scales_with_scene_units(self):
        points=torch.tensor([[0.,0.,0.]]*4+[[1.,0.,0.],[0.,1.,0.]],dtype=torch.float32);d=distances(points)
        a,ra=guarded_log_scales(points,d,2.,.5);b,rb=guarded_log_scales(points*10,d*10,20.,.5)
        self.assertAlmostEqual(rb['positive_scale_floor']/ra['positive_scale_floor'],10.)
        self.assertTrue(torch.allclose(b,a+np.log(10.),rtol=0,atol=2e-6))

    def test_stable_dedup_aggregates_colors_and_preserves_extent(self):
        points=np.array([[0.,0.,0.],[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float32)
        colors=np.array([[0,0,0],[100,50,0],[1,2,3],[4,5,6],[7,8,9]],np.uint8)
        errors=np.arange(5,dtype=np.float32);indices={'image':np.array([0,1,2],np.int32)}
        p,c,e,mapping,_,receipt=stable_exact_deduplicate(points,colors,errors,indices)
        self.assertEqual(len(p),4);self.assertTrue(np.array_equal(c[0],[50.,25.,0.]))
        self.assertEqual(e[0],.5);self.assertTrue(np.array_equal(mapping['image'],[0,1]))
        self.assertTrue(receipt['spatial_coverage_preserved']);self.assertEqual(receipt['removed_points'],1)

    def test_invalid_inputs_are_rejected(self):
        good=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=torch.float32)
        with self.assertRaises(WorkflowError):guarded_log_scales(good,distances(good),0.,1.)
        bad=good.clone();bad[0,0]=float('nan')
        with self.assertRaises(WorkflowError):guarded_log_scales(bad,distances(good),1.,1.)
        with self.assertRaises(WorkflowError):stable_exact_deduplicate(np.zeros((3,3),np.float32),np.zeros((3,3)))

    def test_immediate_splat_check_reports_exact_first_scale_location(self):
        splats={name:torch.ones((4,3),dtype=torch.float32) for name in ("means","scales","sh0","shN")}
        splats["quats"]=torch.ones((4,4),dtype=torch.float32);splats["opacities"]=torch.ones(4)
        splats["scales"][2,1]=float("-inf")
        self.assertEqual(first_nonfinite_splat(splats),
                         {"field":"scales","index":[2,1],"kind":"nonfinite","value":"-inf"})


if __name__=='__main__':unittest.main()
