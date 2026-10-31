import time
import unittest
from camera_arm_mapping import BODY, interpolate, capture_sample


class MappingTests(unittest.TestCase):
    def samples(self):
        return [{'pixel':[x,y], 'raw':{m:2000+x+y for m in BODY}}
                for x,y in ((0,0),(100,0),(0,100),(100,100))]

    def test_interpolates_measured_local_plane(self):
        self.assertEqual(interpolate(self.samples(), [25,25]), {m:2050 for m in BODY})

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError): interpolate(self.samples(), [150,50])

    def test_requires_coverage(self):
        with self.assertRaises(ValueError): interpolate(self.samples()[:3], [25,25])
        samples = self.samples()
        for s in samples: s['pixel'][1] = 0
        with self.assertRaises(ValueError): interpolate(samples, [25,0])

    def test_does_not_blend_ik_branches(self):
        samples = self.samples()
        for i,s in enumerate(samples): s['raw']['wrist_flex'] += i*600
        with self.assertRaises(ValueError): interpolate(samples, [25,25])

    def test_capture_rejects_stale_or_moving_arm(self):
        status={'phase':'holding','last_sample':time.time()-10}
        health={'live':True,'frame_age_seconds':0,'camera_generation':1}
        with self.assertRaises(ValueError): capture_sample([10,20],status,health)
        status.update(last_sample=time.time(),readings={m:{'Status':0,'Moving':1,'Torque_Enable':1} for m in BODY})
        with self.assertRaises(ValueError): capture_sample([10,20],status,health)


if __name__ == '__main__': unittest.main()
