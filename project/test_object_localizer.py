import unittest
import cv2
import numpy as np
from object_localizer import detect_objects, stable_target


class LocalizerTests(unittest.TestCase):
    def scene(self):
        return np.full((400, 600, 3), 180, np.uint8), np.float32([[10,10],[590,10],[590,390],[10,390]])

    def test_locates_object_not_grid_lines(self):
        image, corners = self.scene()
        cv2.line(image, (300,10), (300,390), (30,30,30), 2)
        cv2.rectangle(image, (100,100), (150,200), (20,20,20), -1)
        result = detect_objects(image, corners)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0]['center_px'], [125,150], atol=1)

    def test_edge_connected_dark_shape_is_flagged_not_verified(self):
        image, corners = self.scene()
        cv2.rectangle(image, (400,0), (450,150), (20,20,20), -1)
        result = detect_objects(image, corners)
        self.assertTrue(result[0]['near_board_boundary'])
        self.assertFalse(result[0]['grasp_verified'])

    def test_empty_scene(self):
        image, corners = self.scene()
        self.assertEqual(detect_objects(image, corners), [])

    def test_stability_requires_one_target_each_frame(self):
        obj = {'center_px': [100,100]}
        self.assertIsNone(stable_target([[obj],[],[obj]]))
        self.assertIsNone(stable_target([[obj],[obj,obj],[obj]]))
        self.assertEqual(stable_target([[obj],[obj],[obj]]), [100,100])

    def test_moving_target_is_rejected(self):
        self.assertIsNone(stable_target([[{'center_px':[x,100]}] for x in (100,130,160)]))


if __name__ == '__main__': unittest.main()
