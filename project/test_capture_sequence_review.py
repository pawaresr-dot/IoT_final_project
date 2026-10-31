import unittest
from review_training_capture import jaw_motion_summary

def rows(values): return [{'action_after_raw':[0]*5+[v]} for v in values]

class SequenceReviewTests(unittest.TestCase):
    def test_open_only_is_not_a_complete_pick(self):
        result=jaw_motion_summary(rows([2088,2092,2110,2150,2373]))
        self.assertFalse(result['gripper_closing_observed'])
        self.assertTrue(result['gripper_opening_observed'])
        self.assertFalse(result['full_pick_sequence_observed'])

    def test_slow_close_is_detected_cumulatively(self):
        result=jaw_motion_summary(rows(range(2373,2087,-4)))
        self.assertTrue(result['gripper_closing_observed'])

    def test_both_jaw_directions_do_not_prove_lift(self):
        result=jaw_motion_summary(rows([2373,2100,2373]))
        self.assertTrue(result['gripper_closing_observed'])
        self.assertTrue(result['gripper_opening_observed'])
        self.assertFalse(result['full_pick_sequence_observed'])

    def test_small_jitter_is_not_a_grasp(self):
        result=jaw_motion_summary(rows([2200,2201,2199,2200]))
        self.assertFalse(result['gripper_closing_observed'])
        self.assertFalse(result['gripper_opening_observed'])

if __name__=='__main__': unittest.main()
