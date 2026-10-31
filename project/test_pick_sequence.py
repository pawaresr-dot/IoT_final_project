import copy
import json
import unittest
from pathlib import Path
from pick_sequence import BODY, REQUIRED, build_sequence, readiness


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(Path(__file__).with_name('pick_place_poses.json').read_text())

    def test_order_and_gripper_actions(self):
        steps = build_sequence(self.document, 'bottom_left')
        self.assertEqual(len(steps), 13)
        self.assertEqual([s['kind'] for s in steps].count('gripper'), 3)
        self.assertEqual(steps[6]['target_raw'], {'gripper': 2242})
        self.assertEqual(steps[0]['target_raw'], steps[-1]['target_raw'])
        for step in steps:
            self.assertEqual(set(step['target_raw']), set(BODY) if step['kind'] == 'body' else {'gripper'})

    def test_other_cells_not_invented(self):
        for cell in ('top_left', 'top_right', 'bottom_right', 'invalid'):
            self.assertFalse(readiness(self.document, cell, None)['ready'])
            self.assertEqual(readiness(self.document, cell, None)['steps'], [])

    def test_old_observations_not_marked_motion_ready(self):
        result = readiness(self.document, 'bottom_left', None)
        self.assertFalse(result['ready'])
        self.assertFalse(result['object_pickup_verified'])
        self.assertGreaterEqual(len(result['reasons']), 2)

    def test_no_mutation_of_teaching_data(self):
        original = copy.deepcopy(self.document)
        build_sequence(self.document, 'bottom_left')
        self.assertEqual(self.document, original)

    def test_healthy_fixture_and_temperature_fault(self):
        poses = self.document['poses']['bottom_left']
        for stage in REQUIRED:
            poses[stage]['validation']['ready_for_motion'] = True
        motors = {}
        for m in (*BODY, 'gripper'):
            motors[m] = dict(Present_Position=poses['retreat']['raw_position'][m],
                Torque_Enable=0, Status=0, Operating_Mode=0, Present_Temperature=30,
                Present_Voltage=120, Min_Voltage_Limit=60, Max_Voltage_Limit=140,
                Min_Position_Limit=0, Max_Position_Limit=4095)
        snap = {'calibrated': True, 'motors': motors}
        self.assertTrue(readiness(self.document, 'bottom_left', snap)['ready'])
        motors['gripper']['Present_Temperature'] = 58
        self.assertFalse(readiness(self.document, 'bottom_left', snap)['ready'])


if __name__ == '__main__': unittest.main()
