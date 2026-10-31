import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import leader_teaching as t
from test_hold_profile import FakeBus


def follower():
    return {'phase': 'holding', 'manual_mode': True, 'manual_move_complete': True,
            'error': None, 'seconds': 1,
            'manual_target_raw': dict.fromkeys(t.BODY, 2000),
            'target_raw': dict.fromkeys(t.BODY, 2000),
            'readings': {m: {'Present_Position': 2000, 'Present_Temperature': 30,
                             'Status': 0, 'Torque_Enable': 1}
                         for m in (*t.BODY, 'gripper')}}


class LeaderTeachingTests(unittest.TestCase):
    def test_anchor_has_no_initial_jump(self):
        left = dict.fromkeys(t.BODY, 1500)
        right = dict.fromkeys(t.BODY, 2500)
        self.assertEqual(t.relative_goal(left, left, right), right)

    def test_roll_wrap_and_small_steps(self):
        anchor = dict.fromkeys(t.BODY, 2000)
        anchor['wrist_roll'] = 4090
        current = dict(anchor, wrist_roll=10)
        right = dict.fromkeys(t.BODY, 2000)
        desired = t.relative_goal(current, anchor, right)
        self.assertEqual(desired['wrist_roll'], 2016)
        self.assertEqual(t.next_jog(desired, right), ('wrist_roll', 11))
        self.assertIsNone(t.next_jog(right, right))

    def test_follower_fault_rejected(self):
        for field, value in [('Status', 8), ('Torque_Enable', 0), ('Present_Temperature', 58)]:
            s = follower()
            s['readings']['elbow_flex'][field] = value
            with self.assertRaises(RuntimeError): t.require_follower(s)

    def test_captures_do_not_overwrite_and_are_not_auto_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            session = t.TeachingSession()
            session.path = Path(directory) / 'session.json'
            right = follower()
            session.capture('grasp', {'elbow_flex': 1800}, right)
            session.capture('lift', {'elbow_flex': 1900}, right)
            saved = session.path.read_text()
            with self.assertRaises(ValueError): session.capture('grasp', {}, right)
            self.assertEqual(saved, session.path.read_text())
            poses = json.loads(saved)['poses']['bottom_left']
            self.assertEqual(set(poses), {'grasp', 'lift'})
            self.assertFalse(poses['grasp']['validation']['ready_for_motion'])
            right['manual_move_complete'] = False
            with self.assertRaises(RuntimeError): session.capture('place', {}, right)

    def run_reader(self, input_fault=False, fault_motor='elbow_flex'):
        bus = FakeBus()
        bus.connect = lambda **kwargs: setattr(bus, 'is_connected', True)
        original_read = bus.read
        left_fault = [False]
        def read(register, motor, **kwargs):
            if input_fault and left_fault[0] and motor == fault_motor and register == 'Torque_Enable':
                raise ConnectionError('Input voltage error')
            return original_read(register, motor, **kwargs)
        bus.read = read
        right = follower()
        jogs, moved = [], [False]
        def get_status():
            right['seconds'] += .3
            return copy.deepcopy(right)
        def jog(motor, ticks):
            jogs.append((motor, ticks))
            right['manual_target_raw'][motor] += ticks
            right['readings'][motor]['Present_Position'] += ticks
            left_fault[0] = True
        original_update = t.update
        def update(**values):
            original_update(**values)
            if values.get('phase') == 'following' and not moved[0]:
                moved[0] = True
                bus.positions['elbow_flex'] += 10
            if values.get('phase') == 'following' and jogs and not input_fault:
                t.shutdown.set()
            if values.get('error') and left_fault[0]: t.shutdown.set()
            if values.get('gripper_warning') and fault_motor == 'gripper': t.shutdown.set()
        t.shutdown.clear()
        t.state.clear()
        while not t.commands.empty(): t.commands.get_nowait()
        t.commands.put('start')
        with patch.object(t, 'SO100Leader', return_value=SimpleNamespace(bus=bus)), \
             patch.object(t.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='')), \
             patch.object(t, 'follower_status', side_effect=get_status), \
             patch.object(t, 'follower_jog', side_effect=jog), \
             patch.object(t, 'update', side_effect=update), \
             patch.object(t.shutdown, 'wait', return_value=False):
            t.run_reader()
        self.assertEqual(bus.writes, [])
        self.assertEqual(jogs, [('elbow_flex', 10)])
        self.assertFalse(bus.is_connected)
        self.assertTrue(all(r['Torque_Enable'] == 1 for r in right['readings'].values()))

    def test_one_leader_reader_forwards_bounded_http_jog(self):
        self.run_reader()

    def test_input_fault_pauses_without_cutting_follower_torque(self):
        self.run_reader(input_fault=True)
        self.assertIn('Input voltage error', t.state['error'])
        self.assertFalse(t.state['following'])

    def test_unused_gripper_fault_visible_but_not_used_for_body_control(self):
        self.run_reader(input_fault=True, fault_motor='gripper')
        self.assertIn('Input voltage error', t.state['gripper_warning'])
        self.assertIsNone(t.state['error'])
        self.assertFalse(t.state['gripper_input_enabled'])


if __name__ == '__main__': unittest.main()
