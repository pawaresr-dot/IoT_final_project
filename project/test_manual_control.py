import unittest
from types import SimpleNamespace
from unittest.mock import patch

import hold_arm_pose as h
from test_hold_profile import FakeBus


class ManualTests(unittest.TestCase):
    def test_jog_bounds_and_immutability(self):
        target = {'elbow_flex': 2000, 'gripper': 2000}
        limits = {'elbow_flex': (1000, 3000), 'gripper': (1000, 3000)}
        self.assertEqual(h.jog_target(target, 'elbow_flex', 11, limits)['elbow_flex'], 2011)
        self.assertEqual(target['elbow_flex'], 2000)
        for motor, ticks in [('elbow_flex', 23), ('gripper', 81), ('gripper', True), ('unknown', 1)]:
            with self.assertRaises(ValueError): h.jog_target(target, motor, ticks, limits)
        with self.assertRaises(ValueError):
            h.jog_target({'elbow_flex': 2990}, 'elbow_flex', 22, limits)

    def run_manual(self, alarm=False, hot=False):
        bus = FakeBus()
        original_read = bus.read
        def read(register, motor, **kwargs):
            if motor == 'gripper' and bus.registers.get(('Torque_Enable', motor)):
                if alarm and register == 'Status': return 8
                if hot and register == 'Present_Temperature': return 58
            return original_read(register, motor, **kwargs)
        bus.read = read
        h.stop.clear()
        h.state.clear()
        h.state.update(error=None)
        while not h.manual_commands.empty(): h.manual_commands.get_nowait()
        original_update = h.update
        turns = []
        def update(**values):
            original_update(**values)
            if values.get('phase') == 'holding':
                h.manual_commands.put({'motor': 'elbow_flex', 'ticks': 11})
                h.manual_commands.put({'motor': 'gripper', 'ticks': 80})
            if 'manual_target_raw' in values:
                turns.append(True)
                if (values.get('manual_move_complete') and bus.positions['gripper'] == 2080) or len(turns) > 120:
                    h.stop.set()
        with patch.object(h, 'SO100Follower', return_value=SimpleNamespace(bus=bus, is_calibrated=True)), \
             patch.object(h, 'update', side_effect=update), \
             patch.object(h, 'stop_motor', side_effect=lambda r,m: (bus.write('Torque_Enable', m, 0) or True)), \
             patch.object(h.stop, 'wait', return_value=False), patch('builtins.print'):
            h.hold(manual=True)
        self.assertTrue(h.state['torque_off_verified'])
        self.assertFalse(bus.is_connected)
        self.assertFalse(any(reg in ('P_Coefficient', 'I_Coefficient', 'Lock', 'Min_Voltage_Limit', 'Max_Voltage_Limit')
                             for reg, motor, value in bus.writes))
        self.assertTrue(all(bus.registers.get(('Torque_Enable', m), 0) == 0 for m in bus.motors))
        return bus

    def test_body_and_gripper_have_one_owner_and_cleanup(self):
        bus = self.run_manual()
        self.assertEqual(bus.positions['elbow_flex'], 2011)
        self.assertEqual(bus.positions['gripper'], 2080)
        for motor in bus.motors:
            first_goal = next(value for reg, m, value in bus.writes if reg == 'Goal_Position' and m == motor)
            self.assertEqual(first_goal, 2000)
        self.assertEqual(h.state['phase'], 'stopped')

    def test_gripper_alarm_stops_all_enabled_motors(self):
        self.run_manual(alarm=True)
        self.assertEqual(h.state['phase'], 'fault')
        self.assertEqual(h.state['last_fault']['motor'], 'gripper')

    def test_gripper_temperature_fault_captured(self):
        self.run_manual(hot=True)
        self.assertEqual(h.state['last_fault']['reading']['Present_Temperature'], 58)


if __name__ == '__main__': unittest.main()
