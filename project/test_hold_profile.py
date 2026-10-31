import unittest
from types import SimpleNamespace
from unittest.mock import patch
import hold_arm_pose as h


class FakeBus:
    def __init__(self, alarm=False):
        self.is_connected = False
        self.motors = dict.fromkeys((*h.BODY, 'gripper'))
        self.positions = dict.fromkeys(self.motors, 2000)
        self.registers, self.writes = {}, []
        self.alarm = alarm

    def connect(self): self.is_connected = True
    def disconnect(self, **kwargs): self.is_connected = False

    def read(self, register, motor, **kwargs):
        if register == 'Present_Position': return self.positions[motor]
        if register == 'Status' and self.alarm and self.registers.get(('Torque_Enable', motor)):
            return 8
        defaults = {'Min_Position_Limit': 1000, 'Max_Position_Limit': 3000,
                    'Present_Temperature': 30, 'Present_Voltage': 120,
                    'Min_Voltage_Limit': 60, 'Max_Voltage_Limit': 140,
                    'Goal_Velocity': 2, 'Lock': 1, 'P_Coefficient': 16}
        return self.registers.get((register, motor), defaults.get(register, 0))

    def write(self, register, motor, value, **kwargs):
        self.writes.append((register, motor, value))
        self.registers[register, motor] = value
        if register == 'Goal_Position': self.positions[motor] = value


class ProfileTests(unittest.TestCase):
    def test_recovery_only_expands_elbow_entry(self):
        limits = h.route_entry_limit('/transfer24')
        self.assertEqual(limits['elbow_flex'], 450)
        self.assertTrue(all(limits[m] == 90 for m in h.BODY if m != 'elbow_flex'))
        self.assertEqual(h.route_entry_limit('/dryrun'), 35)

    def run_profile(self, alarm):
        bus = FakeBus(alarm)
        robot = SimpleNamespace(bus=bus, is_calibrated=True)
        h.stop.clear()
        h.state.clear()
        h.state.update(error=None)
        original_update = h.update
        def update(**values):
            original_update(**values)
            if values.get('route_complete'): h.stop.set()
        def off(robot, motor):
            bus.write('Torque_Enable', motor, 0)
            return True
        route = [('test', dict.fromkeys(h.BODY, 2004))]
        with patch.object(h, 'SO100Follower', return_value=robot), \
             patch.object(h, 'update', side_effect=update), \
             patch.object(h, 'stop_motor', side_effect=off), \
             patch.object(h.stop, 'wait', return_value=False), patch('builtins.print'):
            h.hold(route=route, route_p=24)
        self.assertEqual([v for r, m, v in bus.writes if r == 'P_Coefficient'], [24, 16])
        self.assertTrue(h.state['gain_restored'])
        self.assertTrue(h.state['torque_off_verified'])
        self.assertFalse(bus.is_connected)
        self.assertFalse(any(m == 'gripper' or r in ('Lock', 'Torque_Limit', 'Max_Torque_Limit')
                             for r, m, v in bus.writes))
        return dict(h.state)

    def test_success_restores_gain(self):
        self.assertTrue(self.run_profile(False)['route_complete'])

    def test_alarm_restores_gain(self):
        self.assertEqual(self.run_profile(True)['phase'], 'fault')


if __name__ == '__main__': unittest.main()
