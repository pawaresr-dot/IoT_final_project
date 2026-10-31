import unittest
from types import SimpleNamespace
from unittest.mock import patch
from test_hold_profile import FakeBus
from gripper_probe import probe


class GripperTests(unittest.TestCase):
    def test_read_only_default(self):
        bus = FakeBus()
        self.assertFalse(probe(SimpleNamespace(bus=bus))['motion_requested'])
        self.assertEqual(bus.writes, [])

    def test_open_return_only_writes_gripper(self):
        bus = FakeBus()
        robot = SimpleNamespace(bus=bus, is_calibrated=True)
        with patch('gripper_probe.time.sleep'), patch('gripper_probe.stop_motor',
                side_effect=lambda r,m: (bus.write('Torque_Enable',m,0) or True)):
            result = probe(robot, move=True)
        self.assertTrue(result['passed'])
        self.assertEqual([s['target'] for s in result['samples']], [2080, 2000])
        self.assertTrue(all(m == 'gripper' for _,m,_ in bus.writes))
        self.assertTrue(result['torque_off_verified'])

    def test_alarm_aborts_without_writes(self):
        bus = FakeBus()
        bus.registers['Status', 'gripper'] = 1
        with self.assertRaises(RuntimeError):
            probe(SimpleNamespace(bus=bus, is_calibrated=True), move=True)
        self.assertEqual(bus.writes, [])

    def test_temporary_torque_limit_restores_without_eeprom_writes(self):
        bus = FakeBus()
        bus.registers['Torque_Limit','gripper'] = 500
        with patch('gripper_probe.time.sleep'), patch('gripper_probe.stop_motor',
                side_effect=lambda r,m: (bus.write('Torque_Enable',m,0) or True)):
            result = probe(SimpleNamespace(bus=bus,is_calibrated=True),move=True,torque_limit=600)
        self.assertEqual(result['torque_limit_tested'],600)
        self.assertEqual(result['torque_limit_restored'],500)
        self.assertTrue(all(m=='gripper' for reg,m,v in bus.writes))
        self.assertFalse(any(reg in ('Max_Torque_Limit','Unloading_Condition','Lock','Protection_Current') for reg,m,v in bus.writes))

    def test_torque_limit_cannot_exceed_short_test_cap(self):
        bus = FakeBus();bus.registers['Torque_Limit','gripper'] = 500
        with self.assertRaises(ValueError):
            probe(SimpleNamespace(bus=bus,is_calibrated=True),move=True,torque_limit=700)
        self.assertEqual(bus.writes,[])


if __name__ == '__main__': unittest.main()
