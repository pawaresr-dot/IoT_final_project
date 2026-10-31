import unittest
from types import SimpleNamespace
from unittest.mock import patch
import leader_follower_bridge as bridge
from test_hold_profile import FakeBus


class BridgeTests(unittest.TestCase):
    def test_voltage_drop_allowed_but_limits_and_servo_alarm_still_stop(self):
        for voltage, alarm, expected_fault in [(113, False, False), (59, False, True),
                                                (141, False, True), (113, True, True)]:
            with self.subTest(voltage=voltage, alarm=alarm):
                left, right = FakeBus(), FakeBus(alarm=alarm)
                left.read_calibration = lambda: {
                    m: SimpleNamespace(range_min=1000, range_max=3000) for m in left.motors}
                left.sync_read = lambda *a, **k: dict(left.positions)
                right.sync_read = lambda *a, **k: dict(right.positions)
                original_read = right.read
                def read(register, motor, **kwargs):
                    if register == 'Present_Voltage' and right.registers.get(('Torque_Enable', motor)):
                        return voltage
                    return original_read(register, motor, **kwargs)
                right.read = read
                bridge.shutdown.clear()
                bridge.state.clear()
                while not bridge.commands.empty(): bridge.commands.get_nowait()
                bridge.commands.put('start')
                original_update = bridge.update
                armed_samples = []
                def update(**values):
                    original_update(**values)
                    if 'target_raw' in values:
                        armed_samples.append(True)
                        if len(armed_samples) >= 3: bridge.shutdown.set()
                with patch.object(bridge, 'SO100Leader', return_value=SimpleNamespace(bus=left)), \
                     patch.object(bridge, 'SO100Follower', return_value=SimpleNamespace(bus=right, is_calibrated=True)), \
                     patch.object(bridge, 'update', side_effect=update), \
                     patch.object(bridge, 'stop_motor', side_effect=lambda r,m: (right.write('Torque_Enable',m,0) or True)), \
                     patch.object(bridge.shutdown, 'wait', return_value=False):
                    bridge.worker()
                self.assertEqual(bridge.state['phase'] == 'fault', expected_fault)
                self.assertTrue(bridge.state['torque_off_verified'])
                if not expected_fault: self.assertGreaterEqual(len(armed_samples), 3)

    def test_follow_hold_off_has_single_owner_and_no_jump(self):
        left,right=FakeBus(),FakeBus()
        left.read_calibration=lambda:{m:SimpleNamespace(range_min=1000,range_max=3000)
                                      for m in left.motors}
        left.sync_read=lambda *a,**k:dict(left.positions)
        right.sync_read=lambda *a,**k:dict(right.positions)
        bridge.shutdown.clear()
        bridge.state.clear()
        while not bridge.commands.empty():bridge.commands.get_nowait()
        bridge.commands.put('start')
        original=bridge.update
        moved=[False]
        def update(**values):
            original(**values)
            if values.get('phase')=='following':left.positions['elbow_flex']+=10
            if values.get('target_raw',{}).get('elbow_flex',2000)>2000 and not moved[0]:
                moved[0]=True
                bridge.commands.put('hold')
            if values.get('phase')=='holding':bridge.commands.put('off')
            if values.get('phase')=='connected' and bridge.state.get('torque_off_verified'):
                bridge.shutdown.set()
        with patch.object(bridge,'SO100Leader',return_value=SimpleNamespace(bus=left)), \
             patch.object(bridge,'SO100Follower',return_value=SimpleNamespace(bus=right,is_calibrated=True)), \
             patch.object(bridge,'update',side_effect=update), \
             patch.object(bridge,'stop_motor',side_effect=lambda r,m:(right.write('Torque_Enable',m,0) or True)), \
             patch.object(bridge.shutdown,'wait',return_value=False):
            bridge.worker()
        self.assertTrue(moved[0])
        self.assertEqual(left.writes,[])
        self.assertFalse(any(m=='gripper' or r in ('P_Coefficient','I_Coefficient','Lock')
                             for r,m,v in right.writes))
        for motor in bridge.BODY:
            goals=[v for r,m,v in right.writes if r=='Goal_Position' and m==motor]
            self.assertEqual(goals[0],2000)
        self.assertTrue(bridge.state['torque_off_verified'])

    def test_capture_has_no_jump(self):
        leader = dict.fromkeys(bridge.BODY, 1200)
        follower = dict.fromkeys(bridge.BODY, 2200)
        limits = {m: (1000,3000) for m in bridge.BODY}
        self.assertEqual(bridge.relative_targets(leader,leader,follower,limits),follower)

    def test_delta_mapping_excludes_gripper(self):
        anchor = dict.fromkeys(bridge.BODY, 1200)
        moved = dict(anchor, elbow_flex=1210, gripper=4095)
        target = bridge.relative_targets(moved,anchor,dict.fromkeys(bridge.BODY,2000),
                                         {m:(1000,3000) for m in bridge.BODY})
        self.assertEqual(target['elbow_flex'],2010)
        self.assertNotIn('gripper',target)

    def test_limit_rejects(self):
        anchor = dict.fromkeys(bridge.BODY,1200)
        with self.assertRaises(ValueError):
            bridge.relative_targets(dict(anchor,elbow_flex=2500),anchor,
                                    dict.fromkeys(bridge.BODY,2000),
                                    {m:(1000,3000) for m in bridge.BODY})

    def test_roll_wrap(self):
        anchor=dict.fromkeys(bridge.BODY,2000)
        anchor['wrist_roll']=4090
        moved=dict(anchor,wrist_roll=10)
        result=bridge.relative_targets(moved,anchor,dict.fromkeys(bridge.BODY,2000),
                                       {m:(1000,3000) for m in bridge.BODY})
        self.assertEqual(result['wrist_roll'],2016)

    def test_startup_is_read_only(self):
        left,right=FakeBus(),FakeBus()
        left.read_calibration=lambda:{m:SimpleNamespace(range_min=1000,range_max=3000)
                                      for m in left.motors}
        left.sync_read=lambda *a,**k:dict(left.positions)
        right.sync_read=lambda *a,**k:dict(right.positions)
        bridge.shutdown.clear()
        bridge.state.clear()
        original=bridge.update
        def update(**values):
            original(**values)
            if 'leader_raw' in values:bridge.shutdown.set()
        with patch.object(bridge,'SO100Leader',return_value=SimpleNamespace(bus=left)), \
             patch.object(bridge,'SO100Follower',return_value=SimpleNamespace(bus=right,is_calibrated=True)), \
             patch.object(bridge,'update',side_effect=update):
            bridge.worker()
        self.assertEqual(left.writes,[])
        self.assertEqual(right.writes,[])
        self.assertIn('leader_raw',bridge.state)
        self.assertFalse(left.is_connected or right.is_connected)


if __name__=='__main__':unittest.main()
