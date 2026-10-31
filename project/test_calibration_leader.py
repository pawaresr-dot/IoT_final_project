import unittest
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hand_teach as h
from test_hand_teach import session
from test_hold_profile import FakeBus
from calibration_leader import BODY,next_jog,checked_positions,gripper_target,motion_timer
from hand_teach import Controller


class CalibrationControlTests(unittest.TestCase):
    def test_stream_mailbox_keeps_only_latest_target_and_validates_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            c=Controller(directory);c.thread=SimpleNamespace(is_alive=lambda:True)
            c.calibration_local_limit=None
            c.update(phase='holding',body_joint_limits={m:[1000,3000] for m in BODY},body_origin_raw=dict.fromkeys(BODY,2000))
            c.teleop(dict.fromkeys(BODY,2010));c.teleop(dict.fromkeys(BODY,2020))
            self.assertEqual(c.teleop_pending[0],dict.fromkeys(BODY,2020))
            self.assertTrue(c.commands.empty())
            with self.assertRaises(ValueError):c.teleop(dict.fromkeys(BODY,4000))
            with self.assertRaises(ValueError):c.teleop(dict(dict.fromkeys(BODY,2000),gripper=2100))
            self.assertEqual(c.teleop_pending[0],dict.fromkeys(BODY,2020))

    def test_stream_watchdog_uses_recent_input_not_motor_speed(self):
        self.assertFalse(h.stream_input_stale(100,100.5))
        self.assertTrue(h.stream_input_stale(100,100.8))

    def test_stream_moves_all_body_joints_and_restores_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            c,bus=Controller(directory),FakeBus()
            c.last_file=Path(directory)/'session.json';c.last_file.write_text(json.dumps(session()))
            update=c.update
            def observe(**values):
                update(**values)
                if values.get('phase')=='holding':c.teleop_pending=(dict.fromkeys(BODY,2012),time.monotonic())
                if values.get('command_raw') and c.snapshot()['phase']=='teleoperating' and all(bus.positions[m]==2012 for m in BODY):c.stop.set()
            with patch.object(h,'SO100Follower',return_value=SimpleNamespace(bus=bus,is_calibrated=True)), \
                 patch.object(h.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='')), \
                 patch.object(c,'update',side_effect=observe),patch.object(c.stop,'wait',return_value=False), \
                 patch.object(h,'stop_motor',side_effect=lambda r,m:(bus.write('Torque_Enable',m,0) or True)):
                c.run('hold')
            self.assertTrue(all(bus.positions[m]==2012 for m in BODY))
            self.assertEqual(bus.positions['gripper'],2000)
            for m in BODY:
                speeds=[v for reg,motor,v in bus.writes if reg=='Goal_Velocity' and motor==m]
                self.assertIn(6,speeds)
                self.assertEqual(speeds[-1],2)

    def test_hold_from_torque_off_seeds_current_pose_not_old_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            c,bus=Controller(directory),FakeBus()
            bus.positions.update(dict.fromkeys(h.MOTORS,2050))
            c.last_file=Path(directory)/'session.json'
            c.last_file.write_text(json.dumps(session()))
            update=c.update
            def observe(**values):
                update(**values)
                if values.get('phase')=='holding':c.stop.set()
            with patch.object(h,'SO100Follower',return_value=SimpleNamespace(bus=bus,is_calibrated=True)), \
                 patch.object(h.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='')), \
                 patch.object(c,'update',side_effect=observe), \
                 patch.object(h,'stop_motor',side_effect=lambda r,m:(bus.write('Torque_Enable',m,0) or True)):
                c.run('hold')
            goals=[(m,v) for reg,m,v in bus.writes if reg=='Goal_Position']
            self.assertEqual(len(goals),6)
            self.assertTrue(all(v==2050 for m,v in goals))
            for m in h.MOTORS:
                self.assertLess(bus.writes.index(('Goal_Position',m,2050)),bus.writes.index(('Torque_Enable',m,1)))

    def test_full_scale_still_uses_small_bounded_commands(self):
        anchor={m:2000 for m in BODY};right={m:2500 for m in BODY}
        left=dict(anchor,elbow_flex=2020)
        self.assertIsNone(next_jog(left,anchor,right,right,scale=.25))
        self.assertEqual(next_jog(left,anchor,right,right,scale=1.),('elbow_flex',11))

    def test_manual_jaw_after_idle_does_not_timeout_immediately(self):
        self.assertEqual(motion_timer(None,False,1000),(None,False))
        self.assertEqual(motion_timer(None,True,2000),(2000,False))
        self.assertEqual(motion_timer(2000,True,2036),(2000,True))
        self.assertEqual(motion_timer(2000,False,2036),(None,False))

    def test_gripper_mapping_direction_and_endpoint_clamping(self):
        config={'leader_closed_raw':1710,'leader_open_raw':2407,
                'follower_closed_raw':1930,'follower_open_raw':2373}
        self.assertEqual(gripper_target(1710,config),1930)
        self.assertEqual(gripper_target(2407,config),2373)
        self.assertEqual(gripper_target(3000,config),2373)
        self.assertEqual(gripper_target(1000,config),1930)
        with self.assertRaises(ValueError):gripper_target(1700,dict(config,leader_open_raw=1711))

    def test_disable_follower_local_radius_retains_joint_bounds(self):
        origin={m:2000 for m in BODY};actual=dict(origin,elbow_flex=2200)
        limits={m:(1000,3000) for m in BODY}
        with self.assertRaisesRegex(ValueError,'local range'):
            h.calibration_jog_target(actual,'elbow_flex',11,limits,origin)
        target=h.calibration_jog_target(actual,'elbow_flex',11,limits,origin,None)
        self.assertEqual(target['elbow_flex'],2211)
        with self.assertRaisesRegex(ValueError,'joint limits'):
            h.calibration_jog_target(dict(actual,elbow_flex=2995),'elbow_flex',11,limits,origin,None)
        with self.assertRaises(ValueError):
            h.calibration_jog_target(actual,'elbow_flex',100,limits,origin,None)

    def test_disabling_leader_range_keeps_small_output_commands(self):
        anchor={m:1000 for m in BODY};right={m:2500 for m in BODY}
        left=dict(anchor,wrist_flex=2200)
        self.assertEqual(next_jog(left,anchor,right,right,leader_range_limit=None),('wrist_flex',11))
        with self.assertRaises(ValueError):next_jog(left,anchor,right,right)

    def test_inconsistent_input_forwards_no_snapshot(self):
        previous={m:2000 for m in BODY}
        readings=iter([4000,2000,3000,2000])
        def read(key,m):return next(readings) if m=='elbow_flex' else 2000
        positions,rejected=checked_positions(read,previous)
        self.assertIsNone(positions)
        self.assertIn('elbow_flex',rejected)

    def test_transient_jump_requires_three_consistent_rereads(self):
        previous={m:2000 for m in BODY}
        readings=iter([4000,2000,2001,2000])
        positions,rejected=checked_positions(lambda key,m:next(readings) if m=='elbow_flex' else 2000,previous)
        self.assertEqual(positions['elbow_flex'],2000)
        self.assertEqual(rejected,{})

    def test_range_error_names_joint_and_anchor(self):
        anchor={m:1000 for m in BODY};right={m:2500 for m in BODY}
        with self.assertRaisesRegex(ValueError,'elbow_flex.*anchor'):
            next_jog(dict(anchor,elbow_flex=1800),anchor,right,right)

    def run_jog(self,motor='elbow_flex',ticks=11,jaw_action=None):
        with tempfile.TemporaryDirectory() as directory:
            c, bus = Controller(directory), FakeBus()
            for m in h.MOTORS:
                bus.registers['Torque_Enable',m]=1
                bus.registers['Goal_Position',m]=2000
            c.last_file=Path(directory)/'session.json'
            c.last_file.write_text(json.dumps(session()))
            update=c.update
            holding_count=0
            def observed_update(**values):
                nonlocal holding_count
                update(**values)
                if values.get('phase')=='holding':
                    holding_count+=1
                    if holding_count==1:c.commands.put(jaw_action or {'jog_motor':motor,'ticks':ticks})
                    else:c.stop.set()
            with patch.object(h,'SO100Follower',return_value=SimpleNamespace(bus=bus,is_calibrated=True)), \
                 patch.object(h.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='')), \
                 patch.object(c,'update',side_effect=observed_update), \
                 patch.object(c.stop,'wait',return_value=False), \
                 patch.object(h,'stop_motor',side_effect=lambda r,m:(bus.write('Torque_Enable',m,0) or True)):
                c.run('adopt_hold')
            self.assertEqual(bus.positions[motor],(1003 if jaw_action=='jaw_close' else 2373) if jaw_action else 2000+ticks)
            if motor!='gripper':self.assertFalse(any(reg=='Goal_Position' and m=='gripper' for reg,m,v in bus.writes))
            else:self.assertFalse(any(reg=='Goal_Position' and m in BODY for reg,m,v in bus.writes))
            self.assertEqual(holding_count,2)

    def test_jog_runs_inside_existing_owner_without_gripper_motion(self):self.run_jog()

    def test_gripper_jog_runs_inside_owner_without_body_motion(self):self.run_jog('gripper',-11)

    def test_one_click_jaw_close_finishes_without_body_motion(self):self.run_jog('gripper',jaw_action='jaw_close')

    def test_one_click_jaw_open_finishes_without_body_motion(self):self.run_jog('gripper',jaw_action='jaw_open')

    def test_relative_start_has_no_jump(self):
        left={m:1000 for m in BODY};right={m:2500 for m in BODY}
        self.assertIsNone(next_jog(left,left,right,right))

    def test_step_is_bounded(self):
        anchor={m:1000 for m in BODY};left=dict(anchor,elbow_flex=1060)
        right={m:2500 for m in BODY}
        self.assertEqual(next_jog(left,anchor,right,right),('elbow_flex',11))

    def test_large_leader_motion_does_not_forward(self):
        anchor={m:1000 for m in BODY};right={m:2500 for m in BODY}
        with self.assertRaises(ValueError):next_jog(dict(anchor,elbow_flex=1800),anchor,right,right)

    def test_leader_motion_is_scaled_down(self):
        anchor={m:1000 for m in BODY};right={m:2500 for m in BODY}
        self.assertIsNone(next_jog(dict(anchor,elbow_flex=1040),anchor,right,right))

    def test_controller_rejects_invalid_or_unheld_jog(self):
        c=Controller()
        for motor,ticks in (('gripper',11),('shoulder_pan',12),('shoulder_pan',True),('shoulder_pan',0),('shoulder_pan',1)):
            with self.assertRaises(ValueError):c.jog(motor,ticks)


if __name__=='__main__':unittest.main()
