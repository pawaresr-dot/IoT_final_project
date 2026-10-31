import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import hand_teach as h
from test_hold_profile import FakeBus


def session():
    first = dict.fromkeys(h.MOTORS, 2000)
    last = dict(first, elbow_flex=2010)
    return {'schema': 'hand_teach_v1', 'robot_id': h.ROBOT_ID,
            'limits': dict.fromkeys(h.MOTORS, [1000, 3000]),
            'homing_offsets': dict.fromkeys(h.MOTORS, 0),
            'frames': [{'t': 0., 'raw_position': first, 'gripper_goal_raw': 2000},
                       {'t': 1., 'raw_position': last, 'gripper_goal_raw': 2004}]}


class HandTeachTests(unittest.TestCase):
    def test_validate_rejects_bad_identity_limits_and_time(self):
        limits = dict.fromkeys(h.MOTORS, (1000, 3000))
        self.assertEqual(len(h.validate_session(session(), limits)), 2)
        for kind in ('identity', 'position', 'time', 'gripper'):
            data = session()
            if kind == 'identity': data['robot_id'] = 'other'
            elif kind == 'position': data['frames'][1]['raw_position']['elbow_flex'] = 4000
            elif kind == 'time': data['frames'][1]['t'] = -1
            else: data['frames'][1]['gripper_goal_raw'] = 999
            with self.assertRaises(ValueError): h.validate_session(data, limits)

    def test_keyframes_keep_jaw_changes_and_final_pose(self):
        data = session()
        middle = copy.deepcopy(data['frames'][0]); middle.update(t=.2, gripper_goal_raw=2010)
        frames = [data['frames'][0], middle, data['frames'][1]]
        self.assertEqual(h.keyframes(frames), frames)

    def test_gripper_command_lead_is_bounded(self):
        self.assertEqual(h.gripper_step(2000, 1900, 2000, (1000,3000)), 1996)
        self.assertEqual(h.gripper_step(1970, 1900, 2000, (1000,3000)), 1970)
        with self.assertRaises(ValueError): h.gripper_step(2000, 999, 2000, (1000,3000))

    def test_return_uses_observed_reverse_body_positions(self):
        frames = session()['frames']
        middle = copy.deepcopy(frames[0]); middle.update(t=.5, gripper_goal_raw=2500)
        self.assertEqual(h.return_frames([frames[0], middle, frames[1]]),
                         [frames[1], middle, frames[0]])

    def test_near_start_alignment_stages_lift_before_base(self):
        data = session(); initial = dict.fromkeys(h.MOTORS, 2150)
        frames, mode = h.return_entry(data['frames'], initial)
        self.assertEqual(mode, 'near_start_alignment')
        self.assertEqual(frames[1]['raw_position']['shoulder_pan'], 2150)
        self.assertEqual(frames[1]['raw_position']['shoulder_lift'], 2000)
        self.assertEqual(frames[-1], data['frames'][0])
        initial['shoulder_pan'] = 2400
        with self.assertRaises(RuntimeError): h.return_entry(data['frames'], initial)

    def test_cycle_entry_returns_before_replaying_if_at_endpoint(self):
        frames = session()['frames']
        frames[-1]['raw_position']['shoulder_pan'] = 2500
        route, returning, mode = h.cycle_entry(frames, frames[-1]['raw_position'])
        self.assertTrue(returning)
        self.assertEqual(route[0],frames[-1])
        self.assertEqual(route[-1],frames[0])

    def test_health_block_decodes_positions_temperature_status_and_signed_load(self):
        data=[0]*15
        for index,value in ((0,2515),(2,32768+12),(4,1024+40),(13,7)):
            data[index:index+2]=[value&255,value>>8]
        data[6],data[7],data[9]=119,43,0
        row=h.decode_health_block(data)
        self.assertEqual(row['Present_Position'],2515)
        self.assertEqual(row['Present_Velocity'],-12)
        self.assertEqual(row['Present_Load'],-40)
        self.assertEqual(row['Present_Temperature'],43)
        self.assertEqual(row['Present_Current'],7)
        for invalid in ([43],data+[0],[256]*15):
            with self.assertRaises(ValueError):h.decode_health_block(invalid)

    def test_health_block_rejects_short_replies_and_packet_alarms(self):
        bus=SimpleNamespace(motors={'gripper':SimpleNamespace(id=6,model='sts3215')},
                            port_handler=SimpleNamespace(clearPort=lambda:None),
                            packet_handler=SimpleNamespace(readTxRx=lambda *args:([43],0,0)))
        with self.assertRaises(RuntimeError):h.read_health(bus,'gripper')
        bus.packet_handler.readTxRx=lambda *args:([0]*15,0,4)
        with self.assertRaises(RuntimeError):h.read_health(bus,'gripper')

    def test_temperature_single_jump_requires_three_cool_alarm_free_rereads(self):
        calls = []
        def read(reg,m):
            calls.append(reg)
            return 0 if reg=='Status' else 43
        accepted, samples = h.confirm_temperature(read,'gripper',57,43,pause=lambda _:None)
        self.assertEqual(accepted,43)
        self.assertEqual(len(samples),3)
        self.assertEqual(calls,['Status','Present_Temperature']*3)

    def test_temperature_alarm_or_elevated_confirmation_is_not_filtered(self):
        for reported, previous, status, temperature in ((57,43,0,57),(57,43,0,51),(57,43,8,43),
                                                       (65,43,0,43),(57,None,0,43),(57,51,0,43)):
            with self.assertRaises(RuntimeError):
                h.confirm_temperature(lambda reg,m: status if reg=='Status' else temperature,
                                      'gripper',reported,previous,pause=lambda _:None)

    def recording(self, grip=False):
        with tempfile.TemporaryDirectory() as directory:
            c, bus = h.Controller(directory), FakeBus()
            original = c.update
            def update(**values):
                original(**values)
                if values.get('phase') == 'recording' and grip: c.commands.put('open')
                if values.get('recorded_frames'):
                    bus.positions['elbow_flex'] += 2
                    if values['recorded_frames'] >= 10: c.stop.set()
            with patch.object(h, 'SO100Follower', return_value=SimpleNamespace(bus=bus, is_calibrated=True)), \
                 patch.object(h.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='')), \
                 patch.object(c, 'update', side_effect=update), \
                 patch.object(c.stop, 'wait', return_value=False), \
                 patch.object(h, 'stop_motor', side_effect=lambda r,m: (bus.write('Torque_Enable',m,0) or True)), patch('builtins.print'):
                c.run('record')
            self.assertEqual(c.snapshot()['phase'], 'recorded')
            self.assertTrue(c.last_file.exists())
            saved = json.loads(c.last_file.read_text())
            self.assertGreaterEqual(len(saved['frames']), 10)
            self.assertFalse(saved['validation']['object_pickup_verified'])
            self.assertTrue(saved['recording_complete'])
            restored = h.Controller(directory)
            self.assertEqual(restored.last_file, c.last_file)
            self.assertTrue(restored.snapshot()['restored'])
            self.assertFalse(any(m in h.BODY for reg,m,v in bus.writes))
            self.assertFalse(bus.is_connected)
            if grip:
                self.assertTrue(bus.writes)
                self.assertTrue(c.snapshot()['torque_off_verified'])
            else: self.assertEqual(bus.writes, [])

    def test_recording_never_energizes_body(self): self.recording()
    def test_gripper_buttons_share_recording_worker_and_save_commands(self): self.recording(grip=True)

    def playback(self, alarm=False, bad_entry=False, bad_offsets=False, operation='replay', elbow_p=None, offset=0,
                 software_trip=True, reported_temperature=None, firmware_protection=44,shoulder_p=None):
        with tempfile.TemporaryDirectory() as directory:
            c, bus = h.Controller(directory), FakeBus(alarm=alarm)
            data = session()
            c.elbow_p, c.replay_from = elbow_p, offset
            c.shoulder_p = shoulder_p
            c.software_temperature_trip = software_trip
            if not software_trip:
                for m in h.MOTORS:
                    bus.registers['Unloading_Condition',m] = firmware_protection
                    bus.registers['Max_Temperature_Limit',m] = 70
            if reported_temperature is not None:
                bus.registers['Present_Temperature','shoulder_lift'] = reported_temperature
            if operation == 'adopt_replay':
                for m in h.BODY:
                    bus.registers['Torque_Enable', m] = 1
                    bus.registers['Goal_Position', m] = 2000
            if operation == 'return': bus.positions.update(data['frames'][-1]['raw_position'])
            if bad_entry: bus.positions['shoulder_pan'] = 2500 if operation == 'return' else 2200
            if bad_offsets: data['homing_offsets']['elbow_flex'] = 1
            c.last_file = Path(directory) / 'taught.json'
            c.last_file.write_text(json.dumps(data))
            original = c.update
            def update(**values):
                original(**values)
                if values.get('phase') in ('complete', 'ready_start', 'cycle_complete'): c.stop.set()
            with patch.object(h, 'SO100Follower', return_value=SimpleNamespace(bus=bus, is_calibrated=True)), \
                 patch.object(h.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='')), \
                 patch.object(c, 'update', side_effect=update), \
                 patch.object(c.stop, 'wait', return_value=False), \
                 patch.object(h, 'stop_motor', side_effect=lambda r,m: (bus.write('Torque_Enable',m,0) or True)), patch('builtins.print'):
                c.run(operation)
            self.assertFalse(bus.is_connected)
            if bad_entry or bad_offsets or (not software_trip and not firmware_protection & 4):
                if operation != 'adopt_replay': self.assertEqual(bus.writes, [])
                self.assertEqual(c.snapshot()['phase'], 'fault')
            else:
                self.assertTrue(c.snapshot()['torque_off_verified'])
                active = h.BODY if operation == 'return' or (operation == 'adopt_replay' and alarm) else h.MOTORS
                self.assertTrue(all(bus.registers.get(('Torque_Enable',m)) == 0 for m in active))
                if operation == 'return':
                    self.assertFalse(any(m=='gripper' for reg,m,v in bus.writes))
                self.assertFalse(any(reg in ('Lock','I_Coefficient','Min_Voltage_Limit') for reg,m,v in bus.writes))
                self.assertFalse(any(reg in ('Unloading_Condition','Max_Temperature_Limit') for reg,m,v in bus.writes))
                if not software_trip:
                    self.assertTrue(c.snapshot()['firmware_thermal_protection_verified'])
                    self.assertFalse(c.snapshot()['software_temperature_cutoff'])
                    if reported_temperature is not None and not alarm:
                        self.assertEqual(c.snapshot()['readings']['shoulder_lift']['Present_Temperature'],reported_temperature)
                p_writes = [v for reg,m,v in bus.writes if reg=='P_Coefficient']
                profile_count = bool(elbow_p) + bool(shoulder_p)
                self.assertEqual(p_writes,[24]*profile_count + [16]*profile_count)
                if shoulder_p:self.assertTrue(c.snapshot()['shoulder_p_restored'])
                for motor in active:
                    if operation == 'adopt_replay' and motor in h.BODY:
                        self.assertFalse(any(reg=='Torque_Enable' and m==motor and v==1 for reg,m,v in bus.writes))
                        continue
                    expected = data['frames'][-1]['raw_position'][motor] if operation == 'return' else 2000
                    self.assertEqual(next(v for reg,m,v in bus.writes if reg=='Goal_Position' and m==motor),expected)
                if not alarm:
                    self.assertEqual(c.snapshot()['phase'], 'stopped')
                    self.assertEqual(c.snapshot()['playback_index'], 2-offset)
                    self.assertFalse(c.snapshot()['object_pickup_verified'])
                else: self.assertEqual(c.snapshot()['phase'], 'fault')

    def test_replay_seeds_current_goals_and_finishes_taught_frames(self): self.playback()
    def test_replay_alarm_disables_all_enabled_motors(self): self.playback(alarm=True)
    def test_wrong_start_does_not_enable_torque(self): self.playback(bad_entry=True)
    def test_changed_calibration_does_not_enable_torque(self): self.playback(bad_offsets=True)
    def test_return_finishes_at_start_without_any_gripper_writes(self): self.playback(operation='return')
    def test_return_refuses_wrong_end_without_writes(self): self.playback(bad_entry=True, operation='return')
    def test_return_alarm_cleans_up_body(self): self.playback(alarm=True, operation='return')
    def test_adopted_hold_replays_without_body_torque_cycle(self): self.playback(operation='adopt_replay')
    def test_adopted_hold_alarm_stops_body(self): self.playback(alarm=True, operation='adopt_replay')
    def test_resume_with_temporary_elbow_profile_restores_p(self): self.playback(operation='adopt_replay',elbow_p=24,offset=1)
    def test_single_cycle_replays_then_automatically_opens_gripper(self): self.playback(operation='cycle')
    def test_software_temp_trip_can_be_disabled_without_changing_raw_telemetry(self):
        self.playback(software_trip=False,reported_temperature=150)
    def test_servo_status_alarm_still_stops_with_software_temperature_trip_disabled(self):
        self.playback(software_trip=False,reported_temperature=150,alarm=True)
    def test_disabling_software_trip_requires_servo_thermal_protection(self):
        self.playback(software_trip=False,firmware_protection=40)
    def test_temporary_shoulder_and_elbow_profiles_restore_without_unlocking(self):
        self.playback(operation='adopt_replay',elbow_p=24,shoulder_p=24,offset=1)
    def test_shoulder_profile_restores_on_hardware_alarm(self):
        self.playback(operation='adopt_replay',shoulder_p=24,alarm=True)

    def test_held_return_then_replay_and_temporary_grip_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            c, bus = h.Controller(directory), FakeBus()
            c.last_file = Path(directory) / 'taught.json'
            c.last_file.write_text(json.dumps(session()))
            c.gripper_torque = 600
            for m in h.MOTORS:
                bus.registers['Torque_Enable',m] = 1
                bus.registers['Goal_Position',m] = 2000
            bus.registers['Torque_Limit','gripper'] = 500
            bus.registers['Max_Torque_Limit','gripper'] = 500
            original = c.update
            phases = []
            def update(**values):
                original(**values)
                phase = values.get('phase')
                if phase: phases.append(phase)
                if phase == 'holding': c.commands.put('return')
                elif phase == 'ready_start': c.commands.put('replay')
                elif phase == 'complete': c.commands.put('release')
                elif phase == 'released': c.stop.set()
            with patch.object(h, 'SO100Follower', return_value=SimpleNamespace(bus=bus,is_calibrated=True)), \
                 patch.object(h.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='')), \
                 patch.object(c,'update',side_effect=update), patch.object(c.stop,'wait',return_value=False), \
                 patch.object(h,'stop_motor',side_effect=lambda r,m:(bus.write('Torque_Enable',m,0) or True)):
                c.run('adopt_hold')
            self.assertIn('ready_start',phases)
            self.assertIn('complete',phases)
            self.assertIn('released',phases)
            self.assertTrue(c.snapshot()['release_motion_verified'])
            self.assertEqual([v for reg,m,v in bus.writes if reg=='Torque_Limit'],[600,500])
            self.assertTrue(c.snapshot()['gripper_torque_restored'])
            self.assertFalse(any(reg in ('Max_Torque_Limit','Unloading_Condition','Lock','Protection_Current','P_Coefficient') for reg,m,v in bus.writes))
            self.assertFalse(any(reg=='Torque_Enable' and v==1 for reg,m,v in bus.writes))


if __name__ == '__main__': unittest.main()
