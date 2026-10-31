import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import hand_teach as h
from test_hand_teach import session
from test_hold_profile import FakeBus


class JournalTests(unittest.TestCase):
    def test_interrupted_frames_preserved_but_not_replayable(self):
        with tempfile.TemporaryDirectory() as directory:
            data = session()
            frames = data.pop('frames')
            journal = h.RecordingJournal(Path(directory)/'capture.jsonl', data)
            for frame in frames: journal.frame(frame)
            journal.close(False, 'Device not configured')
            recovered = h.recover_journal(journal.path)
            self.assertEqual(recovered['frames'], frames)
            self.assertFalse(recovered['recording_complete'])
            with self.assertRaises(ValueError): h.validate_session(recovered, data['limits'])

    def test_torn_last_line_does_not_destroy_previous_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            data = session(); frames = data.pop('frames')
            journal = h.RecordingJournal(Path(directory)/'capture.jsonl', data)
            journal.frame(frames[0])
            journal.stream.write('{"type":"frame"')
            journal.stream.close()  # Simulate interrupted final write.
            recovered = h.recover_journal(journal.path)
            self.assertEqual(recovered['frames'], frames[:1])
            self.assertFalse(recovered['recording_complete'])

    def test_incomplete_json_not_auto_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/'hand_teach_sessions'; folder.mkdir()
            data = session(); data['recording_complete'] = False
            (folder/'interrupted.json').write_text(json.dumps(data))
            self.assertIsNone(h.Controller(directory).last_file)

    def test_usb_disconnect_preserves_observed_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            c, bus = h.Controller(directory), FakeBus()
            disconnected = [False]
            original_read, original_update = bus.read, c.update
            def read(*args, **kwargs):
                if disconnected[0]: raise OSError(6, 'Device not configured')
                return original_read(*args, **kwargs)
            bus.read = read
            def update(**values):
                original_update(**values)
                if values.get('recorded_frames'):
                    bus.positions['elbow_flex'] += 2
                    if values['recorded_frames'] == 5: disconnected[0] = True
            with patch.object(h, 'SO100Follower', return_value=SimpleNamespace(bus=bus, is_calibrated=True)), \
                 patch.object(h.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='')), \
                 patch.object(c, 'update', side_effect=update), \
                 patch.object(c.stop, 'wait', return_value=False):
                c.run('record')
            recovered = h.recover_journal(c.snapshot()['recovery_file'])
            self.assertEqual(len(recovered['frames']), 5)
            self.assertEqual([f['raw_position']['elbow_flex'] for f in recovered['frames']], [2000,2002,2004,2006,2008])
            self.assertFalse(recovered['recording_complete'])
            self.assertIsNone(c.last_file)
            self.assertEqual(bus.writes, [])
            self.assertFalse(bus.is_connected)

    def test_save_failure_still_disables_gripper(self):
        with tempfile.TemporaryDirectory() as directory:
            c, bus = h.Controller(directory), FakeBus()
            c.commands.put('open')
            with patch.object(h, 'SO100Follower', return_value=SimpleNamespace(bus=bus, is_calibrated=True)), \
                 patch.object(h.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='')), \
                 patch.object(h.RecordingJournal, 'frame', side_effect=OSError('disk full')), \
                 patch.object(h, 'stop_motor', side_effect=lambda r,m: (bus.write('Torque_Enable',m,0) or True)), patch('builtins.print'):
                c.run('record')
            self.assertTrue(c.snapshot()['torque_off_verified'])
            self.assertEqual(bus.registers['Torque_Enable','gripper'], 0)
            self.assertFalse(any(m in h.BODY for reg,m,v in bus.writes))


if __name__ == '__main__': unittest.main()
