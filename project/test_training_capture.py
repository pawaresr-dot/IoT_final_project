import unittest
from record_training_capture import validate_status,validate_input_heartbeat,duration_deadline,MOTORS


class CaptureTests(unittest.TestCase):
    def test_unlimited_recording_has_no_three_minute_deadline(self):
        self.assertIsNone(duration_deadline(0,100))
        self.assertEqual(duration_deadline(180,100),280)
        with self.assertRaises(ValueError):duration_deadline(-1,100)

    def test_input_process_failure_does_not_record_follower_hold_as_demo(self):
        validate_input_heartbeat({'event':'sample','time':100},100.2)
        with self.assertRaises(ValueError):validate_input_heartbeat({'event':'stopped','time':100},100.2)
        with self.assertRaises(ValueError):validate_input_heartbeat({'event':'sample','time':100},102)

    def status(self):
        return {'phase':'holding','last_sample':100,
                'readings':{m:{'Status':0,'Torque_Enable':1} for m in MOTORS},
                'command_raw':{m:2000 for m in MOTORS}}

    def test_accepts_fresh_issued_targets(self):validate_status(self.status(),100.1)

    def test_does_not_invent_actions_from_states(self):
        s=self.status();s['command_raw']=None
        with self.assertRaises(ValueError):validate_status(s,100.1)

    def test_stale_alarm_or_controller_error_rejected(self):
        for kind in ('stale','alarm','error','phase'):
            s=self.status()
            if kind=='alarm':s['readings']['elbow_flex']['Status']=8
            if kind=='error':s['calibration_error']='local bound'
            if kind=='phase':s['phase']='paused'
            with self.assertRaises(ValueError):validate_status(s,102 if kind=='stale' else 100.1)


if __name__=='__main__':unittest.main()
