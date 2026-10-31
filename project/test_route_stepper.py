import unittest
from route_stepper import RouteStepper, make_transfer_route, make_source_route, make_return_route


class RouteTests(unittest.TestCase):
    def test_stream_reverses_without_chasing_obsolete_loaded_goal(self):
        stepper=RouteStepper({'elbow_flex':2549}, {'elbow_flex':(729,2921)},max_step=4,independent=True)
        stepper.elbow.trim=13
        command,_=stepper.step({'elbow_flex':2389},{'elbow_flex':2532})
        self.assertLess(command['elbow_flex'],2532)
        self.assertLessEqual(abs(command['elbow_flex']-2532),30)

    def test_stream_hold_at_actual_cancels_old_trim(self):
        stepper=RouteStepper({'elbow_flex':2549}, {'elbow_flex':(729,2921)},max_step=4,independent=True)
        stepper.elbow.trim=13
        command,reached=stepper.step({'elbow_flex':2532},{'elbow_flex':2532})
        self.assertEqual(command['elbow_flex'],2532)
        self.assertTrue(reached)

    def test_stream_retreated_target_does_not_overshoot_to_old_nominal(self):
        stepper=RouteStepper({'elbow_flex':2549}, {'elbow_flex':(729,2921)},max_step=4,independent=True)
        command,_=stepper.step({'elbow_flex':2535},{'elbow_flex':2532})
        self.assertEqual(command['elbow_flex'],2535)

    def test_stream_profile_moves_multiple_ready_joints_together(self):
        current={'elbow_flex':2000,'shoulder_pan':2000}
        stepper=RouteStepper(current,{m:(1000,3000) for m in current},max_step=4,independent=True)
        command,reached=stepper.step(dict.fromkeys(current,2100),current)
        self.assertEqual(command,dict.fromkeys(current,2004))
        self.assertFalse(reached)

    def test_stream_profile_does_not_block_ready_joint_on_other_joint(self):
        current={'elbow_flex':2000,'shoulder_pan':2000}
        stepper=RouteStepper(current,{m:(1000,3000) for m in current},max_step=4,independent=True)
        command,_=stepper.step(dict.fromkeys(current,2100),dict(current,elbow_flex=1989))
        self.assertEqual(stepper.nominal['elbow_flex'],2000)
        self.assertEqual(stepper.nominal['shoulder_pan'],2004)
        self.assertTrue(all(abs(command[m]-dict(current,elbow_flex=1989)[m])<=30 for m in current))

    def test_stream_profile_still_caps_catchup_when_stuck(self):
        current={'elbow_flex':2000,'shoulder_pan':2000}
        stepper=RouteStepper(current,{m:(1000,3000) for m in current},max_step=4,independent=True)
        for _ in range(100):
            command,reached=stepper.step(dict.fromkeys(current,2200),current)
            self.assertFalse(reached)
            self.assertTrue(all(abs(command[m]-current[m])<=30 for m in current))

    def test_shoulder_small_settling_error_does_not_deadlock(self):
        stepper = RouteStepper({'shoulder_lift': 2500, 'elbow_flex': 2500},
                               {'shoulder_lift': (1500, 3900), 'elbow_flex': (1000, 3000)})
        command, reached = stepper.step({'shoulder_lift': 2502, 'elbow_flex': 2500},
                                       {'shoulder_lift': 2493, 'elbow_flex': 2500})
        self.assertEqual(stepper.nominal['shoulder_lift'], 2502)
        self.assertTrue(reached)
        self.assertLessEqual(abs(command['shoulder_lift'] - 2493), 30)

    def test_shoulder_and_elbow_load_bias(self):
        current = {'elbow_flex': 2000, 'shoulder_lift': 2000}
        stepper = RouteStepper(current, {m: (1000, 3000) for m in current})
        actual = dict(current)
        for target in ({m: 1900 for m in current}, current):
            for _ in range(500):
                command, reached = stepper.step(target, actual)
                self.assertTrue(all(abs(command[m]-actual[m]) <= 30 for m in command))
                self.assertTrue(all(abs(t.trim) <= 32 for t in stepper.trimmers.values()))
                actual = {m: command[m] + 24 for m in command}
                if reached: break
            self.assertTrue(reached)

    def test_return_route_lifts_before_yaw(self):
        poses = {name: {'raw_position': {'shoulder_pan': 2000, 'elbow_flex': 2000}}
                 for name in ('retreat', 'lift', 'approach', 'grasp')}
        route = make_return_route(poses, ('shoulder_pan', 'elbow_flex'))
        self.assertEqual([name for name, _ in route],
                         ['grasp','approach','lift','outside_transfer_height','retreat'])

    def test_source_route_retraces_descent_without_gripper_commands(self):
        motors = ('shoulder_pan', 'elbow_flex')
        poses = {name: {'raw_position': {'shoulder_pan': 2000, 'elbow_flex': 2000, 'gripper': 2400}}
                 for name in ('retreat', 'lift', 'approach', 'grasp')}
        route = make_source_route(poses, motors)
        self.assertEqual([name for name, _ in route],
                         ['retreat','outside_transfer_height','lift','approach','grasp',
                          'approach','lift','outside_transfer_height','retreat'])
        self.assertTrue(all('gripper' not in target for _, target in route))

    def test_transfer_rotation_keeps_other_joints_fixed(self):
        motors = ('shoulder_pan', 'elbow_flex')
        poses = {'retreat': {'raw_position': {'shoulder_pan': 2400, 'elbow_flex': 2700}},
                 'lift': {'raw_position': {'shoulder_pan': 1900, 'elbow_flex': 2600}}}
        route = make_transfer_route(poses, motors)
        self.assertEqual(len(route), 5)
        self.assertEqual(route[1][1]['elbow_flex'], route[2][1]['elbow_flex'])
        self.assertEqual(route[1][1]['shoulder_pan'], route[0][1]['shoulder_pan'])
        self.assertEqual(poses['lift']['raw_position']['shoulder_pan'], 1900)

    def test_loaded_elbow_reaches_both_waypoints(self):
        current = {'elbow_flex': 2000, 'wrist_flex': 2000}
        stepper = RouteStepper(current, {m: (1000, 3000) for m in current})
        actual = dict(current)
        for target in ({'elbow_flex': 2076, 'wrist_flex': 1900}, current):
            reached = False
            for _ in range(500):
                command, reached = stepper.step(target, actual)
                actual = dict(command)
                actual['elbow_flex'] += 24  # artificial constant load bias
                if reached:
                    break
            self.assertTrue(reached)

    def test_stuck_joint_does_not_get_large_catchup_command(self):
        stepper = RouteStepper({'elbow_flex': 2000}, {'elbow_flex': (1000, 3000)})
        commands = []
        for _ in range(100):
            command, reached = stepper.step({'elbow_flex': 2100}, {'elbow_flex': 2000})
            commands.append(command['elbow_flex'])
            self.assertLessEqual(abs(command['elbow_flex'] - 2000), 30)
            self.assertLessEqual(abs(stepper.elbow.trim), 32)
            self.assertFalse(reached)
        self.assertEqual(commands[-1], commands[-10])

    def test_outside_target_cannot_cross_limit(self):
        stepper = RouteStepper({'elbow_flex': 2996}, {'elbow_flex': (1000, 3000)})
        with self.assertRaises(ValueError):
            stepper.step({'elbow_flex': 3100}, {'elbow_flex': 2996})


if __name__ == '__main__':
    unittest.main()
