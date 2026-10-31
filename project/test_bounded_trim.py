import unittest
from bounded_trim import BoundedTrim


class TrimTests(unittest.TestCase):
    def test_simulated_constant_load(self):
        trim = BoundedTrim(2200, 1000, 3000)
        actual = 2224
        last = 2200
        for _ in range(40):
            command, reached = trim.step(actual)
            self.assertLessEqual(abs(command - last), 1)
            last = command
            actual = command + 24  # simple load bias, not a servo model
        self.assertTrue(reached)
        self.assertLessEqual(abs(actual - 2200), 6)

    def test_no_motion_saturates(self):
        trim = BoundedTrim(2200, 1000, 3000)
        for _ in range(100):
            command, reached = trim.step(2224)
        self.assertEqual(command, 2168)
        self.assertFalse(reached)

    def test_boundary_rejects(self):
        with self.assertRaises(ValueError):
            BoundedTrim(1000, 1000, 3000).step(1024)

    def test_deadband_does_not_integrate(self):
        trim = BoundedTrim(2200, 1000, 3000)
        self.assertEqual(trim.step(2203), (2200, True))

    def test_streaming_trim_saturates_without_crossing_calibrated_boundary(self):
        trim = BoundedTrim(2918, 732, 2918, saturate_at_limits=True)
        for _ in range(100):
            command, reached = trim.step(2902)
            self.assertEqual(command, 2918)
            self.assertEqual(trim.trim, 0)
            self.assertFalse(reached)


if __name__ == '__main__':
    unittest.main()
