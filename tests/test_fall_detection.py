import unittest

from runtime import fall_detection as fd


SENSITIVITY_MEDIUM = "中等"


class FallDetectionTests(unittest.TestCase):
    def test_update_height_history_appends_normal_sample(self):
        history = []
        sample = (1000, 0.8, 0.1, 1.0)
        updated = fd.update_height_history(history, sample, max_history=5)
        self.assertEqual(updated, [sample])

    def test_update_height_history_smooths_outlier(self):
        history = [
            (0, 0.9, 0.0, 1.0),
            (100, 0.88, 0.02, 1.0),
            (200, 0.87, 0.01, 1.0),
        ]
        updated = fd.update_height_history(history, (300, 5.0, 5.0, 1.0), max_history=5)
        self.assertEqual(len(updated), 4)
        self.assertAlmostEqual(updated[-1][1], 0.88, places=6)
        self.assertAlmostEqual(updated[-1][2], 0.01, places=6)

    def test_detect_fall_returns_false_for_stable_standing(self):
        settings = {"sensitivity_level": SENSITIVITY_MEDIUM}
        stable = [
            (0, 1.0, 0.0, 1.0),
            (500, 1.02, 0.01, 1.0),
            (1000, 0.98, -0.01, 1.0),
        ]
        result = fd.detect_fall(stable, settings, 1000)
        self.assertFalse(result["detected"])
        self.assertEqual(result["strategy"], "new")

    def test_detect_fall_returns_true_for_clear_fall(self):
        settings = {"sensitivity_level": SENSITIVITY_MEDIUM}
        fall = [
            (0, 1.2, 0.0, 1.2),
            (150, 1.0, 0.0, 1.2),
            (300, 0.5, 0.01, 1.2),
            (450, -0.4, 0.02, 1.2),
            (900, -0.35, 0.02, 1.2),
            (1200, -0.3, 0.02, 1.2),
        ]
        result = fd.detect_fall(fall, settings, 1200)
        self.assertTrue(result["detected"])
        self.assertEqual(result["strategy"], "new")
        self.assertIn("metrics", result)

    def test_detect_fall_returns_false_for_slow_squat(self):
        settings = {"sensitivity_level": SENSITIVITY_MEDIUM}
        squat = [
            (0, 1.0, 0.0, 1.0),
            (1000, 0.8, 0.0, 1.0),
            (2000, 0.55, 0.0, 1.0),
            (3000, 0.25, 0.0, 1.0),
        ]
        result = fd.detect_fall(squat, settings, 3000)
        self.assertFalse(result["detected"])
        self.assertEqual(result["strategy"], "new")


if __name__ == "__main__":
    unittest.main()
