import unittest

from runtime import target_tracking as tt


class TargetTrackingTests(unittest.TestCase):
    def test_initial_lock_initializes_filter_state(self):
        state = tt.reset_tracker()
        new_state, event = tt.update_tracker(
            state,
            [(1.0, 0.1, 0.2, 1.05), (2.0, 0.0, 0.0, 2.0)],
            current_time_ms=1000,
        )

        self.assertEqual(event, "locked")
        self.assertTrue(new_state.locked)
        self.assertEqual(new_state.status, "locked")
        self.assertEqual(new_state.hit_streak, 1)
        self.assertFalse(new_state.stable_latched)
        self.assertEqual(new_state.miss_count, 0)
        self.assertAlmostEqual(new_state.position[0], 1.0, places=6)
        self.assertAlmostEqual(new_state.position[1], 0.1, places=6)
        self.assertAlmostEqual(new_state.position[2], 0.2, places=6)

    def test_tracking_updates_filtered_position_and_velocity(self):
        state, _ = tt.update_tracker(
            tt.reset_tracker(),
            [(1.0, 0.0, 0.0, 1.0)],
            current_time_ms=1000,
        )

        updated, event = tt.update_tracker(
            state,
            [(1.2, 0.0, 0.0, 1.2)],
            current_time_ms=1050,
            gate_m=0.5,
        )

        self.assertEqual(event, "tracked")
        self.assertEqual(updated.status, "tracked")
        self.assertEqual(updated.miss_count, 0)
        self.assertEqual(updated.hit_streak, 2)
        self.assertTrue(updated.stable_latched)
        self.assertIsNotNone(updated.predicted_position)
        self.assertGreater(updated.velocity[0], 0.0)
        self.assertGreater(updated.position[0], 1.0)
        self.assertLess(updated.position[0], 1.2)

    def test_tracker_becomes_stable_after_two_hits_and_tolerates_two_misses(self):
        state = tt.reset_tracker()
        for index, x in enumerate((1.0, 1.1), start=0):
            state, _ = tt.update_tracker(
                state,
                [(x, 0.0, 0.0, x)],
                current_time_ms=1000 + index * 50,
                gate_m=0.5,
                stable_hits=2,
            )

        self.assertTrue(tt.is_tracker_stable(state, min_hits=2, max_miss=2))
        self.assertTrue(state.stable_latched)

        first_predicted, event = tt.update_tracker(
            state,
            [],
            current_time_ms=1100,
            gate_m=0.5,
            max_miss=2,
            stable_hits=2,
        )
        self.assertEqual(event, "predicted")
        self.assertTrue(first_predicted.locked)
        self.assertEqual(first_predicted.miss_count, 1)
        self.assertTrue(tt.is_tracker_stable(first_predicted, min_hits=2, max_miss=2))

        second_predicted, event = tt.update_tracker(
            first_predicted,
            [],
            current_time_ms=1150,
            gate_m=0.5,
            max_miss=2,
            stable_hits=2,
        )
        self.assertEqual(event, "predicted")
        self.assertTrue(second_predicted.locked)
        self.assertEqual(second_predicted.miss_count, 2)
        self.assertTrue(tt.is_tracker_stable(second_predicted, min_hits=2, max_miss=2))

        lost, event = tt.update_tracker(
            second_predicted,
            [],
            current_time_ms=1200,
            gate_m=0.5,
            max_miss=2,
            stable_hits=2,
        )
        self.assertEqual(event, "lost")
        self.assertFalse(lost.locked)
        self.assertFalse(tt.is_tracker_stable(lost, min_hits=2, max_miss=2))

    def test_tracker_reports_lost_after_timeout(self):
        state, _ = tt.update_tracker(
            tt.reset_tracker(),
            [(1.0, 0.0, 0.0, 1.0)],
            current_time_ms=1000,
        )

        lost, event = tt.update_tracker(
            state,
            [],
            current_time_ms=4005,
            timeout_ms=2000,
        )

        self.assertEqual(event, "lost")
        self.assertFalse(lost.locked)
        self.assertEqual(lost.status, "lost")
        self.assertIsNone(lost.position)

    def test_tracker_relock_within_grace_restores_stable_state(self):
        state = tt.reset_tracker()
        state, _ = tt.update_tracker(state, [(1.0, 0.0, 0.0, 1.0)], current_time_ms=1000, stable_hits=2)
        state, _ = tt.update_tracker(state, [(1.1, 0.0, 0.0, 1.1)], current_time_ms=1050, stable_hits=2)
        state, event = tt.update_tracker(state, [], current_time_ms=1200, max_miss=0, stable_hits=2)

        self.assertEqual(event, "lost")
        self.assertFalse(state.locked)
        self.assertIsNotNone(state.last_stable_position)

        relocked, event = tt.update_tracker(
            state,
            [(1.15, 0.02, 0.01, 1.15)],
            current_time_ms=1500,
            stable_hits=2,
            relock_grace_ms=1000,
            relock_distance_m=0.6,
        )

        self.assertEqual(event, "relocked")
        self.assertTrue(relocked.locked)
        self.assertEqual(relocked.status, "relocked")
        self.assertTrue(relocked.stable_latched)
        self.assertEqual(relocked.hit_streak, 2)
        self.assertTrue(tt.is_tracker_stable(relocked, min_hits=2, max_miss=2))

    def test_tracker_relock_outside_grace_resets_warmup(self):
        state = tt.reset_tracker()
        state, _ = tt.update_tracker(state, [(1.0, 0.0, 0.0, 1.0)], current_time_ms=1000, stable_hits=2)
        state, _ = tt.update_tracker(state, [(1.1, 0.0, 0.0, 1.1)], current_time_ms=1050, stable_hits=2)
        state, _ = tt.update_tracker(state, [], current_time_ms=1200, max_miss=0, stable_hits=2)

        relocked, event = tt.update_tracker(
            state,
            [(2.0, 0.0, 0.0, 2.0)],
            current_time_ms=2600,
            stable_hits=2,
            relock_grace_ms=1000,
            relock_distance_m=0.6,
        )

        self.assertEqual(event, "relocked")
        self.assertTrue(relocked.locked)
        self.assertFalse(relocked.stable_latched)
        self.assertEqual(relocked.hit_streak, 1)
        self.assertFalse(tt.is_tracker_stable(relocked, min_hits=2, max_miss=2))

    def test_tracker_state_to_dict_exposes_extended_fields(self):
        state, _ = tt.update_tracker(
            tt.reset_tracker(),
            [(0.5, 0.1, 0.2, 0.6)],
            current_time_ms=1000,
        )
        payload = tt.tracker_state_to_dict(state)

        self.assertIn("predicted_position", payload)
        self.assertIn("velocity", payload)
        self.assertIn("covariance", payload)
        self.assertIn("hit_streak", payload)
        self.assertIn("stable_latched", payload)
        self.assertIn("last_stable_position", payload)
        self.assertIn("last_stable_time_ms", payload)
        self.assertEqual(payload["status"], "locked")


if __name__ == "__main__":
    unittest.main()
