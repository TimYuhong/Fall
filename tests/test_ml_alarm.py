import unittest

from runtime import ml_alarm as mla


class DummyPrediction:
    def __init__(self, *, available=True, label="non-fall", probability=0.0, inference_id=None, buffered_frames=0, required_frames=100):
        self.available = available
        self.label = label
        self.probability = probability
        self.metadata = {}
        if inference_id is not None:
            self.metadata["inference_id"] = inference_id
        if buffered_frames:
            self.metadata["buffered_frames"] = buffered_frames
        if required_frames:
            self.metadata["required_frames"] = required_frames


class MLAlarmStateTests(unittest.TestCase):
    def test_two_positive_inferences_confirm_alert(self):
        state = mla.build_initial_alarm_state(
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            required_streak=2,
            threshold=0.5,
        )

        first, triggered = mla.update_alarm_state(
            state,
            DummyPrediction(label="fall", probability=0.91, inference_id=1),
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            tracker_ready=True,
            threshold=0.5,
            required_streak=2,
        )
        self.assertFalse(triggered)
        self.assertFalse(first.active)
        self.assertEqual(first.status, "positive_pending")
        self.assertEqual(first.streak, 1)

        second, triggered = mla.update_alarm_state(
            first,
            DummyPrediction(label="fall", probability=0.93, inference_id=2),
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            tracker_ready=True,
            threshold=0.5,
            required_streak=2,
        )
        self.assertTrue(triggered)
        self.assertTrue(second.active)
        self.assertEqual(second.status, "alert")
        self.assertEqual(second.streak, 2)

    def test_duplicate_inference_id_does_not_increment_streak(self):
        state = mla.build_alarm_state(
            active=False,
            status="positive_pending",
            streak=1,
            required_streak=2,
            threshold=0.5,
            last_inference_id=7,
            last_label="fall",
            last_probability=0.82,
            message="ML pending 1/2 p=0.82",
            reason="pending_confirmation",
        )

        updated, triggered = mla.update_alarm_state(
            state,
            DummyPrediction(label="fall", probability=0.82, inference_id=7),
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            tracker_ready=True,
            threshold=0.5,
            required_streak=2,
        )
        self.assertFalse(triggered)
        self.assertEqual(updated.streak, 1)
        self.assertEqual(updated.status, "positive_pending")

    def test_positive_without_tracking_enters_tracking_required(self):
        state = mla.build_initial_alarm_state(
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            required_streak=2,
            threshold=0.5,
        )

        updated, triggered = mla.update_alarm_state(
            state,
            DummyPrediction(label="fall", probability=0.87, inference_id=3),
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            tracker_ready=False,
            threshold=0.5,
            required_streak=2,
        )
        self.assertFalse(triggered)
        self.assertFalse(updated.active)
        self.assertEqual(updated.status, "tracking_required")
        self.assertIn("tracking required", updated.message.lower())

    def test_warming_mismatch_and_error_never_alert(self):
        state = mla.build_initial_alarm_state(
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            required_streak=2,
            threshold=0.5,
        )

        warming, triggered = mla.update_alarm_state(
            state,
            DummyPrediction(available=False, buffered_frames=23, required_frames=100),
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            tracker_ready=True,
            threshold=0.5,
            required_streak=2,
        )
        self.assertFalse(triggered)
        self.assertEqual(warming.status, "warming")

        mismatch, triggered = mla.update_alarm_state(
            warming,
            DummyPrediction(label="fall", probability=0.95, inference_id=4),
            enabled=True,
            model_loaded=True,
            contract_valid=False,
            contract_status="mismatch",
            tracker_ready=True,
            threshold=0.5,
            required_streak=2,
        )
        self.assertFalse(triggered)
        self.assertEqual(mismatch.status, "disabled")
        self.assertEqual(mismatch.message, "ML disabled: cfg mismatch")

        errored, triggered = mla.update_alarm_state(
            mismatch,
            DummyPrediction(label="fall", probability=0.95, inference_id=5),
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            tracker_ready=True,
            threshold=0.5,
            required_streak=2,
            error_message="runtime boom",
        )
        self.assertFalse(triggered)
        self.assertEqual(errored.status, "error")

    def test_lagging_blocks_alert_for_same_inference(self):
        state = mla.build_initial_alarm_state(
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            required_streak=1,
            threshold=0.5,
        )

        lagging, triggered = mla.update_alarm_state(
            state,
            DummyPrediction(label="fall", probability=0.92, inference_id=9),
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            tracker_ready=True,
            threshold=0.5,
            required_streak=1,
            lagging=True,
        )
        self.assertFalse(triggered)
        self.assertEqual(lagging.message, "ML lagging")

        recovered, triggered = mla.update_alarm_state(
            lagging,
            DummyPrediction(label="fall", probability=0.92, inference_id=9),
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            tracker_ready=True,
            threshold=0.5,
            required_streak=1,
        )
        self.assertFalse(triggered)
        self.assertFalse(recovered.active)
        self.assertEqual(recovered.status, "ready")

    def test_threshold_change_re_evaluates_same_inference(self):
        state = mla.build_alarm_state(
            active=True,
            status="alert",
            streak=2,
            required_streak=2,
            threshold=0.5,
            last_inference_id=11,
            last_label="fall",
            last_probability=0.60,
            message="ML FALL p=0.60",
            reason="confirmed_fall",
        )

        updated, triggered = mla.update_alarm_state(
            state,
            DummyPrediction(label="fall", probability=0.60, inference_id=11),
            enabled=True,
            model_loaded=True,
            contract_valid=True,
            contract_status="ready",
            tracker_ready=True,
            threshold=0.65,
            required_streak=2,
        )
        self.assertFalse(triggered)
        self.assertFalse(updated.active)
        self.assertEqual(updated.status, "ready")
        self.assertEqual(updated.threshold, 0.65)

    def test_alert_result_payload_uses_ml_metrics(self):
        state = mla.build_alarm_state(
            active=True,
            status="alert",
            streak=2,
            required_streak=2,
            threshold=0.5,
            last_inference_id=12,
            last_label="fall",
            last_probability=0.88,
            message="ML FALL p=0.88",
            reason="confirmed_fall",
        )
        prediction = DummyPrediction(label="fall", probability=0.88, inference_id=12)
        result = mla.build_alert_result(state, prediction)

        self.assertTrue(result["detected"])
        self.assertEqual(result["source"], "ml")
        self.assertEqual(result["metrics"]["inference_id"], 12)
        self.assertEqual(result["metrics"]["streak"], 2)

    def test_preset_config_matches_expected_defaults(self):
        self.assertEqual(mla.get_preset_config("灵敏"), {"threshold": 0.5, "required_streak": 1})
        self.assertEqual(mla.get_preset_config("中等"), {"threshold": 0.5, "required_streak": 2})
        self.assertEqual(mla.get_preset_config("稳健"), {"threshold": 0.65, "required_streak": 2})


if __name__ == "__main__":
    unittest.main()
