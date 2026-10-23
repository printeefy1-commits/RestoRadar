"""Synthetic graphs exercise the real CPU runtime, not model accuracy on people."""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/restoradar-test-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from fastapi.testclient import TestClient

import analyze
from decisions import SeatingVerifier, mapped_table, seated_anchor, seating_evidence
import models
import service
import supervision as sv
from focused_detection import detect_with_seat_focus, seat_crops


def constant_graph(path, shape, outputs):
    nodes, definitions = [], []
    for name, value in outputs.items():
        nodes.append(
            helper.make_node(
                "Constant", [], [name], value=numpy_helper.from_array(value)
            )
        )
        definitions.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
        )
    graph = helper.make_graph(
        nodes,
        "synthetic-test-only",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        definitions,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))


class DecisionTests(unittest.TestCase):
    def test_diagnostics_distinguish_posture_mapping_and_detection_score(self):
        region = {"tableId": "table", "x": 0, "y": 0, "width": 0.5, "height": 0.5}
        self.assertEqual(
            seating_evidence(None, [region], 100, 100, 0.9), (None, "POSE_UNCERTAIN")
        )
        self.assertEqual(
            seating_evidence([75, 75], [region], 100, 100, 0.9), (None, "UNMAPPED")
        )
        self.assertEqual(
            seating_evidence(
                [25, 25], [region, {**region, "tableId": "other"}], 100, 100, 0.9
            ),
            (None, "AMBIGUOUS_REGIONS"),
        )
        # Matching two seats of the same table is still an unambiguous table mapping.
        self.assertEqual(
            seating_evidence([25, 25], [region, region], 100, 100, 0.35),
            ("table", "ACCEPTED"),
        )
        self.assertEqual(
            seating_evidence([25, 25], [region], 100, 100, 0.1659),
            ("table", "LOW_DETECTION_SCORE"),
        )

    def test_five_seconds_and_unknown_never_departure(self):
        verifier = SeatingVerifier(["table"])
        events = []
        for step in range(11):
            states, result = verifier.observe(step / 2, {"table"})
            events.extend(result)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["observedSeconds"], 0)
        self.assertEqual(events[0]["confirmedSeconds"], 5)
        for step in range(11, 90):
            states, result = verifier.observe(step / 2, set())
            self.assertEqual(states["table"], "UNKNOWN")
            self.assertEqual(result, [])

    def test_missing_evidence_or_timestamp_gap_cannot_confirm(self):
        for missing in (True, False):
            verifier = SeatingVerifier(["table"])
            for step in range(10):
                verifier.observe(step / 2, {"table"})
            if missing:
                verifier.observe(5, set())
                timestamp = 5.5
            else:
                timestamp = 6
            _, events = verifier.observe(timestamp, {"table"})
            self.assertEqual(events, [])
        with self.assertRaises(ValueError):
            verifier.observe(timestamp, {"table"})

    def test_posture_requires_reliable_bend_and_unambiguous_mapping(self):
        points = np.zeros((17, 2))
        points[5], points[11], points[13] = [10, 0], [10, 10], [20, 10]
        scores = np.ones(17)
        self.assertIsNotNone(seated_anchor(points, scores))
        scores[:] = 0.1
        self.assertIsNone(seated_anchor(points, scores))
        points[13] = [10, 20]
        self.assertIsNone(seated_anchor(points, np.ones(17)))
        regions = [
            {"tableId": t, "x": 0, "y": 0, "width": 1, "height": 1} for t in ("a", "b")
        ]
        self.assertIsNone(mapped_table([10, 10], regions, 100, 100))


class FocusedDetectionTests(unittest.TestCase):
    def test_crop_coordinates_and_duplicate_views_are_merged_before_tracking(self):
        frame = np.zeros((1000, 1000, 3), np.uint8)
        regions = [
            {"tableId": "table", "x": 0.4, "y": 0.4, "width": 0.1, "height": 0.2}
        ]
        shapes = []

        def detector(image):
            shapes.append(image.shape[:2])
            if image.shape[:2] == (1000, 1000):
                boxes, scores = [[400, 400, 500, 600]], [0.4]
            else:
                boxes, scores = [[50, 150, 150, 350], [0, 20, 30, 80]], [0.9, 0.8]
            return sv.Detections(
                xyxy=np.array(boxes, np.float32),
                confidence=np.array(scores, np.float32),
                class_id=np.zeros(len(boxes), int),
            )

        detected, crops = detect_with_seat_focus(detector, frame, regions)
        self.assertEqual(crops, 1)
        self.assertEqual(shapes[0], (1000, 1000))
        np.testing.assert_allclose(shapes[1], (400, 200), atol=1)
        np.testing.assert_allclose(
            detected.xyxy, [[400, 400, 500, 600], [350, 270, 380, 330]]
        )
        np.testing.assert_allclose(detected.confidence, [0.9, 0.8])

    def test_bounds_empty_detections_and_cpu_limits(self):
        regions = [
            {"tableId": "table", "x": 0.95, "y": 0.9, "width": 0.05, "height": 0.1}
        ]
        for x1, y1, x2, y2 in seat_crops(regions, 640, 480):
            self.assertTrue(0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480)
        detected, _ = detect_with_seat_focus(
            lambda _: sv.Detections.empty(), np.zeros((480, 640, 3), np.uint8), regions
        )
        self.assertEqual(len(detected), 0)
        self.assertEqual(
            seat_crops(
                [{**regions[0], "x": 0, "y": 0, "width": 1, "height": 1}], 640, 480
            ),
            [],
        )
        self.assertEqual(
            len(seat_crops(regions + [{**regions[0], "tableId": "other"}], 640, 480)), 1
        )
        excessive = [{**regions[0], "tableId": str(i)} for i in range(5)]
        with self.assertRaisesRegex(ValueError, "four tables"):
            seat_crops(excessive, 640, 480)

    def test_request_validation_preserves_default_mode_and_limits_focused_trials(self):
        region = {
            "id": str(uuid4()),
            "tableId": str(uuid4()),
            "x": 0.1,
            "y": 0.2,
            "width": 0.2,
            "height": 0.3,
        }
        trial = {
            "sourceId": "a" * 64,
            "startSeconds": 0,
            "durationSeconds": 10,
            "regions": [region],
        }
        self.assertEqual(service.Trial(**trial).detectionMode, "FULL_FRAME")
        self.assertEqual(
            service.Trial(**{**trial, "detectionMode": "SEAT_FOCUS"}).detectionMode,
            "SEAT_FOCUS",
        )
        with self.assertRaises(ValueError):
            service.Trial(**{**trial, "detectionMode": "untrusted"})
        with self.assertRaises(ValueError):
            service.Trial(
                **{
                    **trial,
                    "detectionMode": "SEAT_FOCUS",
                    "regions": [
                        {**region, "id": str(uuid4()), "tableId": str(uuid4())}
                        for _ in range(5)
                    ],
                }
            )


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "models"
        self.model_dir.mkdir()
        self.source = self.root / "synthetic.avi"
        writer = cv2.VideoWriter(
            str(self.source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240)
        )
        self.assertTrue(writer.isOpened())
        for i in range(81):
            image = np.full((240, 320, 3), 80, dtype=np.uint8)
            image[30:200, 30:150] = (40, 190, 140)
            image[20:30, i : i + 20] = 240
            writer.write(image)
        writer.release()
        constant_graph(
            self.model_dir / "detector.onnx",
            [1, 3, 416, 416],
            {"dets": np.array([[[40, 30, 250, 300, 0.95]]], dtype=np.float32)},
        )
        x, y = np.zeros((1, 17, 384), np.float32), np.zeros((1, 17, 512), np.float32)
        for index in range(17):
            px, py = (96, 140)
            if index in (5, 6):
                py = 70
            if index in (13, 14):
                px = 150
            x[0, index, px * 2] = 1
            y[0, index, py * 2] = 1
        constant_graph(
            self.model_dir / "pose.onnx", [1, 3, 256, 192], {"simcc_x": x, "simcc_y": y}
        )
        records = {
            name: {"url": url, "sha256": models.digest(self.model_dir / f"{name}.onnx")}
            for name, url in models.ARTIFACTS.items()
        }
        (self.model_dir / "manifest.json").write_text(json.dumps(records))
        self.patches = [
            patch.object(models, "MODEL_DIR", self.model_dir),
            patch.object(analyze, "MODEL_DIR", self.model_dir),
            patch.object(service, "SOURCE", self.source),
            patch.object(service, "sample_cache", None),
            patch.dict(
                os.environ,
                {
                    "MODEL_DIR": str(self.model_dir),
                    "VIDEO_SAMPLE_PATH": str(self.source),
                    "MPLCONFIGDIR": "/tmp/restoradar-test-matplotlib",
                },
            ),
        ]
        for change in self.patches:
            change.start()
        self.config = {
            "sourceId": models.digest(self.source),
            "startSeconds": 0,
            "durationSeconds": 8,
            "regions": [
                {
                    "id": str(uuid4()),
                    "tableId": str(uuid4()),
                    "x": 0,
                    "y": 0,
                    "width": 1,
                    "height": 1,
                }
            ],
        }

    def tearDown(self):
        for change in reversed(self.patches):
            change.stop()
        with service.lock:
            for job in service.jobs.values():
                if job["process"].poll() is None:
                    job["process"].kill()
                    job["process"].wait()
                job["temp"].cleanup()
            service.jobs.clear()
            service.active = None
        self.temp.cleanup()

    def test_actual_onnx_and_bytetrack_pipeline_preserves_video_time(self):
        output = self.root / "result.json"
        analyze.analyze(self.source, self.config, output, self.root / "progress.json")
        report = json.loads(output.read_text())
        self.assertEqual(report["mode"], "RECORDED_VIDEO_DRY_RUN")
        self.assertNotIn("focusMergePolicy", report)
        self.assertFalse(report["departureEvaluated"])
        self.assertEqual(len(report["events"]), 1)
        self.assertEqual(report["events"][0]["observedSeconds"], 0)
        self.assertAlmostEqual(report["events"][0]["confirmedSeconds"], 5, places=2)
        self.assertGreaterEqual(report["analyzedSamples"], 16)
        self.assertEqual(report["evidencePolicy"]["minDetectionScore"], 0.35)
        for sample in report["samples"]:
            self.assertEqual(sample["detectedBoxes"], 1)
            for person in sample["people"]:
                self.assertEqual(person["evidence"], "ACCEPTED")
                self.assertEqual(len(person["seatedAnchor"]), 2)
                self.assertEqual(
                    {p["part"] for p in person["posePoints"]},
                    {
                        "LEFT_SHOULDER",
                        "RIGHT_SHOULDER",
                        "LEFT_HIP",
                        "RIGHT_HIP",
                        "LEFT_KNEE",
                        "RIGHT_KNEE",
                    },
                )
        self.assertEqual(
            len({p["trackId"] for s in report["samples"] for p in s["people"]}), 1
        )

    def test_black_frames_stay_unknown_even_with_constant_detection_graph(self):
        source = self.root / "black.avi"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240)
        )
        for _ in range(61):
            writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()
        output = self.root / "result.json"
        analyze.analyze(source, self.config, output, self.root / "progress.json")
        report = json.loads(output.read_text())
        self.assertEqual(report["events"], [])
        self.assertTrue(
            all(set(s["tables"].values()) == {"UNKNOWN"} for s in report["samples"])
        )
        self.assertTrue(all(s["detectedBoxes"] == 0 for s in report["samples"]))

    def test_focused_mode_runs_real_onnx_crops_and_records_mode(self):
        config = {
            **self.config,
            "detectionMode": "SEAT_FOCUS",
            "regions": [
                {
                    **self.config["regions"][0],
                    "x": 0.2,
                    "y": 0.3,
                    "width": 0.2,
                    "height": 0.3,
                }
            ],
        }
        output = self.root / "focused.json"
        analyze.analyze(self.source, config, output, self.root / "progress.json")
        report = json.loads(output.read_text())
        self.assertEqual(report["detectionMode"], "SEAT_FOCUS")
        self.assertEqual(report["focusMergePolicy"], "ALIGNED_CONTAINMENT_V1")
        self.assertTrue(all(s["focusCrops"] == 1 for s in report["samples"]))
        self.assertTrue(all(s["detectedBoxes"] >= 1 for s in report["samples"]))
        self.assertEqual(report["evidencePolicy"]["minDetectionScore"], 0.35)
        self.assertEqual(report["evidencePolicy"]["minKeypointScore"], 0.4)
        self.assertEqual(report["evidencePolicy"]["seatingSeconds"], 5)

    def test_low_detection_score_does_not_confirm_even_when_pose_and_mapping_match(
        self,
    ):
        # Exercise actual ORT and ByteTrack, including a previously activated track.
        real_detector = analyze.PersonDetector

        class FadingDetector:
            def __init__(self, path):
                self.detector = real_detector(path)
                self.calls = 0

            def __call__(self, frame):
                detections = self.detector(frame)
                self.calls += 1
                if self.calls > 2:
                    detections.confidence[:] = 0.1659
                return detections

        output = self.root / "low-score.json"
        with patch.object(analyze, "PersonDetector", FadingDetector):
            analyze.analyze(
                self.source, self.config, output, self.root / "progress.json"
            )
        report = json.loads(output.read_text())
        rejected = [
            p
            for s in report["samples"]
            for p in s["people"]
            if p["evidence"] == "LOW_DETECTION_SCORE"
        ]
        self.assertTrue(rejected)
        self.assertTrue(
            all(p["posture"] == "SEATED" and p["tableId"] for p in rejected)
        )
        self.assertEqual(report["events"], [])

    def test_changed_model_is_rejected(self):
        with (self.model_dir / "detector.onnx").open("ab") as out:
            out.write(b"changed")
        self.assertIsNone(models.manifest())
        with self.assertRaises(ValueError):
            analyze.analyze(
                self.source, self.config, self.root / "out", self.root / "progress"
            )

    def test_worker_range_tenant_isolation_and_child_process(self):
        service.model_state["models"] = "READY"
        tenant, other = str(uuid4()), str(uuid4())
        with TestClient(service.app) as client, patch.object(service, "prepare_models"):
            # The startup thread may already have run; the model manifest is local and valid.
            self.assertEqual(
                client.get("/status").json()["source"]["id"], self.config["sourceId"]
            )
            response = client.get("/video", headers={"Range": "bytes=0-10"})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, self.source.read_bytes()[:11])
            invalid = {**self.config, "sourceId": "0" * 64}
            self.assertEqual(
                client.post(f"/restaurants/{tenant}/jobs", json=invalid).status_code,
                409,
            )
            response = client.post(f"/restaurants/{tenant}/jobs", json=self.config)
            self.assertEqual(response.status_code, 202, response.text)
            job_id = response.json()["id"]
            self.assertEqual(
                client.get(f"/restaurants/{other}/jobs/{job_id}").status_code, 404
            )
            self.assertIsNone(client.get(f"/restaurants/{other}/jobs/latest").json())
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                job = client.get(f"/restaurants/{tenant}/jobs/{job_id}").json()
                if job["status"] != "RUNNING":
                    break
                time.sleep(0.1)
            self.assertEqual(job["status"], "COMPLETED", job)
            self.assertEqual(len(job["report"]["events"]), 1)

    def test_downloader_does_not_extract_archive_paths(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("../../outside.txt", "never extract")
            bundle.writestr("nested/end2end.onnx", b"synthetic artifact")
        (self.model_dir / "manifest.json").unlink()
        with patch(
            "urllib.request.urlopen",
            side_effect=lambda *a, **k: io.BytesIO(archive.getvalue()),
        ):
            models.prepare()
        self.assertFalse((self.root / "outside.txt").exists())
        self.assertEqual(
            (self.model_dir / "pose.onnx").read_bytes(), b"synthetic artifact"
        )
        self.assertIsNotNone(models.manifest())


if __name__ == "__main__":
    unittest.main()
