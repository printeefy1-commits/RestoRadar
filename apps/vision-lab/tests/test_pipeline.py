"""Synthetic graphs exercise the real CPU runtime, not model accuracy on people."""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
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
import yolo26_models
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
    def test_wrist_trial_requires_valid_separate_tabletops_of_seated_tables(self):
        seat = {
            "id": str(uuid4()),
            "tableId": str(uuid4()),
            "x": 0.1,
            "y": 0.2,
            "width": 0.2,
            "height": 0.2,
        }
        trial = {
            "sourceId": "a" * 64,
            "startSeconds": 0,
            "durationSeconds": 10,
            "regions": [seat],
        }
        top = {**seat, "id": str(uuid4())}
        self.assertFalse(service.Trial(**trial).wristTrial)
        self.assertTrue(
            service.Trial(
                **{**trial, "wristTrial": True, "tabletopRegions": [top]}
            ).wristTrial
        )
        for extra in (
            {"wristTrial": True},
            {"wristTrial": "true", "tabletopRegions": [top]},
            {"tabletopRegions": [top]},
            {"wristTrial": True, "tabletopRegions": [seat]},
            {"wristTrial": True, "tabletopRegions": [{**top, "tableId": str(uuid4())}]},
            {"wristTrial": True, "tabletopRegions": [{**top, "x": 0.95}]},
            {
                "wristTrial": True,
                "tabletopRegions": [{**top, "id": str(uuid4())} for _ in range(41)],
            },
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                service.Trial(**{**trial, **extra})

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
        self.assertEqual(service.Trial(**trial).poseModel, "RTMPOSE_TINY")
        self.assertEqual(
            service.Trial(**{**trial, "poseModel": "YOLO26_NANO_POSE"}).poseModel,
            "YOLO26_NANO_POSE",
        )
        self.assertEqual(
            service.Trial(**{**trial, "poseModel": "RTMPOSE_MEDIUM"}).poseModel,
            "RTMPOSE_MEDIUM",
        )
        for invalid in ("unknown", "https://example.invalid/model.onnx"):
            with self.assertRaises(ValueError):
                service.Trial(**{**trial, "poseModel": invalid})
        with self.assertRaises(ValueError):
            service.Trial(**{**trial, "poseUrl": "https://example.invalid/model.onnx"})
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
        detections = np.zeros((1, 84, 8400), np.float32)
        detections[0, :5, 0] = [222, 334, 324, 416, 0.95]
        constant_graph(
            self.model_dir / "detector.onnx", [1, 3, 640, 640], {"output0": detections}
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
        self.bundle_dir = self.root / "bundle"
        self.bundle_dir.mkdir()
        (self.bundle_dir / "yolo11l.onnx").write_bytes(
            (self.model_dir / "detector.onnx").read_bytes()
        )
        records["detector"].update(
            sourceSha256=models.DETECTOR_SOURCE_SHA256, model="YOLO11_LARGE"
        )
        (self.bundle_dir / "detector.json").write_text(json.dumps(records["detector"]))
        (self.model_dir / "manifest.json").write_text(json.dumps(records))
        self.patches = [
            patch.object(models, "MODEL_DIR", self.model_dir),
            patch.object(models, "BUNDLED_DIR", self.bundle_dir),
            patch.object(analyze, "MODEL_DIR", self.model_dir),
            patch.object(service, "SOURCE", self.source),
            patch.object(service, "sample_cache", None),
            patch.dict(
                os.environ,
                {
                    "MODEL_DIR": str(self.model_dir),
                    "DETECTOR_BUNDLE_DIR": str(self.bundle_dir),
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

    def medium_model(self, score=1):
        model = onnx.load(str(self.model_dir / "pose.onnx"))
        model.graph.name = "synthetic-medium-test-only"
        for node in model.graph.node:
            for attribute in node.attribute:
                if attribute.name == "value":
                    value = numpy_helper.to_array(attribute.t).copy() * score
                    attribute.t.CopyFrom(numpy_helper.from_array(value))
        target = self.model_dir / "pose-medium.onnx"
        onnx.save(model, str(target))
        record = {
            "url": models.POSE_MODELS["RTMPOSE_MEDIUM"]["url"],
            "sha256": models.digest(target),
        }
        (self.model_dir / "pose-medium.json").write_text(json.dumps(record))
        return record

    def nano_model(self, keypoint_score=1):
        raw = np.zeros((1, 300, 57), np.float32)
        raw[0, 0, :6] = [60, 126, 384, 542, 0.95, 0]
        points = np.full((17, 3), [222, 334, keypoint_score], np.float32)
        points[5:7, 1] = 180
        points[13:15, 0] = 340
        raw[0, 0, 6:] = points.flatten()
        target = self.bundle_dir / "yolo26n-pose.onnx"
        constant_graph(target, [1, 3, 640, 640], {"output0": raw})
        record = {
            "model": yolo26_models.MODEL,
            "url": yolo26_models.SOURCE_URL,
            "sourceSha256": yolo26_models.SOURCE_SHA256,
            "sha256": models.digest(target),
            "inputSize": [640, 640],
            "outputShape": [1, 300, 57],
            "end2end": True,
        }
        (self.bundle_dir / "yolo26n-pose.json").write_text(json.dumps(record))
        return record

    def test_nano_one_pass_pipeline_preserves_evidence_and_wrist_rules_in_both_modes(
        self,
    ):
        original = (self.model_dir / "manifest.json").read_bytes()
        for mode in ("FULL_FRAME", "SEAT_FOCUS"):
            for score in (1, 0.1):
                with self.subTest(mode=mode, score=score):
                    record = self.nano_model(score)
                    output = self.root / "nano.json"
                    config = {
                        **self.config,
                        "poseModel": "YOLO26_NANO_POSE",
                        "detectionMode": mode,
                        "wristTrial": True,
                        "tabletopRegions": [
                            {**self.config["regions"][0], "id": str(uuid4())}
                        ],
                    }
                    with (
                        patch.object(
                            analyze,
                            "CpuPose",
                            side_effect=AssertionError("No per-person pose"),
                        ),
                        patch.object(
                            analyze,
                            "prepare_pose",
                            side_effect=AssertionError("No RTMPose download"),
                        ),
                    ):
                        analyze.analyze(
                            self.source, config, output, self.root / "progress"
                        )
                    report = json.loads(output.read_text())
                    self.assertEqual(report["poseModel"], "YOLO26_NANO_POSE")
                    self.assertEqual(report["detectorModel"], "YOLO26_NANO_POSE")
                    self.assertEqual(
                        report["models"]["artifacts"],
                        {"pose": record["sha256"], "detector": record["sha256"]},
                    )
                    self.assertEqual(
                        report["evidencePolicy"],
                        {
                            "minDetectionScore": 0.35,
                            "minKeypointScore": 0.4,
                            "seatingSeconds": 5,
                        },
                    )
                    self.assertEqual(
                        report["wristPolicy"]["version"], "WRIST_SEAT_DWELL_V3"
                    )
                    self.assertEqual(len(report["events"]), int(score == 1))
                    if score == 1:
                        self.assertEqual(report["events"][0]["confirmedSeconds"], 5)
                        self.assertIn(
                            "PROBABLE", report["samples"][-1]["wristTables"].values()
                        )
                    else:
                        self.assertTrue(
                            all(
                                p["evidence"] == "POSE_UNCERTAIN"
                                for s in report["samples"]
                                for p in s["people"]
                            )
                        )
                        self.assertNotIn(
                            "PROBABLE", report["samples"][-1]["wristTables"].values()
                        )
                    self.assertTrue(
                        all(s["detectedBoxes"] == 1 for s in report["samples"])
                    )
                    self.assertEqual(
                        (self.model_dir / "manifest.json").read_bytes(), original
                    )

    def test_nano_corrupt_or_wrong_bundle_fails_without_fallback(self):
        self.nano_model()
        target = self.bundle_dir / "yolo26n-pose.onnx"
        target.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            analyze.analyze(
                self.source,
                {**self.config, "poseModel": "YOLO26_NANO_POSE"},
                self.root / "out",
                self.root / "progress",
            )
        record = self.nano_model()
        record["sourceSha256"] = "a" * 64
        (self.bundle_dir / "yolo26n-pose.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "provenance"):
            yolo26_models.bundled_record()

    def test_worker_runs_nano_in_real_child_process(self):
        record = self.nano_model()
        service.model_state["models"] = "READY"
        with patch.object(service, "prepare_models"), TestClient(service.app) as client:
            tenant = str(uuid4())
            response = client.post(
                f"/restaurants/{tenant}/jobs",
                json={**self.config, "poseModel": "YOLO26_NANO_POSE"},
            )
            self.assertEqual(response.status_code, 202, response.text)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                job = client.get(
                    f"/restaurants/{tenant}/jobs/{response.json()['id']}"
                ).json()
                if job["status"] != "RUNNING":
                    break
                time.sleep(0.1)
            self.assertEqual(job["status"], "COMPLETED", job)
            self.assertEqual(job["report"]["detectorModel"], "YOLO26_NANO_POSE")
            self.assertEqual(
                job["report"]["models"]["artifacts"]["pose"], record["sha256"]
            )

    def test_actual_onnx_and_bytetrack_pipeline_preserves_video_time(self):
        output = self.root / "result.json"
        analyze.analyze(self.source, self.config, output, self.root / "progress.json")
        report = json.loads(output.read_text())
        self.assertEqual(report["mode"], "RECORDED_VIDEO_DRY_RUN")
        self.assertEqual(report["poseModel"], "RTMPOSE_TINY")
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

    def test_duplicate_torso_is_filtered_before_actual_bytetrack_and_pose(self):
        raw = np.zeros((1, 84, 8400), np.float32)
        raw[0, :5, 0] = [222, 334, 324, 416, 0.95]
        raw[0, :5, 1] = [222, 266, 324, 280, 0.99]
        target = self.model_dir / "detector.onnx"
        constant_graph(target, [1, 3, 640, 640], {"output0": raw})
        # Register the newly generated synthetic graph, never real model weights.
        records = json.loads((self.model_dir / "manifest.json").read_text())
        records["detector"]["sha256"] = models.digest(target)
        (self.model_dir / "manifest.json").write_text(json.dumps(records))
        (self.bundle_dir / "detector.json").write_text(json.dumps(records["detector"]))
        (self.bundle_dir / "yolo11l.onnx").write_bytes(target.read_bytes())
        output = self.root / "deduplicated.json"
        analyze.analyze(self.source, self.config, output, self.root / "progress.json")
        report = json.loads(output.read_text())
        self.assertEqual(report["personMergePolicy"], "STRICT_UPPER_BODY_ALIGNMENT_V1")
        self.assertTrue(
            all(
                s["detectedBoxes"] == 1 and len(s["people"]) == 1
                for s in report["samples"]
            )
        )
        self.assertEqual(
            len({p["trackId"] for s in report["samples"] for p in s["people"]}), 1
        )
        self.assertEqual(len(report["events"]), 1)
        for sample in report["samples"]:
            person = sample["people"][0]
            self.assertAlmostEqual(person["score"], 0.95, places=5)
            self.assertEqual(person["evidence"], "ACCEPTED")

    def test_selected_medium_uses_one_cpu_pose_session_and_preserves_evidence_rules(
        self,
    ):
        baseline = (self.model_dir / "manifest.json").read_bytes()
        for mode in ("FULL_FRAME", "SEAT_FOCUS"):
            for score in (1, 0.1):
                with self.subTest(mode=mode, score=score):
                    record = self.medium_model(score)
                    config = {
                        **self.config,
                        "poseModel": "RTMPOSE_MEDIUM",
                        "detectionMode": mode,
                    }
                    output = self.root / "medium.json"
                    with (
                        patch.object(
                            analyze, "CpuPose", wraps=analyze.CpuPose
                        ) as loader,
                        patch("urllib.request.urlopen") as fetch,
                    ):
                        analyze.analyze(
                            self.source, config, output, self.root / "progress.json"
                        )
                    loader.assert_called_once_with(self.model_dir / "pose-medium.onnx")
                    fetch.assert_not_called()
                    report = json.loads(output.read_text())
                    self.assertEqual(report["poseModel"], "RTMPOSE_MEDIUM")
                    self.assertEqual(report["models"]["pose"], "RTMPose-m (body7 ONNX)")
                    self.assertEqual(
                        report["models"]["artifacts"]["pose"], record["sha256"]
                    )
                    self.assertNotEqual(
                        record["sha256"], models.manifest()["pose"]["sha256"]
                    )
                    self.assertEqual(
                        report["evidencePolicy"],
                        {
                            "minDetectionScore": 0.35,
                            "minKeypointScore": 0.4,
                            "seatingSeconds": 5,
                        },
                    )
                    self.assertFalse(report["departureEvaluated"])
                    people = [p for s in report["samples"] for p in s["people"]]
                    self.assertTrue(people)
                    if score == 1:
                        self.assertEqual(len(report["events"]), 1)
                        self.assertEqual(report["events"][0]["confirmedSeconds"], 5)
                    else:
                        self.assertEqual(report["events"], [])
                        self.assertTrue(
                            all(p["evidence"] == "POSE_UNCERTAIN" for p in people)
                        )
                        self.assertTrue(
                            all(
                                set(s["tables"].values()) == {"UNKNOWN"}
                                for s in report["samples"]
                            )
                        )
        self.assertEqual((self.model_dir / "manifest.json").read_bytes(), baseline)

    def test_pose_interface_and_non_finite_outputs_are_rejected(self):
        path = self.root / "invalid-pose.onnx"
        for shape, points in (([1, 3, 384, 288], 17), ([1, 3, 256, 192], 18)):
            constant_graph(
                path,
                shape,
                {
                    "simcc_x": np.zeros((1, points, 384), np.float32),
                    "simcc_y": np.zeros((1, points, 512), np.float32),
                },
            )
            with self.assertRaisesRegex(ValueError, "pose model interface"):
                analyze.CpuPose(path)
        constant_graph(
            path,
            [1, 3, 256, 192],
            {
                "simcc_x": np.full((1, 17, 384), np.nan, np.float32),
                "simcc_y": np.zeros((1, 17, 512), np.float32),
            },
        )
        pose = analyze.CpuPose(path)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pose(np.zeros((240, 320, 3), np.uint8), bboxes=[[0, 0, 320, 240]])

    def test_optional_preparation_failure_leaves_baseline_ready_and_runnable(self):
        service.model_state["models"] = "READY"
        progress = self.root / "progress.json"
        with patch.object(analyze, "prepare_pose", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                analyze.analyze(
                    self.source,
                    {**self.config, "poseModel": "RTMPOSE_MEDIUM"},
                    self.root / "result.json",
                    progress,
                )
        process = Mock()
        process.wait.return_value = 1
        job = {
            "id": str(uuid4()),
            "status": "RUNNING",
            "progress": 0,
            "path": self.root,
            "process": process,
        }
        service.active = job
        self.assertIn("جارٍ تجهيز", service.public_job(job)["message"])
        with patch("builtins.print"):
            service.monitor(job, self.config["sourceId"])
        self.assertEqual(job["status"], "FAILED")
        self.assertIn("النموذج الخفيف الحالي", job["message"])
        self.assertNotIn("offline", job["message"])
        self.assertIsNone(service.active)
        self.assertEqual(service.model_state["models"], "READY")
        output = self.root / "baseline.json"
        analyze.analyze(self.source, self.config, output, progress)
        report = json.loads(output.read_text())
        self.assertEqual(report["poseModel"], "RTMPOSE_TINY")
        self.assertEqual(len(report["events"]), 1)

    def test_wrist_probability_never_promotes_uncertain_pose_or_changes_baseline(self):
        # Hide only knees in a real ONNX graph: wrists and torso remain reliable.
        path = self.model_dir / "pose.onnx"
        graph = onnx.load(str(path))
        for node in graph.graph.node:
            for attr in node.attribute:
                if attr.name == "value":
                    array = numpy_helper.to_array(attr.t).copy()
                    array[:, [13, 14], :] *= 0.1
                    attr.t.CopyFrom(numpy_helper.from_array(array))
        onnx.save(graph, str(path))
        manifest_path = self.model_dir / "manifest.json"
        record = json.loads(manifest_path.read_text())
        record["pose"]["sha256"] = models.digest(path)
        manifest_path.write_text(json.dumps(record))
        baseline_path, wrist_path = (
            self.root / "baseline.json",
            self.root / "wrists.json",
        )
        analyze.analyze(
            self.source, self.config, baseline_path, self.root / "progress.json"
        )
        config = {
            **self.config,
            "wristTrial": True,
            "tabletopRegions": [{**self.config["regions"][0], "id": str(uuid4())}],
        }
        analyze.analyze(self.source, config, wrist_path, self.root / "progress.json")
        baseline, report = json.loads(baseline_path.read_text()), json.loads(
            wrist_path.read_text()
        )
        self.assertNotIn("wristTrial", baseline)
        self.assertEqual(report["events"], [])
        self.assertEqual(baseline["events"], [])
        self.assertTrue(report["wristTrial"])
        self.assertEqual(report["wristPolicy"]["version"], "WRIST_SEAT_DWELL_V3")
        probable = [
            s for s in report["samples"] if "PROBABLE" in s["wristTables"].values()
        ]
        self.assertTrue(probable)
        self.assertEqual(probable[0]["seconds"], 5)
        for sample, original in zip(report["samples"], baseline["samples"]):
            self.assertEqual(sample["tables"], original["tables"])
            self.assertEqual(set(sample["tables"].values()), {"UNKNOWN"})
            for person in sample["people"]:
                self.assertEqual(person["evidence"], "POSE_UNCERTAIN")
                self.assertEqual(len(person["posePoints"]), 8)
                person.pop("wristEvidence")
                person["posePoints"] = [
                    p for p in person["posePoints"] if "WRIST" not in p["part"]
                ]
            sample.pop("wristTables")
            self.assertEqual(sample, original)

    def test_score_dip_preserves_only_wrist_evidence_in_actual_runtime(self):
        real_detector = analyze.PersonDetector

        class BriefScoreDip:
            def __init__(self, path):
                self.detector = real_detector(path)
                self.calls = 0

            def __call__(self, frame):
                self.calls += 1
                detected = self.detector(frame)
                if self.calls == 13:  # 6 seconds, after the first five-second dwell.
                    detected.confidence[:] = 0.3039
                return detected

        baseline_path, wrist_path = (
            self.root / "baseline-dip.json",
            self.root / "wrist-dip.json",
        )
        config = {
            **self.config,
            "wristTrial": True,
            "tabletopRegions": [{**self.config["regions"][0], "id": str(uuid4())}],
        }
        with patch.object(analyze, "PersonDetector", BriefScoreDip):
            analyze.analyze(
                self.source, self.config, baseline_path, self.root / "progress.json"
            )
            analyze.analyze(
                self.source, config, wrist_path, self.root / "progress.json"
            )
        baseline, report = json.loads(baseline_path.read_text()), json.loads(
            wrist_path.read_text()
        )
        self.assertEqual(report["events"], baseline["events"])
        sample = min(report["samples"], key=lambda s: abs(s["seconds"] - 6))
        person = sample["people"][0]
        self.assertEqual(person["evidence"], "LOW_DETECTION_SCORE")
        self.assertEqual(person["wristEvidence"]["state"], "PROBABLE")
        self.assertTrue(person["wristEvidence"]["scoreContinuation"])
        self.assertEqual(person["wristEvidence"]["lowScoreSeconds"], 0.5)
        self.assertEqual(report["wristPolicy"]["startDetectionScore"], 0.35)
        self.assertEqual(report["wristPolicy"]["continueDetectionScore"], 0.30)
        for current, original in zip(report["samples"], baseline["samples"]):
            self.assertEqual(current["tables"], original["tables"])
            for p in current["people"]:
                p.pop("wristEvidence")
                p["posePoints"] = [
                    point for point in p["posePoints"] if "WRIST" not in point["part"]
                ]
            current.pop("wristTables")
            self.assertEqual(current, original)

    def test_black_frames_stay_unknown_even_with_constant_detection_graph(self):
        source = self.root / "black.avi"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240)
        )
        for _ in range(61):
            writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()
        output = self.root / "result.json"
        config = {
            **self.config,
            "wristTrial": True,
            "tabletopRegions": [{**self.config["regions"][0], "id": str(uuid4())}],
        }
        analyze.analyze(source, config, output, self.root / "progress.json")
        report = json.loads(output.read_text())
        self.assertEqual(report["events"], [])
        self.assertTrue(
            all(set(s["tables"].values()) == {"UNKNOWN"} for s in report["samples"])
        )
        self.assertTrue(all(s["detectedBoxes"] == 0 for s in report["samples"]))
        self.assertTrue(
            all(
                set(s["wristTables"].values()) == {"UNKNOWN"} for s in report["samples"]
            )
        )

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
            self.assertEqual(job["report"]["detectorModel"], "YOLO11_LARGE")
            self.assertEqual(
                job["report"]["personMergePolicy"], "STRICT_UPPER_BODY_ALIGNMENT_V1"
            )
            self.assertEqual(
                job["report"]["models"]["detector"], "YOLO11l (COCO ONNX, 640x640)"
            )

    def test_worker_runs_selected_medium_in_child_process(self):
        record = self.medium_model()
        service.model_state["models"] = "READY"
        tenant = str(uuid4())
        with patch.object(service, "prepare_models"), TestClient(service.app) as client:
            response = client.post(
                f"/restaurants/{tenant}/jobs",
                json={
                    **self.config,
                    "poseModel": "RTMPOSE_MEDIUM",
                    "wristTrial": True,
                    "tabletopRegions": [
                        {**self.config["regions"][0], "id": str(uuid4())}
                    ],
                },
            )
            self.assertEqual(response.status_code, 202, response.text)
            job_id = response.json()["id"]
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                job = client.get(f"/restaurants/{tenant}/jobs/{job_id}").json()
                if job["status"] != "RUNNING":
                    break
                time.sleep(0.1)
            self.assertEqual(job["status"], "COMPLETED", job)
            self.assertEqual(job["report"]["poseModel"], "RTMPOSE_MEDIUM")
            self.assertTrue(job["report"]["wristTrial"])
            self.assertTrue(
                any(
                    "PROBABLE" in s["wristTables"].values()
                    for s in job["report"]["samples"]
                )
            )
            self.assertEqual(
                job["report"]["models"]["artifacts"]["pose"], record["sha256"]
            )
            self.assertEqual(job["report"]["events"][0]["confirmedSeconds"], 5)
            self.assertEqual(client.get("/status").json()["models"], "READY")

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
