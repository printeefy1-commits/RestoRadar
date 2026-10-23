"""Artifact isolation and cache integrity using local synthetic archives only."""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models


class OptionalPoseArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "models"
        self.root.mkdir()
        self.bundle = Path(self.temp.name) / "bundle"
        self.bundle.mkdir()
        self.bundle_change = patch.object(models, "BUNDLED_DIR", self.bundle)
        self.bundle_change.start()
        self.change = patch.object(models, "MODEL_DIR", self.root)
        self.change.start()
        for name in models.ARTIFACTS:
            (self.root / f"{name}.onnx").write_bytes(f"baseline {name}".encode())
        self.baseline = {
            name: {"url": url, "sha256": models.digest(self.root / f"{name}.onnx")}
            for name, url in models.ARTIFACTS.items()
        }
        self.baseline["detector"].update(
            sourceSha256=models.DETECTOR_SOURCE_SHA256, model="YOLO11_LARGE"
        )
        (self.bundle / "yolo11l.onnx").write_bytes(
            (self.root / "detector.onnx").read_bytes()
        )
        (self.bundle / "detector.json").write_text(
            json.dumps(self.baseline["detector"])
        )
        (self.root / "manifest.json").write_text(json.dumps(self.baseline))
        self.originals = {p.name: p.read_bytes() for p in self.root.iterdir()}

    def tearDown(self):
        self.change.stop()
        self.bundle_change.stop()
        self.temp.cleanup()

    def assert_baseline_untouched(self):
        for name, data in self.originals.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        self.assertEqual(models.manifest(), self.baseline)

    @staticmethod
    def archive(data=b"synthetic medium artifact", multiple=False):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as bundle:
            bundle.writestr("../../outside.txt", "never extract")
            bundle.writestr("nested/model.onnx", data)
            if multiple:
                bundle.writestr("second.onnx", data)
        return stream.getvalue()

    def test_upgrade_replaces_old_detector_and_preserves_both_pose_caches(self):
        (self.root / "pose-medium.onnx").write_bytes(b"optional pose unchanged")
        (self.root / "pose-medium.json").write_text("optional metadata unchanged")
        old = {**self.baseline, "detector": {"url": "old-yolox", "sha256": "old"}}
        (self.root / "manifest.json").write_text(json.dumps(old))
        (self.root / "detector.onnx").write_bytes(b"old tiny detector")
        self.assertIsNone(models.manifest())
        with patch("urllib.request.urlopen") as fetch:
            models.prepare()
            models.prepare()
        fetch.assert_not_called()
        self.assertEqual(models.manifest(), self.baseline)
        self.assertEqual(
            (self.root / "pose.onnx").read_bytes(), self.originals["pose.onnx"]
        )
        self.assertEqual(
            (self.root / "detector.onnx").read_bytes(), self.originals["detector.onnx"]
        )
        self.assertEqual(
            (self.root / "pose-medium.onnx").read_bytes(), b"optional pose unchanged"
        )
        self.assertEqual(
            (self.root / "pose-medium.json").read_text(), "optional metadata unchanged"
        )

    def test_corrupt_bundle_or_wrong_source_fails_without_repinning_cache(self):
        (self.root / "detector.onnx").write_bytes(b"old detector")
        old = (self.root / "detector.onnx").read_bytes()
        (self.bundle / "yolo11l.onnx").write_bytes(b"corrupted large model")
        with self.assertRaisesRegex(ValueError, "checksum"):
            models.prepare()
        self.assertEqual((self.root / "detector.onnx").read_bytes(), old)
        record = self.baseline["detector"] | {"sourceSha256": "b" * 64}
        (self.bundle / "detector.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "provenance"):
            models.prepare()
        self.assertEqual((self.root / "detector.onnx").read_bytes(), old)

    def test_pose_repair_cannot_repin_a_previously_recorded_weight(self):
        (self.root / "pose.onnx").write_bytes(b"damaged cache")
        with patch(
            "urllib.request.urlopen",
            return_value=io.BytesIO(self.archive(b"different pose source")),
        ):
            with self.assertRaisesRegex(ValueError, "recorded digest"):
                models.prepare()
        self.assertEqual(
            json.loads((self.root / "manifest.json").read_text()), self.baseline
        )

    def test_tiny_uses_existing_manifest_without_optional_download(self):
        with patch("urllib.request.urlopen") as fetch:
            self.assertEqual(models.prepare_pose("RTMPOSE_TINY"), self.baseline["pose"])
        fetch.assert_not_called()
        self.assert_baseline_untouched()

    def test_medium_download_is_separate_cached_and_does_not_extract_paths(self):
        with patch(
            "urllib.request.urlopen", return_value=io.BytesIO(self.archive())
        ) as fetch:
            record = models.prepare_pose("RTMPOSE_MEDIUM")
        fetch.assert_called_once_with(
            models.POSE_MODELS["RTMPOSE_MEDIUM"]["url"], timeout=60
        )
        self.assertEqual(
            record["sha256"], models.digest(self.root / "pose-medium.onnx")
        )
        self.assertEqual(models.pose_manifest("RTMPOSE_MEDIUM"), record)
        self.assertFalse((self.root.parent / "outside.txt").exists())
        with patch("urllib.request.urlopen") as fetch:
            self.assertEqual(models.prepare_pose("RTMPOSE_MEDIUM"), record)
        fetch.assert_not_called()
        self.assert_baseline_untouched()

    def test_download_failure_preserves_baseline_and_can_be_retried(self):
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                models.prepare_pose("RTMPOSE_MEDIUM")
        self.assertIsNone(models.pose_manifest("RTMPOSE_MEDIUM"))
        self.assertEqual(set(p.name for p in self.root.iterdir()), set(self.originals))
        self.assert_baseline_untouched()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(self.archive())):
            self.assertIsNotNone(models.prepare_pose("RTMPOSE_MEDIUM"))

    def test_unexpected_archive_cannot_replace_any_artifact(self):
        with patch(
            "urllib.request.urlopen",
            return_value=io.BytesIO(self.archive(multiple=True)),
        ):
            with self.assertRaisesRegex(ValueError, "Unexpected model archive"):
                models.prepare_pose("RTMPOSE_MEDIUM")
        self.assertIsNone(models.pose_manifest("RTMPOSE_MEDIUM"))
        self.assert_baseline_untouched()

    def test_changed_cached_weight_is_not_loaded_or_repinned_to_new_content(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(self.archive())):
            record = models.prepare_pose("RTMPOSE_MEDIUM")
        (self.root / "pose-medium.onnx").write_bytes(b"changed cache")
        self.assertIsNone(models.pose_manifest("RTMPOSE_MEDIUM"))
        with patch(
            "urllib.request.urlopen",
            return_value=io.BytesIO(self.archive(b"changed source")),
        ):
            with self.assertRaisesRegex(ValueError, "recorded digest"):
                models.prepare_pose("RTMPOSE_MEDIUM")
        self.assertEqual(
            json.loads((self.root / "pose-medium.json").read_text()), record
        )
        self.assert_baseline_untouched()
        # A fresh copy matching the recorded digest can repair a damaged cache.
        with patch("urllib.request.urlopen", return_value=io.BytesIO(self.archive())):
            self.assertEqual(models.prepare_pose("RTMPOSE_MEDIUM"), record)

    def test_unknown_model_or_changed_recorded_source_is_rejected_without_fetch(self):
        with patch("urllib.request.urlopen") as fetch:
            with self.assertRaisesRegex(ValueError, "Unsupported pose model"):
                models.prepare_pose("https://example.invalid/model.onnx")
        fetch.assert_not_called()
        (self.root / "pose-medium.json").write_text(
            json.dumps({"url": "https://example.invalid/model.zip", "sha256": "a" * 64})
        )
        with patch("urllib.request.urlopen") as fetch:
            with self.assertRaisesRegex(ValueError, "Invalid recorded"):
                models.prepare_pose("RTMPOSE_MEDIUM")
        fetch.assert_not_called()
        self.assert_baseline_untouched()


if __name__ == "__main__":
    unittest.main()
