"""Fixed, build-time YOLO26n-Pose bundle; never fetch browser-selected weights."""

import json
import models

MODEL = "YOLO26_NANO_POSE"
LABEL = "YOLO26n-Pose (COCO ONNX, 640x640)"
SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt?download=1"
SOURCE_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
EXPORTER_VERSION = "8.4.184"


def bundled_record():
    record = json.loads((models.BUNDLED_DIR / "yolo26n-pose.json").read_text())
    if (
        record.get("model") != MODEL
        or record.get("url") != SOURCE_URL
        or record.get("sourceSha256") != SOURCE_SHA256
        or record.get("inputSize") != [640, 640]
        or record.get("outputShape") != [1, 300, 57]
        or record.get("end2end") is not True
    ):
        raise ValueError("Invalid bundled YOLO26n-Pose provenance")
    if models.digest(models.BUNDLED_DIR / "yolo26n-pose.onnx") != record["sha256"]:
        raise ValueError("Bundled YOLO26n-Pose checksum mismatch")
    return record


def artifact_path():
    # The immutable image contains the small artifact already. Loading directly
    # avoids downloads, duplicate volume copies and changing the legacy cache.
    return models.BUNDLED_DIR / "yolo26n-pose.onnx"
