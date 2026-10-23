"""Fixed build-time YOLO26 pose bundles; never fetch browser-selected weights."""

import json
import models

MODEL = "YOLO26_NANO_POSE"
SMALL_MODEL = "YOLO26_SMALL_POSE"
LABEL = "YOLO26n-Pose (COCO ONNX, 640x640)"
SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt?download=1"
SOURCE_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
EXPORTER_VERSION = "8.4.184"
SPECS = {
    MODEL: {
        "file": "yolo26n-pose",
        "label": LABEL,
        "url": SOURCE_URL,
        "sourceSha256": SOURCE_SHA256,
        "minParameters": 2_000_000,
        "maxParameters": 4_000_000,
        "maxBytes": 20 * 1024 * 1024,
    },
    SMALL_MODEL: {
        "file": "yolo26s-pose",
        "label": "YOLO26s-Pose (COCO ONNX, 640x640)",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt?download=1",
        "sourceSha256": "a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646",
        "minParameters": 9_000_000,
        "maxParameters": 14_000_000,
        "maxBytes": 50 * 1024 * 1024,
    },
}


def spec(name):
    if name not in SPECS:
        raise ValueError("Unsupported YOLO26 pose model")
    return SPECS[name]


def bundled_record(name=MODEL):
    selected = spec(name)
    record = json.loads((models.BUNDLED_DIR / f"{selected['file']}.json").read_text())
    if (
        record.get("model") != name
        or record.get("url") != selected["url"]
        or record.get("sourceSha256") != selected["sourceSha256"]
        or record.get("inputSize") != [640, 640]
        or record.get("outputShape") != [1, 300, 57]
        or record.get("end2end") is not True
    ):
        raise ValueError("Invalid bundled YOLO26 pose provenance")
    if models.digest(artifact_path(name)) != record["sha256"]:
        raise ValueError("Bundled YOLO26 pose checksum mismatch")
    return record


def artifact_path(name=MODEL):
    # The immutable image contains the small artifact already. Loading directly
    # avoids downloads, duplicate volume copies and changing the legacy cache.
    return models.BUNDLED_DIR / f"{spec(name)['file']}.onnx"
