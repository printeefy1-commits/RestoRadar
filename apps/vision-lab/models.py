"""Fetch named upstream ONNX artifacts; never extract ZIP paths or accept URLs."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
ARTIFACTS = {
    "detector": BASE + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
    "pose": BASE + "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip",
}
POSE_MODELS = {
    "RTMPOSE_TINY": {
        "file": "pose",
        "label": "RTMPose-t (body7 ONNX)",
        "url": ARTIFACTS["pose"],
    },
    # RTMLib 0.0.13 Body.MODE['balanced']; same 17-point, 192x256 interface.
    "RTMPOSE_MEDIUM": {
        "file": "pose-medium",
        "label": "RTMPose-m (body7 ONNX)",
        "url": BASE
        + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    },
}


def pose_spec(name):
    if name not in POSE_MODELS:
        raise ValueError("Unsupported pose model")
    return POSE_MODELS[name]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def manifest():
    try:
        data = json.loads((MODEL_DIR / "manifest.json").read_text())
        for name, url in ARTIFACTS.items():
            if (
                data[name]["url"] != url
                or digest(MODEL_DIR / f"{name}.onnx") != data[name]["sha256"]
            ):
                return None
        return data
    except (OSError, KeyError, ValueError, TypeError):
        return None


def download_artifact(name, url, expected_sha256=None):
    # Arguments come only from the fixed registries in this module.
    with tempfile.TemporaryDirectory(dir=MODEL_DIR) as temp:
        archive = Path(temp) / "model.zip"
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            archive.open("wb") as out,
        ):
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 100 * 1024 * 1024:
                    raise ValueError("Model download exceeds limit")
                out.write(chunk)
        with zipfile.ZipFile(archive) as bundle:
            candidates = [
                info for info in bundle.infolist() if info.filename.endswith(".onnx")
            ]
            if len(candidates) != 1 or candidates[0].file_size > 200 * 1024 * 1024:
                raise ValueError("Unexpected model archive")
            target = Path(temp) / "model.onnx"
            with bundle.open(candidates[0]) as src, target.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            record = {"url": url, "sha256": digest(target)}
            if expected_sha256 is not None and record["sha256"] != expected_sha256:
                raise ValueError("Downloaded artifact differs from its recorded digest")
            target.replace(MODEL_DIR / f"{name}.onnx")
    return record


def prepare():
    if manifest():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    records = {name: download_artifact(name, url) for name, url in ARTIFACTS.items()}
    # HTTPS authenticates the source. These are recorded digests, not upstream signatures.
    tmp = MODEL_DIR / "manifest.tmp"
    tmp.write_text(json.dumps(records, indent=2))
    tmp.replace(MODEL_DIR / "manifest.json")


def pose_manifest(name):
    spec = pose_spec(name)
    if name == "RTMPOSE_TINY":
        baseline = manifest()
        return baseline["pose"] if baseline else None
    try:
        data = json.loads((MODEL_DIR / f"{spec['file']}.json").read_text())
        if (
            data["url"] == spec["url"]
            and digest(MODEL_DIR / f"{spec['file']}.onnx") == data["sha256"]
        ):
            return data
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return None


def prepare_pose(name):
    spec = pose_spec(name)
    record = pose_manifest(name)
    if record:
        return record
    if name == "RTMPOSE_TINY":
        raise ValueError("Baseline pose artifact is missing or has changed")
    # The optional model has its own file and manifest. Its failure must never
    # replace baseline artifacts or change the baseline service readiness.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    expected_sha256 = None
    record_path = MODEL_DIR / f"{spec['file']}.json"
    if record_path.exists():
        saved = json.loads(record_path.read_text())
        expected_sha256 = saved["sha256"]
        if saved["url"] != spec["url"] or not (
            isinstance(expected_sha256, str)
            and len(expected_sha256) == 64
            and all(c in "0123456789abcdef" for c in expected_sha256)
        ):
            raise ValueError("Invalid recorded optional pose artifact")
    record = download_artifact(spec["file"], spec["url"], expected_sha256)
    tmp = MODEL_DIR / f"{spec['file']}.tmp"
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(MODEL_DIR / f"{spec['file']}.json")
    return record


if __name__ == "__main__":
    prepare()
    print(json.dumps(manifest(), indent=2))
