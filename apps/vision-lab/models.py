"""Fetch only the two named upstream ONNX artifacts; never extract ZIP paths."""

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


def prepare():
    if manifest():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, url in ARTIFACTS.items():
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
                    info
                    for info in bundle.infolist()
                    if info.filename.endswith(".onnx")
                ]
                if len(candidates) != 1 or candidates[0].file_size > 200 * 1024 * 1024:
                    raise ValueError("Unexpected model archive")
                target = Path(temp) / "model.onnx"
                with bundle.open(candidates[0]) as src, target.open("wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                records[name] = {"url": url, "sha256": digest(target)}
                target.replace(MODEL_DIR / f"{name}.onnx")
    # HTTPS authenticates the source. These are recorded digests, not upstream signatures.
    tmp = MODEL_DIR / "manifest.tmp"
    tmp.write_text(json.dumps(records, indent=2))
    tmp.replace(MODEL_DIR / "manifest.json")


if __name__ == "__main__":
    prepare()
    print(json.dumps(manifest(), indent=2))
