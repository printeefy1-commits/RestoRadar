"""Build the fixed YOLO11l artifact; PyTorch is never needed in the runtime image."""

import gc
import importlib.metadata
import json
import shutil
from pathlib import Path
import sys
import time
import urllib.request

from models import DETECTOR_SOURCE_SHA256, DETECTOR_SOURCE_URL, digest


def export(destination):
    import onnx
    import torch
    from ultralytics import YOLO

    for package, expected in (
        ("ultralytics", "8.3.240"),
        ("torch", "2.8.0+cpu"),
        ("onnx", "1.18.0"),
    ):
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Expected {package} {expected}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / "yolo11l.pt"
    if not weights.exists():
        temporary = destination / "weights.tmp"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                    DETECTOR_SOURCE_URL, timeout=60
                ) as src, temporary.open("wb") as out:
                    total = 0
                    while chunk := src.read(1024 * 1024):
                        total += len(chunk)
                        if total > 100 * 1024 * 1024:
                            raise ValueError("Detector download exceeds limit")
                        out.write(chunk)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1)
        if digest(temporary) != DETECTOR_SOURCE_SHA256:
            raise ValueError("YOLO11l source checksum mismatch")
        temporary.replace(weights)
    if digest(weights) != DETECTOR_SOURCE_SHA256:
        raise ValueError("YOLO11l source checksum mismatch")
    model = YOLO(str(weights), task="detect")
    parameters = sum(p.numel() for p in model.model.parameters())
    if (
        not 25_000_000 <= parameters < 26_000_000
        or model.names.get(0) != "person"
        or len(model.names) != 80
    ):
        raise ValueError("Expected the large YOLO11 COCO detector")
    output = Path(
        model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            device="cpu",
            opset=17,
            dynamic=False,
            simplify=False,
            nms=False,
            half=False,
            int8=False,
        )
    )
    del model
    gc.collect()
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    inputs, outputs = graph.graph.input, graph.graph.output

    def shape(value):
        return [dim.dim_value for dim in value.type.tensor_type.shape.dim]

    if (
        len(inputs) != 1
        or shape(inputs[0]) != [1, 3, 640, 640]
        or len(outputs) != 1
        or shape(outputs[0]) != [1, 84, 8400]
    ):
        raise ValueError("Unexpected YOLO11l ONNX interface")
    metadata = {
        "url": DETECTOR_SOURCE_URL,
        "sourceSha256": DETECTOR_SOURCE_SHA256,
        "sha256": digest(output),
        "model": "YOLO11_LARGE",
        "parameters": parameters,
        "exporter": "ultralytics 8.3.240 / torch 2.8.0+cpu / onnx 1.18.0",
        "inputSize": [640, 640],
        "personClassId": 0,
    }
    license_path = importlib.metadata.distribution("ultralytics").locate_file(
        "ultralytics-8.3.240.dist-info/licenses/LICENSE"
    )
    shutil.copyfile(license_path, destination / "ULTRALYTICS-LICENSE")
    (destination / "detector.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    export(Path(sys.argv[1]))
