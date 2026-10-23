"""Export the pinned Nano pose model without adding PyTorch to production."""

import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import urllib.request

from models import digest
from yolo26_models import EXPORTER_VERSION, MODEL, SOURCE_SHA256, SOURCE_URL


def export(destination):
    import onnx
    import torch
    from ultralytics import YOLO

    for package, expected in (
        ("ultralytics", EXPORTER_VERSION),
        ("torch", "2.8.0+cpu"),
        ("onnx", "1.18.0"),
    ):
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Expected {package} {expected}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / "yolo26n-pose.pt"
    if not weights.exists():
        temporary = destination / "weights.tmp"
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as src, temporary.open(
            "wb"
        ) as dst:
            total = 0
            while chunk := src.read(1024 * 1024):
                total += len(chunk)
                if total > 20 * 1024 * 1024:
                    raise ValueError("Nano pose download exceeds limit")
                dst.write(chunk)
        if digest(temporary) != SOURCE_SHA256:
            raise ValueError("YOLO26n-Pose source checksum mismatch")
        temporary.replace(weights)
    if digest(weights) != SOURCE_SHA256:
        raise ValueError("YOLO26n-Pose source checksum mismatch")
    model = YOLO(str(weights), task="pose")
    parameters = sum(p.numel() for p in model.model.parameters())
    if (
        not 2_000_000 <= parameters < 4_000_000
        or model.names != {0: "person"}
        or list(model.model.model[-1].kpt_shape) != [17, 3]
    ):
        raise ValueError("Expected the Nano YOLO26 COCO pose model")
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
            end2end=True,
            half=False,
            int8=False,
        )
    )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)

    def shape(value):
        return [dim.dim_value for dim in value.type.tensor_type.shape.dim]

    if (
        len(graph.graph.input) != 1
        or shape(graph.graph.input[0]) != [1, 3, 640, 640]
        or len(graph.graph.output) != 1
        or shape(graph.graph.output[0]) != [1, 300, 57]
    ):
        raise ValueError("Unexpected YOLO26n-Pose ONNX interface")
    record = {
        "model": MODEL,
        "url": SOURCE_URL,
        "sourceSha256": SOURCE_SHA256,
        "sha256": digest(output),
        "parameters": parameters,
        "exporter": f"ultralytics {EXPORTER_VERSION} / torch 2.8.0+cpu / onnx 1.18.0",
        "inputSize": [640, 640],
        "outputShape": [1, 300, 57],
        "end2end": True,
    }
    distribution = importlib.metadata.distribution("ultralytics")
    license_path = next(
        f for f in distribution.files if str(f).endswith("licenses/LICENSE")
    )
    shutil.copyfile(
        distribution.locate_file(license_path), destination / "YOLO26-LICENSE"
    )
    (destination / "yolo26n-pose.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    export(Path(sys.argv[1]))
