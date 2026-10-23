"""CPU-only recorded-video trial, invoked in a bounded child process."""

import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import RTMPose
import supervision as sv

from decisions import (
    MIN_DETECTION_SCORE,
    MIN_KEYPOINT_SCORE,
    SeatingVerifier,
    seated_anchor,
    seating_evidence,
)
from models import MODEL_DIR, manifest, pose_spec, prepare_pose
from focused_detection import (
    FOCUS_MERGE_POLICY,
    MAX_FOCUS_TABLES,
    detect_with_seat_focus,
)
from wrist_trial import WRIST_POLICY, WristTrial, wrist_observation


class CpuPose(RTMPose):
    def __init__(self, path):
        # RTMLib's default constructor creates an unbounded ORT threadpool.
        # Configure the same pinned 17-keypoint model with a bounded CPU session.
        self.model_input_size = (192, 256)
        self.mean = (123.675, 116.28, 103.53)
        self.std = (58.395, 57.12, 57.375)
        self.to_openpose = False
        self.backend = "onnxruntime"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()

        def compatible(shape, expected):
            # MMDeploy may use symbolic batch/keypoint dimensions. Check known
            # metadata here and validate actual SimCC tensors on every inference.
            return len(shape) == len(expected) and all(
                not isinstance(value, int) or value == size
                for value, size in zip(shape, expected)
            )

        if (
            len(inputs) != 1
            or not compatible(inputs[0].shape, [1, 3, 256, 192])
            or inputs[0].type != "tensor(float)"
            or len(outputs) != 2
            or not compatible(outputs[0].shape, [1, 17, 384])
            or not compatible(outputs[1].shape, [1, 17, 512])
            or any(output.type != "tensor(float)" for output in outputs)
        ):
            raise ValueError("Unsupported 17-point pose model interface")

    def inference(self, image):
        outputs = super().inference(image)
        if (
            len(outputs) != 2
            or outputs[0].shape != (1, 17, 384)
            or outputs[1].shape != (1, 17, 512)
            or any(not np.isfinite(value).all() for value in outputs)
        ):
            raise ValueError("Unsupported or non-finite 17-point pose output")
        return outputs


class PersonDetector:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0]
        self.size = tuple(
            v if isinstance(v, int) and v > 0 else 416 for v in self.input.shape[2:]
        )

    def __call__(self, frame):
        height, width = self.size
        ratio = min(height / frame.shape[0], width / frame.shape[1])
        image = np.full((height, width, 3), 114, dtype=np.uint8)
        resized = cv2.resize(
            frame, (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
        )
        image[: resized.shape[0], : resized.shape[1]] = resized
        tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)
        outputs = self.session.run(None, {self.input.name: tensor})
        predictions = outputs[0]
        # The named person-only MMDeploy export returns NMS xyxy+score.
        if predictions.ndim == 3 and predictions.shape[-1] == 5:
            boxes, scores = predictions[0, :, :4] / ratio, predictions[0, :, 4]
        else:
            raise ValueError(
                f"Unsupported detector export: {predictions.shape}; expected person-only NMS output"
            )
        keep = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= 0.1)
        boxes, scores = boxes[keep].copy(), scores[keep].copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame.shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame.shape[0])
        valid = ((boxes[:, 2] - boxes[:, 0]) >= 3) & ((boxes[:, 3] - boxes[:, 1]) >= 3)
        boxes, scores = boxes[valid], scores[valid]
        keep = np.argsort(-scores)[:30]
        return sv.Detections(
            xyxy=boxes[keep],
            confidence=scores[keep],
            class_id=np.zeros(len(keep), dtype=int),
        )


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def analyze(source, config, output, progress):
    started = time.monotonic()
    pose_model = config.get("poseModel", "RTMPOSE_TINY")
    selected_pose = pose_spec(pose_model)
    detection_mode = config.get("detectionMode", "FULL_FRAME")
    if detection_mode not in ("FULL_FRAME", "SEAT_FOCUS"):
        raise ValueError("Unsupported detection mode")
    if (
        detection_mode == "SEAT_FOCUS"
        and len({r["tableId"] for r in config["regions"]}) > MAX_FOCUS_TABLES
    ):
        raise ValueError("Focused detection supports at most four tables per trial")
    artifacts = manifest()
    if not artifacts:
        raise ValueError("Model artifacts are missing or have changed")
    atomic_json(progress, {"progress": 0, "phase": "PREPARING_POSE"})
    pose_record = prepare_pose(pose_model)
    detector = PersonDetector(MODEL_DIR / "detector.onnx")
    pose = CpuPose(MODEL_DIR / f"{selected_pose['file']}.onnx")
    atomic_json(progress, {"progress": 0, "phase": "ANALYZING"})
    tracker = sv.ByteTrack(
        track_activation_threshold=MIN_DETECTION_SCORE,
        frame_rate=2,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
    )
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("Video cannot be decoded")
    table_ids = {r["tableId"] for r in config["regions"]}
    verifier = SeatingVerifier(table_ids)
    wrist_enabled = config.get("wristTrial", False)
    wrist_verifier = WristTrial(table_ids) if wrist_enabled else None
    samples, events = [], []
    first = config["startSeconds"]
    end = first + config["durationSeconds"]
    next_sample = first
    previous_timestamp = None
    last_timestamp = first
    try:
        if first > 1:
            cap.set(cv2.CAP_PROP_POS_MSEC, (first - 0.5) * 1000)
        # Decode sequentially: preserve source presentation time rather than wall-clock time.
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not np.isfinite(seconds) or (
                previous_timestamp is not None and seconds <= previous_timestamp
            ):
                raise ValueError("Unsupported or discontinuous video timestamps")
            previous_timestamp = seconds
            if seconds > end:
                break
            if seconds + 0.001 < next_sample:
                continue
            next_sample = seconds + 0.499
            last_timestamp = seconds
            h, w = frame.shape[:2]
            image_ok = (
                min(h, w) >= 100
                and float(np.std(frame)) > 5
                and 5 < float(np.mean(frame)) < 250
            )
            occupied, people = set(), []
            detected_boxes = 0
            focus_crops = 0
            if image_ok:
                if detection_mode == "SEAT_FOCUS":
                    detections, focus_crops = detect_with_seat_focus(
                        detector, frame, config["regions"]
                    )
                else:
                    detections = detector(frame)
                detected_boxes = len(detections)
                tracked = tracker.update_with_detections(detections)
                if len(tracked):
                    # Never call RTMPose with an empty bbox array: it would infer on the whole frame.
                    points, scores = pose(frame, bboxes=tracked.xyxy)
                    for i, box in enumerate(tracked.xyxy):
                        anchor = seated_anchor(points[i], scores[i])
                        score = float(tracked.confidence[i])
                        table_id, evidence = seating_evidence(
                            anchor, config["regions"], w, h, score
                        )
                        if evidence == "ACCEPTED":
                            occupied.add(table_id)
                        people.append(
                            {
                                "box": [
                                    float(box[0] / w),
                                    float(box[1] / h),
                                    float(box[2] / w),
                                    float(box[3] / h),
                                ],
                                "trackId": (
                                    int(tracked.tracker_id[i])
                                    if tracked.tracker_id is not None
                                    else None
                                ),
                                "posture": (
                                    "SEATED" if anchor is not None else "UNCERTAIN"
                                ),
                                "tableId": table_id,
                                "score": round(score, 4),
                                "evidence": evidence,
                                "seatedAnchor": (
                                    [float(anchor[0] / w), float(anchor[1] / h)]
                                    if anchor is not None
                                    else None
                                ),
                                # Body points only; do not export facial landmarks.
                                "posePoints": [
                                    {
                                        "part": part,
                                        "point": [
                                            float(points[i][index][0] / w),
                                            float(points[i][index][1] / h),
                                        ],
                                        "score": float(scores[i][index]),
                                    }
                                    for index, part in (
                                        (5, "LEFT_SHOULDER"),
                                        (6, "RIGHT_SHOULDER"),
                                        (11, "LEFT_HIP"),
                                        (12, "RIGHT_HIP"),
                                        (13, "LEFT_KNEE"),
                                        (14, "RIGHT_KNEE"),
                                    )
                                ],
                            }
                        )
                        if wrist_enabled:
                            person = people[-1]
                            person["wristEvidence"] = wrist_observation(
                                points[i],
                                scores[i],
                                box,
                                person["trackId"],
                                score,
                                config["regions"],
                                config["tabletopRegions"],
                                w,
                                h,
                            )
                            person["posePoints"].extend(
                                {
                                    "part": part,
                                    "point": [
                                        float(points[i][index][0] / w),
                                        float(points[i][index][1] / h),
                                    ],
                                    "score": float(scores[i][index]),
                                }
                                for index, part in (
                                    (9, "LEFT_WRIST"),
                                    (10, "RIGHT_WRIST"),
                                )
                            )
            states, confirmed = verifier.observe(seconds, occupied)
            events.extend(confirmed)
            samples.append(
                {
                    "seconds": round(seconds, 3),
                    "detectedBoxes": detected_boxes,
                    "focusCrops": focus_crops,
                    "people": people,
                    "tables": states,
                }
            )
            if wrist_verifier:
                samples[-1]["wristTables"] = wrist_verifier.observe(seconds, people)
            atomic_json(
                progress,
                {
                    "progress": min(
                        0.99, max(0, (seconds - first) / config["durationSeconds"])
                    )
                },
            )
    finally:
        cap.release()
    if len(samples) < 2:
        raise ValueError("Selected interval contains too few decodable frames")
    elapsed = time.monotonic() - started
    report = {
        "mode": "RECORDED_VIDEO_DRY_RUN",
        "detectionMode": detection_mode,
        "poseModel": pose_model,
        "sourceId": config["sourceId"],
        "models": {
            "detector": "YOLOX-tiny (HumanArt ONNX)",
            "pose": selected_pose["label"],
            "tracker": "ByteTrack (supervision 0.25.1)",
            "artifacts": {
                "detector": artifacts["detector"]["sha256"],
                "pose": pose_record["sha256"],
            },
        },
        "startSeconds": first,
        "endSeconds": last_timestamp,
        "analyzedSamples": len(samples),
        "processingSeconds": round(elapsed, 3),
        "samplesPerProcessingSecond": round(len(samples) / elapsed, 3),
        "departureEvaluated": False,
        "evidencePolicy": {
            "minDetectionScore": MIN_DETECTION_SCORE,
            "minKeypointScore": MIN_KEYPOINT_SCORE,
            "seatingSeconds": 5,
        },
        "regions": config["regions"],
        "events": events,
        "samples": samples,
    }
    if detection_mode == "SEAT_FOCUS":
        report["focusMergePolicy"] = FOCUS_MERGE_POLICY
    if wrist_enabled:
        report.update(
            wristTrial=True,
            tabletopRegions=config["tabletopRegions"],
            wristPolicy=WRIST_POLICY,
        )
    atomic_json(output, report)


if __name__ == "__main__":
    analyze(
        Path(os.environ["VIDEO_SAMPLE_PATH"]),
        json.loads(Path(sys.argv[1]).read_text()),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
    )
