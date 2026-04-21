from __future__ import annotations

import argparse
import json
import inspect
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from utils.mot_io import MOTRow, write_mot


def _create_tracker(track_buffer: int, fps: float):
    tracker_params = {
        "lost_track_buffer": track_buffer,
        "track_buffer": track_buffer,
        "max_age": track_buffer,
        "frame_rate": int(round(fps)) if fps > 0 else 30,
    }

    if hasattr(sv, "ByteTrackTracker"):
        tracker_cls = sv.ByteTrackTracker
    elif hasattr(sv, "ByteTrack"):
        tracker_cls = sv.ByteTrack
    else:
        raise RuntimeError("supervision 中未找到 ByteTrackTracker/ByteTrack，请升级 supervision")

    signature = inspect.signature(tracker_cls)
    config = {name: value for name, value in tracker_params.items() if name in signature.parameters}
    return tracker_cls(**config), config


def _create_trace_annotator(trace_length: int):
    signature = inspect.signature(sv.TraceAnnotator)
    if "trace_length" in signature.parameters:
        return sv.TraceAnnotator(trace_length=trace_length), {"trace_length": trace_length}
    return sv.TraceAnnotator(), {"trace_length": "default"}


def _track(tracker: object, detections: sv.Detections) -> sv.Detections:
    if hasattr(tracker, "update_with_detections"):
        return tracker.update_with_detections(detections)
    if hasattr(tracker, "update"):
        return tracker.update(detections)
    raise RuntimeError("ByteTrack 跟踪器接口不兼容")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ByteTrack + supervision 视频轨迹检测")
    parser.add_argument("--source", required=True, help="输入视频路径")
    parser.add_argument("--output-dir", default="runs/exp", help="输出目录")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO模型")
    parser.add_argument("--device", default="cpu", help="推理设备，如 cpu / 0")
    parser.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--track-buffer", type=int, default=120, help="轨迹丢失缓存帧数（越大越不易中断）")
    parser.add_argument("--trace-length", type=int, default=90, help="轨迹线可视化长度（帧）")
    parser.add_argument("--mock", action="store_true", help="使用内置模拟检测（离线自测）")
    return parser.parse_args()


def run_tracking(
    source: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    conf: float,
    track_buffer: int,
    trace_length: int,
    mock: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / "tracked.mp4"
    mot_output_path = output_dir / "mot_results.txt"
    meta_output_path = output_dir / "run_meta.json"

    model = None if mock else YOLO(model_name)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    track_buffer = max(1, int(track_buffer))
    trace_length = max(1, int(trace_length))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker, tracker_config = _create_tracker(track_buffer=track_buffer, fps=fps)
    trace_annotator, trace_config = _create_trace_annotator(trace_length=trace_length)

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    mot_rows: List[MOTRow] = []
    total_tracks = 0
    start = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        if mock:
            frame_h, frame_w = frame.shape[:2]
            x_center = int((frame_idx * 7) % max(frame_w, 1))
            y_center = int(frame_h * 0.5 + (frame_h * 0.2) * (1 if (frame_idx // 15) % 2 == 0 else -1))
            w_box = max(24, frame_w // 8)
            h_box = max(24, frame_h // 6)
            x1 = max(0, x_center - w_box // 2)
            y1 = max(0, y_center - h_box // 2)
            x2 = min(frame_w - 1, x_center + w_box // 2)
            y2 = min(frame_h - 1, y_center + h_box // 2)
            detections = sv.Detections(
                xyxy=np.array([[float(x1), float(y1), float(x2), float(y2)]], dtype=np.float32),
                confidence=np.array([0.95], dtype=np.float32),
                class_id=np.array([0], dtype=np.int32),
            )
        else:
            result = model.predict(frame, conf=conf, device=device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
        tracked = _track(tracker, detections)

        tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
        confidences = tracked.confidence if tracked.confidence is not None else []

        labels = []
        for index, track_id in enumerate(tracker_ids):
            score = float(confidences[index]) if index < len(confidences) else 1.0
            labels.append(f"id={int(track_id)} conf={score:.2f}")

        annotated = frame.copy()
        if len(tracked) > 0:
            annotated = trace_annotator.annotate(scene=annotated, detections=tracked)
            annotated = box_annotator.annotate(scene=annotated, detections=tracked)
            annotated = label_annotator.annotate(scene=annotated, detections=tracked, labels=labels)

            xyxy = tracked.xyxy
            for index, bbox in enumerate(xyxy):
                track_id = int(tracker_ids[index]) if index < len(tracker_ids) else -1
                confidence = float(confidences[index]) if index < len(confidences) else 1.0
                x1, y1, x2, y2 = map(float, bbox)
                mot_rows.append(
                    MOTRow(
                        frame=frame_idx,
                        track_id=track_id,
                        x=x1,
                        y=y1,
                        w=max(0.0, x2 - x1),
                        h=max(0.0, y2 - y1),
                        confidence=confidence,
                    )
                )

        total_tracks += len(tracked)
        writer.write(annotated)

    cap.release()
    writer.release()

    elapsed = max(time.time() - start, 1e-6)
    write_mot(mot_output_path, mot_rows)

    run_meta = {
        "source": str(source),
        "output_video": str(output_video_path),
        "mot_result": str(mot_output_path),
        "frames": frame_idx,
        "fps_input": fps,
        "avg_tracks_per_frame": round(total_tracks / max(frame_idx, 1), 4),
        "elapsed_sec": round(elapsed, 3),
        "throughput_fps": round(frame_idx / elapsed, 3),
        "model": model_name,
        "device": device,
        "conf": conf,
        "track_buffer": track_buffer,
        "trace_length": trace_length,
        "tracker_config": tracker_config,
        "trace_config": trace_config,
        "mock": mock,
    }

    meta_output_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：输出视频 -> {output_video_path}")
    print(f"完成：MOT结果 -> {mot_output_path}")
    print(f"完成：运行信息 -> {meta_output_path}")


if __name__ == "__main__":
    args = parse_args()
    run_tracking(
        source=Path(args.source),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        device=args.device,
        conf=args.conf,
        track_buffer=args.track_buffer,
        trace_length=args.trace_length,
        mock=args.mock,
    )
