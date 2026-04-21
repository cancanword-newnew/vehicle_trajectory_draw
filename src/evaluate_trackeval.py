from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用官方 TrackEval 评估 MOT 结果")
    parser.add_argument("--trackeval-dir", required=True, help="TrackEval 仓库路径")
    parser.add_argument("--gt-folder", required=True, help="GT根目录")
    parser.add_argument("--trackers-folder", required=True, help="Tracker结果根目录")
    parser.add_argument("--benchmark", default="MOT17", help="如 MOT17 / MOT20")
    parser.add_argument("--split", default="train", help="如 train / test")
    parser.add_argument("--tracker-name", required=True, help="待评测跟踪器目录名")
    parser.add_argument("--metrics", default="HOTA CLEAR Identity", help="指标列表")
    parser.add_argument("--use-parallel", default="False", help="是否并行")
    parser.add_argument("--seqmap-folder", default="", help="自定义seqmap目录路径")
    parser.add_argument("--do-preproc", default="False", help="是否执行预处理")
    parser.add_argument("--classes-to-eval", default="pedestrian", help="类别，如 pedestrian")
    return parser.parse_args()


def run_trackeval(
    trackeval_dir: Path,
    gt_folder: Path,
    trackers_folder: Path,
    benchmark: str,
    split: str,
    tracker_name: str,
    metrics: str,
    use_parallel: str,
    seqmap_folder: str,
    do_preproc: str,
    classes_to_eval: str,
) -> int:
    runner = trackeval_dir / "scripts" / "run_mot_challenge.py"
    if not runner.exists():
        raise FileNotFoundError(f"未找到 TrackEval 脚本: {runner}")

    cmd = [
        sys.executable,
        str(runner),
        "--BENCHMARK",
        benchmark,
        "--SPLIT_TO_EVAL",
        split,
        "--GT_FOLDER",
        str(gt_folder),
        "--TRACKERS_FOLDER",
        str(trackers_folder),
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--METRICS",
        *metrics.split(),
        "--USE_PARALLEL",
        use_parallel,
        "--DO_PREPROC",
        do_preproc,
        "--CLASSES_TO_EVAL",
        classes_to_eval,
    ]

    if seqmap_folder:
        cmd.extend(["--SEQMAP_FOLDER", str(Path(seqmap_folder).resolve())])

    log_file = Path("trackeval_command.txt")
    log_file.write_text(" ".join(cmd), encoding="utf-8")

    print("执行命令:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=trackeval_dir, check=False)
    return completed.returncode


if __name__ == "__main__":
    args = parse_args()
    exit_code = run_trackeval(
        trackeval_dir=Path(args.trackeval_dir).resolve(),
        gt_folder=Path(args.gt_folder).resolve(),
        trackers_folder=Path(args.trackers_folder).resolve(),
        benchmark=args.benchmark,
        split=args.split,
        tracker_name=args.tracker_name,
        metrics=args.metrics,
        use_parallel=args.use_parallel,
        seqmap_folder=args.seqmap_folder,
        do_preproc=args.do_preproc,
        classes_to_eval=args.classes_to_eval,
    )
    raise SystemExit(exit_code)
