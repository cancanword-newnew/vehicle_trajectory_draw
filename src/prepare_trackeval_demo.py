from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备最小 TrackEval MOTChallenge 目录")
    parser.add_argument("--mot-result", required=True, help="跟踪结果文件 mot_results.txt")
    parser.add_argument("--benchmark", default="MOT17", help="基准名")
    parser.add_argument("--split", default="train", help="split")
    parser.add_argument("--seq-name", default="SYNTH-01", help="序列名")
    parser.add_argument("--tracker-name", default="MyTracker", help="tracker名")
    parser.add_argument("--root", default="data/trackeval", help="输出根目录")
    parser.add_argument("--fps", type=int, default=20, help="序列FPS")
    parser.add_argument("--width", type=int, default=640, help="宽")
    parser.add_argument("--height", type=int, default=360, help="高")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mot_result = Path(args.mot_result)
    lines = [line.strip() for line in mot_result.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("mot_result 为空，无法准备评测数据")

    frames = [int(float(line.split(",")[0])) for line in lines]
    seq_length = max(frames)

    root = Path(args.root)
    benchmark_split = f"{args.benchmark}-{args.split}"

    gt_seq_dir = root / "gt" / benchmark_split / args.seq_name
    gt_dir = gt_seq_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    tracker_data_dir = root / "trackers" / benchmark_split / args.tracker_name / "data"
    tracker_data_dir.mkdir(parents=True, exist_ok=True)

    seqmaps_dir = root / "gt" / "seqmaps"
    seqmaps_dir.mkdir(parents=True, exist_ok=True)

    tracker_out = tracker_data_dir / f"{args.seq_name}.txt"
    tracker_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gt_lines = []
    for line in lines:
        parts = line.split(",")
        frame, track_id, x, y, w, h = parts[:6]
        gt_lines.append(f"{frame},{track_id},{x},{y},{w},{h},1,1,1")

    (gt_dir / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")

    seqinfo = (
        "[Sequence]\n"
        f"name={args.seq_name}\n"
        "imDir=img1\n"
        f"frameRate={args.fps}\n"
        f"seqLength={seq_length}\n"
        f"imWidth={args.width}\n"
        f"imHeight={args.height}\n"
        "imExt=.jpg\n"
    )
    (gt_seq_dir / "seqinfo.ini").write_text(seqinfo, encoding="utf-8")

    seqmap_file = seqmaps_dir / f"{benchmark_split}.txt"
    seqmap_file.write_text(f"name\n{args.seq_name}\n", encoding="utf-8")

    print(f"tracker file: {tracker_out}")
    print(f"gt file: {gt_dir / 'gt.txt'}")
    print(f"seqmap: {seqmap_file}")


if __name__ == "__main__":
    main()
