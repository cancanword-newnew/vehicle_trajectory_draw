from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class MOTRow:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0
    world_x: float = -1.0
    world_y: float = -1.0
    world_z: float = -1.0

    def to_mot_line(self) -> str:
        return (
            f"{self.frame},{self.track_id},"
            f"{self.x:.3f},{self.y:.3f},{self.w:.3f},{self.h:.3f},"
            f"{self.confidence:.5f},{self.world_x:.1f},{self.world_y:.1f},{self.world_z:.1f}"
        )


def write_mot(path: Path, rows: Iterable[MOTRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(row.to_mot_line() + "\n")


def read_mot(path: Path) -> List[MOTRow]:
    rows: List[MOTRow] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                raise ValueError(f"Line {line_number} in {path} has fewer than 10 fields")

            rows.append(
                MOTRow(
                    frame=int(float(parts[0])),
                    track_id=int(float(parts[1])),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    w=float(parts[4]),
                    h=float(parts[5]),
                    confidence=float(parts[6]),
                    world_x=float(parts[7]),
                    world_y=float(parts[8]),
                    world_z=float(parts[9]),
                )
            )
    return rows
