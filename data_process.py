"""data_process.py

用于离线/手动处理 `data/archive` 下的经验文件。

文件命名约定：
    step_{model_step}_{game_steps}_YYYYMMDD_HHMMSS.pt
例如：
    step_54000_48_20251216_180740.pt

本模块提供：
- 递归收集文件
- 按 model_step 排序
- 按 model_step 聚合 game_steps 的均值
- 将结果写入同目录的 json
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


_FILENAME_RE = re.compile(r"^step_(\d+)_(\d+)_")


@dataclass
class ParsedFile:
    path: str
    model_step: int
    game_steps: int


def iter_archive_pt_files(archive_dir: str) -> Iterable[str]:
    """递归遍历 archive_dir 下所有 .pt 文件，返回文件绝对路径。"""
    for root, _, files in os.walk(archive_dir):
        for name in files:
            if name.endswith(".pt"):
                yield os.path.join(root, name)


def parse_step_filename(filename: str) -> Optional[Tuple[int, int]]:
    """从文件名解析 (model_step, game_steps)。解析失败返回 None。"""
    base = os.path.basename(filename)
    m = _FILENAME_RE.match(base)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def collect_parsed_files(archive_dir: str) -> List[ParsedFile]:
    """收集并解析 archive_dir 下的 .pt 文件（解析失败的会被跳过）。"""
    parsed: List[ParsedFile] = []
    for path in iter_archive_pt_files(archive_dir):
        info = parse_step_filename(path)
        if info is None:
            continue
        model_step, game_steps = info
        parsed.append(ParsedFile(path=path, model_step=model_step, game_steps=game_steps))

    parsed.sort(key=lambda x: x.model_step)
    return parsed


def aggregate_avg_game_steps_by_model_step(files: List[ParsedFile]) -> Dict[int, float]:
    """按 model_step 聚合 game_steps 平均值。

    返回：
        { model_step: avg_game_steps }
    """
    sums: Dict[int, int] = {}
    counts: Dict[int, int] = {}

    for f in files:
        sums[f.model_step] = sums.get(f.model_step, 0) + int(f.game_steps)
        counts[f.model_step] = counts.get(f.model_step, 0) + 1

    out: Dict[int, float] = {}
    for step in sorted(sums.keys()):
        out[step] = sums[step] / max(counts.get(step, 1), 1)
    return out


def save_avg_game_steps_json(archive_dir: str, out_filename: str = "avg_game_steps_by_model_step.json") -> str:
    """扫描 archive_dir，聚合并把结果写入 archive_dir 下的 json。

    说明：
    - json 的 key 会是字符串（JSON 标准），但含义是 model_step。

    返回：
        输出文件绝对路径
    """
    files = collect_parsed_files(archive_dir)
    agg = aggregate_avg_game_steps_by_model_step(files)

    out_path = os.path.join(archive_dir, out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in agg.items()}, f, ensure_ascii=False, indent=2)

    return out_path


if __name__ == "__main__":
    # 默认处理整个 data/archive
    root = os.path.join(os.path.dirname(__file__), "data", "archive")
    out = save_avg_game_steps_json(root)
    print(f"Saved: {out}")
