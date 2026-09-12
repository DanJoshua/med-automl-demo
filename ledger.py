"""极简账本：ledger.json 是实验状态的唯一事实源，只允许 driver 通过本模块写入。

记录状态规则（与完整系统一致）：
- crash：无有效分数；
- keep：分数严格优于此前所有 keep 记录（分数越小越好）；
- discard：有效但未更优。
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

LEDGER_NAME = "ledger.json"


def _path(run_dir: Path) -> Path:
    return run_dir / LEDGER_NAME


def load(run_dir: Path) -> list[dict]:
    p = _path(run_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"账本文件 {p} 已损坏（{e}）。请手工修复或删除该文件后重跑。") from e


def _save(run_dir: Path, records: list[dict]) -> None:
    p = _path(run_dir)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def best_kept_record(records: list[dict]) -> dict | None:
    kept = [r for r in records if r.get("status") == "keep" and r.get("score") is not None]
    return min(kept, key=lambda r: r["score"]) if kept else None


def _decide_status(records: list[dict], score: float | None) -> str:
    if score is None or not math.isfinite(score):
        return "crash"
    best = best_kept_record(records)
    return "keep" if best is None or score < best["score"] else "discard"


def add_record(run_dir: Path, record: dict) -> dict:
    records = load(run_dir)
    record = dict(record)
    record.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    record["status"] = _decide_status(records, record.get("score"))
    records.append(record)
    _save(run_dir, records)
    return record


def set_final_score(run_dir: Path, run_id: str, final_test_score: float) -> None:
    """把最优候选的最终留出测试分数写回其记录。"""
    records = load(run_dir)
    for r in records:
        if r["run_id"] == run_id:
            r["final_test_score"] = final_test_score
    _save(run_dir, records)


def summary_view(records: list[dict], max_idea_chars: int = 80) -> str:
    """给 LLM 看的紧凑账本视图（score 为验证集分数）。"""
    if not records:
        return "(空账本：还没有任何候选)"
    lines = ["id    | op            | score      | status  | idea"]
    for r in records:
        score = "crash" if r.get("score") is None else f"{r['score']:.4f}"
        op = r.get("op", "?")
        if r.get("parent_id"):
            op = f"{op}({r['parent_id']})"
        idea = (r.get("idea") or "").replace("\n", " ")[:max_idea_chars]
        line = f"{r['run_id']} | {op:<13} | {score:<10} | {r.get('status', '?'):<7} | {idea}"
        if r.get("status") == "crash" and r.get("error"):
            line += f"  [错误: {str(r['error']).splitlines()[0][:60]}]"
        lines.append(line)
    return "\n".join(lines)
