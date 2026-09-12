"""候选筛查：在固定评估表面上评估 BASE_PARAMS + K 个随机配置，取最优分数。

由 driver 作为子进程调用（隔离候选代码的崩溃与超时）：

    python screening.py <candidate_dir> [--k 4] [--seed 42]

无论成功与否，最后一行 stdout 都会输出 `RESULT_JSON: {...}`。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _PrepareBlocker:
    """技术执行「禁止 import prepare」契约：候选代码中的该导入直接失败。

    评估面按路径加载、不经 sys.modules，因此本拦截不影响 screening 自身。
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "prepare" or fullname.startswith("prepare."):
            raise ImportError("候选代码禁止导入 prepare（评估表面不可接触）")
        return None


def _load_train(candidate_dir: Path):
    sys.meta_path.insert(0, _PrepareBlocker())
    return _load_module("train", candidate_dir / "train.py")


def sample_params(space: dict, rng: random.Random) -> dict:
    params = {}
    for name, spec in (space or {}).items():
        t = spec.get("type")
        if t == "choice":
            params[name] = rng.choice(list(spec["values"]))
        elif t == "int":
            lo, hi = int(spec["low"]), int(spec["high"])
            if spec.get("log"):
                params[name] = max(lo, min(hi, int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))))
            else:
                params[name] = rng.randint(lo, hi)
        elif t == "float":
            lo, hi = float(spec["low"]), float(spec["high"])
            if spec.get("log"):
                params[name] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
            else:
                params[name] = rng.uniform(lo, hi)
    return params


def run_screening(candidate_dir: Path, k: int, seed: int) -> dict:
    prepare = _load_module("prepare", candidate_dir / "prepare.py")
    train = _load_train(candidate_dir)
    make_model = train.make_model
    base = dict(getattr(train, "BASE_PARAMS", {}) or {})
    space = getattr(train, "SEARCH_SPACE", {}) or {}

    rng = random.Random(seed)
    trials = []
    for label in ["base"] + [f"random_{i}" for i in range(k)]:
        try:
            params = dict(base) if label == "base" else {**base, **sample_params(space, rng)}
            score = float(prepare.evaluate_config(make_model, params))
            trials.append({"label": label, "params": params, "ok": True, "score": score})
        except Exception as e:  # 单个配置（含其参数采样）失败不拖垮整个候选
            trials.append({"label": label, "params": None, "ok": False, "error": f"{type(e).__name__}: {e}"})

    ok = [t for t in trials if t["ok"] and math.isfinite(t["score"])]
    best = min(ok, key=lambda t: t["score"]) if ok else None
    first_error = next((t["error"] for t in trials if not t["ok"]), None)
    return {
        "best_score": best["score"] if best else None,
        "best_params": best["params"] if best else None,
        "trials": trials,
        "error": None if best else first_error,
    }


def run_final(candidate_dir: Path, params: dict) -> dict:
    """最终留出测试评估：train+val 重训，测试集只此一次评分。"""
    prepare = _load_module("prepare", candidate_dir / "prepare.py")
    train = _load_train(candidate_dir)
    score = float(prepare.final_evaluate(train.make_model, dict(params)))
    return {"final_score": score}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_dir")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final", action="store_true", help="最终留出测试评估（搜索结束后仅一次）")
    ap.add_argument("--params-json", help="--final 时使用的参数（JSON 字符串）")
    args = ap.parse_args()
    try:
        if args.final:
            result = run_final(Path(args.candidate_dir), json.loads(args.params_json) if args.params_json else {})
        else:
            result = run_screening(Path(args.candidate_dir), args.k, args.seed)
    except Exception as e:  # import 失败等候选级崩溃
        result = {"best_score": None, "best_params": None, "trials": [], "final_score": None, "error": f"{type(e).__name__}: {e}"}
    print("RESULT_JSON: " + json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
