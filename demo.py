"""医疗表格 AutoML 搜索 demo —— driver 主循环。

每轮：LLM 提案（fresh/improve + idea + train.py）→ 落盘候选目录 →
子进程筛查（BASE_PARAMS + K 个随机配置，真实运行评分，验证集）→
把最优参数写回候选的 BASE_PARAMS → 记入 ledger.json（keep/discard/crash）。
全部轮次结束后，对最优候选做一次且仅一次留出测试评估。

用法：
    python demo.py --rounds 8                          # 默认乳腺癌数据集
    python demo.py --data 医院数据.csv --target 结局列   # 自己的医疗表格数据
    python demo.py --mock --rounds 4                   # 不耗 API 的流程自检
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import signal
from pathlib import Path

import ledger
import llm
import prompt as prompting

ROOT = Path(__file__).resolve().parent


def materialize(run_dir: Path, cand_id: str, proposal: dict, raw_text: str) -> Path:
    cand_dir = run_dir / "candidates" / cand_id
    cand_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "prepare.py", cand_dir / "prepare.py")
    (cand_dir / "train.py").write_text(proposal["code"], encoding="utf-8")
    (cand_dir / "proposal.md").write_text(raw_text, encoding="utf-8")
    (cand_dir / "idea.json").write_text(
        json.dumps({k: proposal[k] for k in ("op", "parent_id", "idea")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cand_dir


def apply_best_params(cand_dir: Path, best_params: dict) -> bool:
    """把筛查最优参数写回候选 train.py 的 BASE_PARAMS（子代 improve 从该状态出发）。"""
    path = cand_dir / "train.py"
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    lines = src.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and (
            (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BASE_PARAMS" for t in node.targets))
            or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "BASE_PARAMS")
        ):
            lines[node.lineno - 1: node.end_lineno] = [f"BASE_PARAMS = {best_params!r}\n"]
            path.write_text("".join(lines), encoding="utf-8")
            return True
    return False


def run_screening(cand_dir: Path, k: int, seed: int, timeout: int) -> dict:
    return _run_screening_subprocess(cand_dir, timeout, "--k", str(k), "--seed", str(seed))


def run_final_eval(cand_dir: Path, params: dict, timeout: int) -> dict:
    return _run_screening_subprocess(cand_dir, timeout, "--final", "--params-json", json.dumps(params))


def _run_screening_subprocess(cand_dir: Path, timeout: int, *extra_args: str) -> dict:
    log_path = cand_dir / "screening.log"
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "screening.py"), str(cand_dir), *extra_args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cand_dir,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        log_path.write_text(out + err + f"\nTIMEOUT after {timeout}s", encoding="utf-8")
        return {"best_score": None, "best_params": None, "trials": [], "error": f"筛查超时（>{timeout}s）"}
    log_path.write_text(stdout + stderr, encoding="utf-8")
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT_JSON: "):
            return json.loads(line[len("RESULT_JSON: "):])
    return {"best_score": None, "best_params": None, "trials": [], "error": "筛查子进程未返回结果"}


def resolve_proposal(records: list[dict], proposal: dict) -> dict:
    """校验 op 与 parent：improve 的父代必须是有效的 keep 候选，否则整案降级为 fresh。

    不做“换父代”修正——提案代码是按声明父代写的，换父代会制造虚假继承关系。
    """
    proposal = dict(proposal)
    kept_ids = {r["run_id"] for r in records if r.get("status") == "keep"}
    if proposal["op"] == "improve":
        pid = (proposal.get("parent_id") or "").strip()
        if pid.isdigit():
            pid = pid.zfill(3)
        if pid in kept_ids:
            proposal["parent_id"] = pid
        else:
            proposal["op"] = "fresh"
            proposal["parent_id"] = None
            proposal["note"] = f"声明的父代 {pid or 'none'!r} 不是有效的 keep 候选，按 fresh 记账"
    else:
        proposal["parent_id"] = None
    return proposal


def propose(args, records, profile, round_index):
    """调用 LLM（或 mock），解析失败重试一次。"""
    best = ledger.best_kept_record(records)
    best_code = None
    if best is not None:
        code_path = args.run_dir / "candidates" / best["run_id"] / "train.py"
        if not code_path.exists():
            raise SystemExit(f"keep 候选 {best['run_id']} 的代码缺失（{code_path}）。请恢复候选目录或换 tag 重跑。")
        best_code = code_path.read_text(encoding="utf-8")
    user_prompt = prompting.build_round_prompt(profile, ledger.summary_view(records), best, best_code)

    last_err = None
    for _ in range(2):
        if args.mock:
            import mock_llm
            text = mock_llm.mock_chat(round_index)
        else:
            text = llm.chat(prompting.SYSTEM_PROMPT, user_prompt, model=args.model)
        try:
            return llm.parse_proposal(text), text
        except ValueError as e:
            last_err = e
            user_prompt += f"\n\n你上一条回复无法解析：{e}。请严格按约定格式（OP/PARENT/IDEA 头 + 一个 ```python 代码块）重新输出完整提案。"
    raise RuntimeError(f"LLM 提案两次均无法解析：{last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=8, help="搜索轮数（每轮产生一个候选）")
    ap.add_argument("--tag", default="demo", help="运行名，产物在 runs/<tag>/")
    ap.add_argument("--model", default=os.environ.get("DEMO_MODEL") or os.environ.get("ANTHROPIC_MODEL"))
    ap.add_argument("--data", help="自定义 CSV 路径（默认用内置乳腺癌数据集）")
    ap.add_argument("--target", help="CSV 中的目标列名")
    ap.add_argument("--k-random", type=int, default=4, help="每个候选在 SEARCH_SPACE 中随机筛查的配置数")
    ap.add_argument("--timeout", type=int, default=600, help="单个候选筛查整体超时（秒）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mock", action="store_true", help="不调用 LLM，用内置提案做流程自检")
    args = ap.parse_args()

    if args.data:
        if not args.target:
            ap.error("使用 --data 时必须同时给出 --target <目标列名>")
        os.environ["DEMO_DATA"] = str(Path(args.data).resolve())
        os.environ["DEMO_TARGET"] = args.target
    if not args.mock and not args.model:
        ap.error("请用 --model 或环境变量 DEMO_MODEL 指定模型（如 claude-opus-4-6、deepseek-chat 等）")

    args.run_dir = ROOT / "runs" / args.tag
    args.run_dir.mkdir(parents=True, exist_ok=True)

    import prepare
    profile = prepare.data_profile()
    (args.run_dir / "task.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据: {profile['n_samples']} 样本 × {profile['n_features']} 特征，目标 {profile['target']}，类别分布 {profile['class_counts']}")
    print(f"产物目录: {args.run_dir}\n")

    for i in range(args.rounds):
        cand_id = f"{i:03d}"
        records = ledger.load(args.run_dir)
        if any(r["run_id"] == cand_id for r in records):
            print(f"[{cand_id}] 已存在，跳过（支持断点续跑）")
            continue
        try:
            proposal, raw_text = propose(args, records, profile, i)
        except llm.LLMTransportError as e:
            raise SystemExit(f"LLM 出口故障（{e}）。环境性问题重跑无意义，已中止；修复后可用同一 tag 续跑。")
        except Exception as e:
            # 候选性失败（提案内容无法解析等）：不入账本、不占轮次槽位，同 tag 续跑会重试本轮
            print(f"[{cand_id}] 提案失败（未记账，不占用轮次）: {type(e).__name__}: {e}")
            continue
        proposal = resolve_proposal(records, proposal)
        cand_dir = materialize(args.run_dir, cand_id, proposal, raw_text)

        result = run_screening(cand_dir, args.k_random, args.seed + i, args.timeout)
        params_applied = False
        if result.get("best_params"):
            params_applied = apply_best_params(cand_dir, result["best_params"])
        record = {
            "run_id": cand_id,
            "op": proposal["op"],
            "parent_id": proposal.get("parent_id"),
            "idea": proposal["idea"],
            "score": result["best_score"],
            "best_params": result.get("best_params"),
            "n_trials": len(result.get("trials", [])),
            "params_writeback": params_applied,
            "error": result.get("error"),
        }
        if proposal.get("note"):
            record["note"] = proposal["note"]
        record = ledger.add_record(args.run_dir, record)
        score = "crash" if record["score"] is None else f"{record['score']:.4f}"
        print(f"[{cand_id}] {record['op']}(parent={record['parent_id']}) val_score={score} -> {record['status']}  {record['idea'][:60]}")

    # 搜索结束：对最优候选做一次且仅一次留出测试评估
    records = ledger.load(args.run_dir)
    best = ledger.best_kept_record(records)
    if best is not None:
        cand_dir = args.run_dir / "candidates" / best["run_id"]
        if best.get("final_test_score") is None:
            res = run_final_eval(cand_dir, best.get("best_params") or {}, args.timeout)
            if res.get("final_score") is not None:
                ledger.set_final_score(args.run_dir, best["run_id"], res["final_score"])
            else:
                print(f"最终留出测试评估失败: {res.get('error')}")

    write_summary(args.run_dir)
    if not ledger.best_kept_record(ledger.load(args.run_dir)):
        raise SystemExit(1)


def write_summary(run_dir: Path) -> None:
    records = ledger.load(run_dir)
    best = ledger.best_kept_record(records)
    lines = [
        "# 搜索结果汇总",
        "",
        "score 列为**搜索验证分数**（负 balanced accuracy，越小越好）；选择、筛查与排行榜都只用它。",
        "",
        ledger.summary_view(records, max_idea_chars=200),
        "",
    ]
    if best:
        lines += [
            f"最优候选: {best['run_id']}（验证 balanced accuracy = {-best['score']:.4f}）",
            f"代码: candidates/{best['run_id']}/train.py（BASE_PARAMS 已是筛查最优参数）",
            f"参数: `{json.dumps(best.get('best_params'), ensure_ascii=False)}`",
        ]
        if best.get("final_test_score") is not None:
            lines.append(
                f"最终留出测试: balanced accuracy = {-best['final_test_score']:.4f}"
                "（测试集仅评估过这一次，未参与任何选择）"
            )
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
