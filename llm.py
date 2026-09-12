"""LLM 接入：一次对话调用返回文本。支持两种协议，用环境变量配置：

- Anthropic 协议：ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN（或 ANTHROPIC_API_KEY）
- OpenAI 兼容协议：OPENAI_BASE_URL / OPENAI_API_KEY
  （DeepSeek、通义、Kimi 及各类转发服务通常都是这一种）

DEMO_BACKEND=anthropic|openai 可强制选择；默认有 Anthropic 密钥走 Anthropic，否则 OpenAI。
模型名由 --model 或 DEMO_MODEL 指定。
"""

from __future__ import annotations

import os
import re
import ast
import sys


class LLMTransportError(RuntimeError):
    """LLM 出口侧的环境性故障（配置/认证/网络/服务端）：下一轮调用也必然失败。

    与提案内容解析失败（ValueError，候选性问题）区分：调用方应中止运行，而不是逐轮隔离记账。
    """


def pick_backend() -> str:
    forced = os.environ.get("DEMO_BACKEND")
    if forced:
        return forced
    if os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def chat(system: str, user: str, model: str, max_tokens: int = 32000) -> str:
    backend = pick_backend()
    if backend == "anthropic":
        try:
            import anthropic

            client = anthropic.Anthropic(
                base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
                api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
                auth_token=os.environ.get("ANTHROPIC_AUTH_TOKEN") or None,
            )
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as stream:
                resp = stream.get_final_message()
        except Exception as e:
            raise LLMTransportError(f"{type(e).__name__}: {e}") from e
        return "".join(b.text for b in resp.content if b.type == "text")
    if backend == "openai":
        try:
            from openai import BadRequestError, OpenAI

            kwargs = dict(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            client = OpenAI()
            try:
                resp = client.chat.completions.create(max_tokens=max_tokens, **kwargs)
            except BadRequestError as e:
                # 仅「端点不接受 max_tokens」值得换请求形状重试；其余 400 是真实错误，原样上报
                if "max_tokens" not in str(e):
                    raise
                print(f"端点不接受 max_tokens 参数，去掉后重试（首次错误：{e}）", file=sys.stderr)
                resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMTransportError(f"{type(e).__name__}: {e}") from e
        return resp.choices[0].message.content or ""
    raise LLMTransportError(f"未知 DEMO_BACKEND: {backend!r}")


_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def parse_proposal(text: str) -> dict:
    """解析 LLM 提案：OP / PARENT / IDEA 三个头 + 一个 python 代码块。"""
    m_op = re.search(r"^OP:\s*(\w+)", text, re.MULTILINE)
    m_parent = re.search(r"^PARENT:\s*(\S+)", text, re.MULTILINE)
    m_idea = re.search(r"^IDEA:\s*(.+?)(?=^```|\Z)", text, re.MULTILINE | re.DOTALL)
    blocks = _CODE_RE.findall(text)
    m_code = next((code for code in blocks if _defines_make_model(code)), None)
    if m_code is None and blocks:
        m_code = max(blocks, key=len)
    if not (m_op and m_idea and m_code):
        raise ValueError("回复缺少 OP/IDEA 头或 python 代码块")
    op = m_op.group(1).strip().lower()
    if op not in ("fresh", "improve"):
        raise ValueError(f"OP 必须是 fresh 或 improve，实际为 {op!r}")
    parent = m_parent.group(1).strip() if m_parent else "none"
    if parent.lower() in ("none", "null", "-"):
        parent = None
    return {"op": op, "parent_id": parent, "idea": m_idea.group(1).strip(), "code": m_code}


def _defines_make_model(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "make_model" for n in tree.body)
