"""--mock 模式的假 LLM：不消耗 API，用于验证安装与流程。

返回与真实 LLM 完全相同的文本格式（OP/PARENT/IDEA 头 + python 代码块），
因此同一条解析、落盘、筛查、记账路径都会被走到。
"""

from __future__ import annotations

_LOGREG = '''
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_PARAMS = {"C": 1.0, "max_iter": 2000}

SEARCH_SPACE = {
    "C": {"type": "float", "low": 0.01, "high": 100.0, "log": True},
}


def make_model(dataset, params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=params["C"], max_iter=int(params["max_iter"]),
            class_weight="balanced", random_state=42,
        )),
    ])
'''

_RF = '''
from sklearn.ensemble import RandomForestClassifier

BASE_PARAMS = {"n_estimators": 300, "max_depth": None}

SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 100, "high": 600},
    "max_depth": {"type": "choice", "values": [3, 5, 8, None]},
}


def make_model(dataset, params):
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
'''

_RF_IMPROVED = '''
from sklearn.ensemble import RandomForestClassifier

BASE_PARAMS = {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 2}

SEARCH_SPACE = {
    "n_estimators": {"type": "int", "low": 300, "high": 800},
    "max_depth": {"type": "choice", "values": [5, 8, 12]},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 5},
}


def make_model(dataset, params):
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
'''

_HGB = '''
from sklearn.ensemble import HistGradientBoostingClassifier

BASE_PARAMS = {"learning_rate": 0.1, "max_iter": 300}

SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "max_iter": {"type": "int", "low": 100, "high": 500},
}


def make_model(dataset, params):
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_iter=int(params["max_iter"]),
        random_state=42,
    )
'''

_PROPOSALS = [
    {"op": "fresh", "parent_id": None,
     "idea": "基线方向：标准化 + 均衡类权重的逻辑回归，简单稳健，作为后续改进的参照。",
     "code": _LOGREG},
    {"op": "fresh", "parent_id": None,
     "idea": "换模型家族：均衡类权重随机森林，捕捉非线性交互，医疗表格上通常强于线性模型。",
     "code": _RF},
    {"op": "improve", "parent_id": "000",
     "idea": "在随机森林分支上小步改进：增加树数、限制深度与叶节点样本数、改用 balanced_subsample 控制过拟合。",
     "code": _RF_IMPROVED},
    {"op": "fresh", "parent_id": None,
     "idea": "探索梯度提升方向：HistGradientBoosting，对表格数据拟合能力强且训练快。",
     "code": _HGB},
]


def mock_chat(round_index: int) -> str:
    p = _PROPOSALS[round_index % len(_PROPOSALS)]
    parent = p["parent_id"] or "none"
    return f"OP: {p['op']}\nPARENT: {parent}\nIDEA: {p['idea']}\n```python\n{p['code'].strip()}\n```\n"
