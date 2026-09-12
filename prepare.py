"""固定评估表面 —— 候选代码不得修改本文件。

框架统一最小化：score = -balanced_accuracy（负的平衡准确率，越小越好）。

评估协议（防泄漏）：
- 先划分，后预处理：分层划分 训练 60% / 验证 20% / 测试 20%（固定种子）；
  缺失值填充、类别编码等预处理规则只在训练集上拟合，再应用到验证/测试集。
- 搜索与筛查只看验证集分数（evaluate_config）；测试集由 final_evaluate 保管，
  只在搜索结束后对最优候选评估一次。

数据来源：
- 默认：sklearn 自带的乳腺癌（Wisconsin diagnostic）医学数据集，无需联网下载；
- 自定义：设置环境变量 DEMO_DATA=/path/to/data.csv 与 DEMO_TARGET=<目标列名>。

隐私边界：送往 LLM 的数据画像只含匿名特征标识（f0, f1, ...）与编码后的类别计数；
原始特征名与类别取值不离开本地。
"""

from __future__ import annotations

import os
from functools import lru_cache
from types import SimpleNamespace

import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

_TEST_SIZE = 0.2
_VAL_SIZE = 0.2
_SEED = 42
_MAX_CAT_LEVELS = 20  # 类别列基数上限，超过视为 ID 类列丢弃


@lru_cache(maxsize=1)
def _raw_frame():
    """只做加载与目标列编码，不做任何特征预处理（预处理必须在划分之后进行）。"""
    data_path = os.environ.get("DEMO_DATA")
    target = os.environ.get("DEMO_TARGET")
    if not data_path:
        bunch = load_breast_cancer()
        X = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
        y = pd.Series(bunch.target, name="diagnosis")
        return X, y
    if not target:
        raise ValueError("使用自定义 CSV 时必须同时设置 DEMO_TARGET=<目标列名>")
    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"目标列 {target!r} 不在 {data_path} 中：{list(df.columns)}")
    df = df.dropna(subset=[target])
    y = df[target]
    if not is_numeric_dtype(y):
        y = pd.Series(pd.Categorical(y).codes, index=y.index, name=y.name)
    return df.drop(columns=[target]), y


class _Preprocessor:
    """预处理规则只在训练集上 fit，再 transform 验证/测试集。"""

    def fit(self, X: pd.DataFrame) -> "_Preprocessor":
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.medians: dict[str, float] = {}
        self.levels: dict[str, list] = {}
        for col in X.columns:
            s = X[col]
            if s.dropna().empty or s.nunique(dropna=True) <= 1:
                continue
            if is_numeric_dtype(s):
                # Continuous measurements can have near-unique values; only
                # integer columns use this ID heuristic.
                if is_integer_dtype(s) and s.nunique(dropna=True) / max(len(s.dropna()), 1) > 0.98:
                    continue
                self.num_cols.append(col)
                med = pd.to_numeric(s, errors="coerce").median()
                self.medians[col] = 0.0 if pd.isna(med) else float(med)
            elif s.nunique(dropna=True) <= _MAX_CAT_LEVELS:
                self.cat_cols.append(col)
                self.levels[col] = list(pd.unique(s.dropna()))
            # 高基数非数值列（多为 ID / 姓名）直接丢弃
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for col in self.num_cols:
            frames.append(pd.to_numeric(X[col], errors="coerce").fillna(self.medians[col]))
        for col in self.cat_cols:
            cat = pd.Categorical(X[col], categories=self.levels[col])
            frames.append(pd.get_dummies(pd.Series(cat, index=X.index, name=col), prefix=col, dummy_na=True))
        if not frames:
            return pd.DataFrame(index=X.index)
        return pd.concat(frames, axis=1).astype(float)


@lru_cache(maxsize=1)
def _parts() -> SimpleNamespace:
    X, y = _raw_frame()
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=_TEST_SIZE, random_state=_SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=_VAL_SIZE / (1 - _TEST_SIZE), random_state=_SEED, stratify=y_tv
    )
    prep = _Preprocessor().fit(X_train)
    n_classes = int(y.nunique())

    def view(Xd: pd.DataFrame, yd: pd.Series) -> SimpleNamespace:
        return SimpleNamespace(
            X=Xd.reset_index(drop=True),
            y=yd.reset_index(drop=True),
            feature_names=list(Xd.columns),
            n_classes=n_classes,
            task="classification",
        )

    return SimpleNamespace(
        train=view(prep.transform(X_train), y_train),
        val=view(prep.transform(X_val), y_val),
        test=view(prep.transform(X_test), y_test),
        trainval=view(prep.transform(X_tv), y_tv),
        n_samples=int(len(y)),
    )


def data_profile() -> dict:
    """送往 LLM 的数据画像：特征名一律匿名化，类别计数用编码后的整数标签。"""
    p = _parts()
    return {
        "n_samples": p.n_samples,
        "n_features": p.train.X.shape[1],
        "n_classes": p.train.n_classes,
        "class_counts": {str(k): int(v) for k, v in _raw_frame()[1].value_counts().items()},
        "feature_names": [f"f{i}" for i in range(p.train.X.shape[1])],
        "feature_names_note": "特征名已匿名化（原始列名与类别取值不离开本地）",
        "target": "target",
    }


def evaluate_config(make_model, params) -> float:
    """搜索用的唯一 config -> score 函数：训练集拟合，隐藏验证集评分。越小越好。"""
    p = _parts()
    model = make_model(p.train, params)
    model.fit(p.train.X, p.train.y)
    pred = model.predict(p.val.X)
    return -float(balanced_accuracy_score(p.val.y, pred))


def final_evaluate(make_model, params) -> float:
    """搜索结束后仅调用一次：train+val 重训，隐藏测试集评分。越小越好。"""
    p = _parts()
    model = make_model(p.trainval, params)
    model.fit(p.trainval.X, p.trainval.y)
    pred = model.predict(p.test.X)
    return -float(balanced_accuracy_score(p.test.y, pred))
