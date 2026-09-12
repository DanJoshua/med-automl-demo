# 医疗表格 AutoML 搜索 Demo

从我们的 bi-level 自动研究系统抽离的极简版本：LLM 自动提出模型训练方案（代码），系统真实运行并评分，按账本（ledger）迭代搜索更优方案。面向医疗表格分类场景，开箱即用。

## 它是怎么工作的

```
每一轮：
  LLM 提案（fresh 探索新方向 / improve 改进当前最优分支，输出 idea + 完整 train.py）
    → driver 落盘候选目录 candidates/<id>/
    → 子进程筛查：评估 BASE_PARAMS + 在 SEARCH_SPACE 中随机采样的 K 个配置（真实训练、隐藏测试集评分）
    → 记入 ledger.json：keep（刷新最优）/ discard（未更优）/ crash（运行失败）
  下一轮 LLM 能看到账本与当前最优代码，在此基础上继续迭代
```

- 候选代码契约：`train.py` 定义 `make_model(dataset, params)`、`BASE_PARAMS`、`SEARCH_SPACE`（见 `prompt.py`）。
- 评估表面固定且防泄漏：`prepare.py` 先按 60/20/20 分层划分训练/验证/测试（固定种子），缺失值填充与类别编码等预处理规则只在训练集上拟合再应用到验证/测试。搜索、筛查与排行榜全部使用**验证集分数**（负 balanced accuracy，对类别不平衡稳健）；测试集只在搜索结束后对最优候选评估一次（`final_evaluate`），结果单独标注为“最终留出测试”。
- 参数回传：每轮筛查后，该候选的最优参数会被写回其 `train.py` 的 `BASE_PARAMS`，后续 improve 子代从已调优状态出发。
- 隐私边界：送往 LLM 的只有数据画像（样本/特征数、编码后的类别计数、**匿名化特征标识 f0/f1/...**）；原始特征名与类别取值不离开本地。
- 完整日志：每个候选目录保存提案原文（`proposal.md`）、代码（`train.py`）、筛查日志（`screening.log`）；全局有 `ledger.json` 与 `summary.md`。

## 快速开始

环境：Python 3.10+，CPU 即可。

```bash
pip install -r requirements.txt

# 1) 先用 mock 模式自检流程（不消耗 API）
python demo.py --mock --rounds 4 --tag smoke

# 2) 配置 LLM（二选一）
#    Anthropic 协议：
export ANTHROPIC_BASE_URL="https://your-endpoint"
export ANTHROPIC_AUTH_TOKEN="sk-..."
export DEMO_MODEL="claude-opus-4-6"
#    或 OpenAI 兼容协议（DeepSeek / 通义 / Kimi / 各类转发）：
export OPENAI_BASE_URL="https://your-endpoint/v1"
export OPENAI_API_KEY="sk-..."
export DEMO_MODEL="deepseek-chat"

# 3) 跑真实搜索（默认内置乳腺癌医学数据集，无需下载数据）
python demo.py --rounds 8 --tag first-run
```

结果打印在终端并写入 `runs/<tag>/summary.md`。

## 换成医院自己的数据

任意表格 CSV（一行一个样本，一列是结局/标签）：

```bash
python demo.py --data 医院数据.csv --target 结局列名 --rounds 8
```

数据预处理由 `prepare.py` 自动完成：缺失值中位数填充、低基数类别列 one-hot、高基数 ID 类列丢弃。多分类、二分类均可。

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--rounds` | 8 | 搜索轮数（每轮一个候选） |
| `--k-random` | 4 | 每个候选在 SEARCH_SPACE 里随机筛查的配置数 |
| `--timeout` | 600 | 单个候选筛查整体超时（秒） |
| `--tag` | demo | 运行名；同一 tag 重跑会跳过已有候选（断点续跑） |
| `--model` | `$DEMO_MODEL` | 模型名 |
| `--mock` | 关 | 不调用 LLM，用内置提案自检 |

## 与完整系统的对应关系

本 demo 保留了完整系统的核心机制，砍掉了研究组件：

- 保留：候选即代码（`make_model` 契约）、fresh/improve 树状搜索、父代代码继承（improve 从父代码出发小步改）、先轻度调参再评分（筛查）、账本 keep/discard/crash 规则、隐藏测试集。
- 砍掉：语义搜索空间与文献侦察（background-researcher）、PUCB 图选择、judged-slate 评审、HEBO 深度调优与调度器、多 GPU 资源管理。

需要更深度的调优或多任务并行时，可以在 `screening.py` 里把随机采样换成 Optuna/HEBO，或增加轮数与 K。

## 常见问题

- **模型怎么选？** 任何能写代码的强模型均可；`--model` 传服务端识别的名字即可。
- **可以只用 scikit-learn 吗？** 可以，删掉 requirements 里的 xgboost/lightgbm，并在 `prompt.py` 的“可用库”一句中删掉它们。
- **数据隐私？** 数据不出本地：原始数据仅由本地 `prepare.py` 读取；LLM 只看到数据画像（行列数、编码后的类别分布、匿名特征标识 f0/f1/...），原始列名与类别取值（如医师姓名、科室名）不会进入 LLM 请求。
