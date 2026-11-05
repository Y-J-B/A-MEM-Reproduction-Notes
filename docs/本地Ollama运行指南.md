# 本地 Ollama 运行 A-Mem 评测：操作指南

> 环境假设：Windows + Git Bash，Ollama 已装好并能在本机 11434 端口服务
> 目标：先用 1 个对话样本（约 199 题）跑通全链路，再决定是否扩量

---

## 0. 开始前必须知道的三件事

### 0.1 模型名不能带冒号（Windows 专属坑）

评测代码把模型名直接拼进目录名和日志文件名：

```python
# test_advanced_robust.py:204-207
memories_dir = os.path.join(..., "cached_memories_robust_{}_{}".format(backend, model))
# test_advanced_robust.py:180
log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{timestamp}.log"
```

Ollama 的模型名天生带 tag（`qwen2.5:7b`），冒号在 Windows 里是非法路径字符，`os.makedirs()` 会直接抛 `OSError: [WinError 123]`。

**解决：复制一个不带 tag 的别名，跑评测时用别名。**

```bash
ollama cp qwen2.5:7b qwen25-7b
# 之后统一用 --model qwen25-7b
```

### 0.2 时间成本（务必先小样本）

| 阶段 | 每个样本的量 | 说明 |
|---|---|---|
| 记忆构建 | 约 588 turn | 每 turn 至少 1 次 LLM 调用，触发演化再加 1~2 次 |
| 单样本合计 | 约 1200~1800 次调用 | 7B 模型本地推理约 1~2 小时 |
| 答题 | 约 199 题 × 2 次调用 | 约 20~30 分钟 |
| **全量 10 样本** | 约 1.5 万次调用 | **15~20 小时，不要一上来就跑** |

`--ratio 0.1` 在 10 个样本上取 `max(1, int(10*0.1)) = 1`，即**恰好跑 1 个完整对话**，用来验证链路最合适。

### 0.3 模型别选太小

记忆构建依赖 LLM 严格按段标记格式输出（`llm_text_parsers.py` 靠 `KEYWORDS:`、`CONTEXT:` 这类标记切分）。1B~3B 模型经常不守格式，会触发 fallback 重试（额外调用）甚至退化成启发式关键词。

建议起步 **qwen2.5:7b**（英文数据也够用），显存紧张再退到 `qwen2.5:3b`。

---

## 1. 安装依赖

```bash
cd /e/Yang/project/A-mem-main

python -m venv .venv
source .venv/Scripts/activate
```

如果**没有 NVIDIA 显卡**，先装 CPU 版 torch，否则会拉 2GB+ 的 CUDA 包：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

有显卡就直接 `pip install -r requirements.txt`。

> `requirements.txt` 里已包含 `openai`。这个包**必须装**，即便你只用 Ollama —— 因为 `utils.py:16` 有一条未使用的 `from openai import OpenAI`，缺包会在 import 阶段直接报错。

### 1.1 补下 NLTK 数据（高频坑）

代码只下载了 `punkt` 和 `wordnet`，但 **nltk ≥ 3.9 分词需要额外的 `punkt_tab`**，缺了会在算 BLEU 时报 `LookupError: punkt_tab`。

```bash
python -c "import nltk; [nltk.download(x) for x in ['punkt','punkt_tab','wordnet','omw-1.4']]"
```

---

## 2. 准备 Ollama 模型

```bash
ollama list                          # 看现有模型
ollama pull qwen2.5:7b               # 没有就拉
ollama cp qwen2.5:7b qwen25-7b       # 关键：复制成无冒号别名
ollama run qwen25-7b "Reply with exactly one word: READY"
```

最后一条要能返回 `READY` 之外的非空内容，说明服务正常。

---

## 3. 先做连通性自测

**不要跳过这步。** 用十几行代码确认「Ollama 通 + 解析器能工作」，比跑到一半报错省几小时。

在项目根目录建 `check_env.py`：

```python
from memory_layer_robust import RobustLLMController
from llm_text_parsers import ANALYZE_CONTENT_PROMPT, parse_analyze_content

ctl = RobustLLMController(backend="ollama", model="qwen25-7b", check_connection=True)
print("[1] 连通性 OK")

resp = ctl.llm.get_completion(
    ANALYZE_CONTENT_PROMPT.format(content="Caroline went to a charity race on 8 May 2023.")
)
print("[2] 原始输出 >>>\n" + resp[:500])

a = parse_analyze_content(resp, "Caroline went to a charity race on 8 May 2023.")
print("[3] 解析结果:", a)
assert a["keywords"], "关键词解析为空 —— 该模型格式服从性不足，换更大的模型"
print("[4] 解析器 OK")
```

```bash
python check_env.py
```

第 `[3]` 步应该看到非空的 `keywords` 列表。如果为空，别继续跑 —— 换 `qwen2.5:14b` 或 `llama3.1:8b`。

---

## 4. 小样本跑通

```bash
python test_advanced_robust.py \
  --backend ollama \
  --model qwen25-7b \
  --dataset data/locomo10.json \
  --ratio 0.1 \
  --retrieve_k 10 \
  --output results_smoke_qwen25-7b.json
```

跑起来后另外开一个终端看日志：

```bash
tail -f logs/eval_robust_qwen25-7b_ollama_ratio0.1_*.log
```

正常的话会先看到 `Creating new memories`，然后是一长串 `Successfully cached ... memories`。**记忆缓存是断点续跑的关键**：中断后重跑会读 `cached_memories_robust_ollama_qwen25-7b/memory_cache_sample_0.pkl`，不会重建。

---

## 5. 看结果

```python
import json
r = json.load(open("results_smoke_qwen25-7b.json"))
print("题数:", r["total_questions"])
print("分布:", r["category_distribution"])
for split, m in r["aggregate_metrics"].items():
    print(split, " F1=%.4f  BLEU-1=%.4f" % (m["f1"]["mean"], m["bleu1"]["mean"]))
```

报告里**只把 `f1` 和 `bleu1` 当主指标**，其余 12 项作辅助。别报 `exact_match` —— 严格全字符串匹配，LoCoMo 短答案下几乎恒为 0，没有解释力。

---

## 6. 扩量与调 k

跑通后按需扩量：

```bash
# 3 个对话样本
python test_advanced_robust.py --backend ollama --model qwen25-7b \
  --ratio 0.3 --retrieve_k 10 --output results_ratio03.json
```

`retrieve_k` 对结果影响很大，论文是靠逐模型扫 k 取最优。因为记忆已缓存，扫 k 只重跑答题部分，很便宜：

```bash
for k in 5 10 15 20; do
  python test_advanced_robust.py --backend ollama --model qwen25-7b \
    --ratio 0.3 --retrieve_k $k --output results_k$k.json
done
```

> 注意：`run_k_sweep.sh` 和 `run_all_experiments.sh` **不要直接用**。里面 `WORKDIR`、`AMEM_PYTHON` 全是作者集群路径 `/common/users/wx139/...`，而且依赖 `lsof` 和 GPU 端口管理，Windows 上跑不起来。上面这段手写循环等价。

---

## 7. 报错对照表

| 报错 | 原因 | 处理 |
|---|---|---|
| `OSError: [WinError 123]` 路径含非法字符 | 模型名带 `:` | `ollama cp` 复制成无 tag 别名 |
| `ModuleNotFoundError: openai` | 只装了部分依赖 | `pip install openai`（`utils.py:16` 的无用导入导致） |
| `LookupError: punkt_tab` | nltk 新版缺资源 | 见 §1.1 补下载 |
| `ConnectionError: Cannot reach LLM backend` | Ollama 没起 / 模型名不存在 | `ollama list` 核对；`ollama serve` |
| 大量 `Keywords empty after initial parse` | 模型太小，不守格式 | 换 7B 以上模型 |
| `Evolution failed for note ...` | 邻居更新解析失败 | 可容忍（该条记忆降级为不演化），但日志里若占比高要换模型 |
| 跑到一半 OOM | `bert_score` 默认加载 roberta-large | 见下 |

`bert_score` 默认用 `roberta-large`（约 1.4GB），且**每条 QA 单独调用一次**，是内存和时间的隐性大头。显存/内存紧张时，可在 `utils.py:71` 临时加 `model_type='distilbert-base-uncased'`。

---

## 8. 建议的推进顺序

1. §1.1 补 nltk 数据 → §2 复制模型别名 → §3 自测通过
2. §4 跑 `--ratio 0.1`，确认能出 JSON
3. 检查结果里各 category 的题量是否符合预期（样本 0 的分布）
4. 扩到 `--ratio 0.3`，跑一次 k-sweep（k=5/10/15）
5. 有稳定结论后再考虑全量

**报告中不要写任何未实际跑出的分数。** 当前仓库里没有任何历史结果文件。
