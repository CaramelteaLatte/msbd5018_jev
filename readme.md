## 0 API key 与本地数据

API key 通过环境变量读取。将 `.env.example` 复制为 `.env`，填写自己的密钥，然后在仓库根目录执行：

```bash
set -a
source .env
set +a
```

`.env` 只保留在本地，不提交 Git。需要配置 `TYPESAFE_API_KEY` 和 `DASHSCOPE_API_KEY`。

原始 VitaminC JSONL 和实验切分 JSONL 不单独提交 Git，而是打包上传；逐条 API 日志和 SQLite 检查点保留在本地。GitHub 保存代码、数据压缩包、数据来源说明/许可证、配置、结果汇总和图表。

数据包已随仓库提供：[data/vitaminc.zip](data/vitaminc.zip)，包含原始 `vitaminc/` 和本次使用的 `vitaminc_splits/`（包括当前 2,000 条 test）。包内文件及 SHA256 记录在 `data/vitaminc_archive_manifest.json`。克隆仓库后，在仓库根目录还原：

```bash
unzip data/vitaminc.zip -d data
```

离线审计脚本还需要本地 `code/*_results/` 下的 SQLite 检查点；这些实验日志不包含在上述数据包中。

## 1 数据与 v1 划分

本项目使用 VitaminC 的 fact verification 数据：仅根据给定 claim/evidence 判定标签，不包含检索或生成环节。官方数据分为 train/dev/test；本项目只从 dev/test 的 real 样本构建实验集，没有使用 train 训练模型。

三个标签分别表示：S——支持，C——矛盾，I——证据不足

### 1.1 `split_vitaminc.py`

v1 从官方 dev/test 切分为以下四份，每份包含 S/C/I。原算法通过 page、case、完整文本对及 ID 建立连通组，清理冲突标签和重复文本对，再做组隔离与标签分层采样。

| 实验子集 | 从哪里抽取 | 条数 | 用来做什么 |
|---|---|---:|---|
| `prompt_dev.jsonl` | 官方 `dev.jsonl` | 200 | 调整提示词、测试 API 能否正常返回结果 |
| `calibration.jsonl` | 官方 `dev.jsonl` | 2,000 | 预留校准集；v1/v2 均未拟合校准器 |
| `threshold_validation.jsonl` | 官方 `dev.jsonl` | 2,000 | 选择置信度低于多少时，把样本转交给 LLM |
| `test.jsonl` | 官方 `test.jsonl` | 实际 2,000（原抽取 5,000） | 固定提示词与阈值后做最终评估 |

v1 的 `split_report.json` 保存了最初 5,000 条 test 的划分记录，后来缩减为 2,000 条，旧报告未同步；该历史问题保留在审计记录中。v2 的数量和哈希由脚本生成。

## 2 v1 实验记录（已完成）

### 2.1 `test_jev.py`

测试 jev 接口。

- input：python test_jev.py ../data/vitaminc_splits/prompt_dev.jsonl
    - 后不加参数表示默认测试第一条数据，后加参数n表示测试第n条数据
- output：

```json
//python test_jev.py ../data/vitaminc_splits/prompt_dev.jsonl 6
样本编号: 6
Claim: Jamie XX produced Drake 's song Take Care that includes Rihanna 's vocals .
Evidence: It featured the single `` Far Nearer '' which was selected Best New Track by the influential Pitchfork Media music site and charted at # 128 in the UK singles chart.  produced the title track off Drake 's sophomore album , Take Care , which featured Rihanna.
{
  "model": "jev-1.14.0",
  "prediction": "S",
  "probabilities": {
    "S": 0.55,
    "C": 0.04,
    "I": 0.41
  },
  "confidence": 0.33,
  "gold": "S"
}
```

prediction 是 jev 的预测结果，probability 是 jev 得到的可能性分数，confidence 是 jev 对这次打分的自信程度，gold 是这条测试用例的 gt 值

confidence 是 API 返回的原始分数，不是经本项目校准的正确率。高分与低分的实际表现需要分组验证；低置信度不保证交给 LLM 后更准确。

### 2.2 llm & tokens

v1 使用 [阿里云百炼](https://bailian.console.aliyun.com/cn-beijing/model/market)，当时预算为 200 元。下表及代码中的费用是保存的实验估算口径，不是当前价格承诺。调用模型：
- qwen3.8-max
- deepseek-v4.1-flash-1017
- kimi-k3.1-pro
- glm-5.3-prime：太贵了，最后我们不调了

百炼国际站（新加坡）的 API key 通过 `DASHSCOPE_API_KEY` 配置。实验中北京区 API key 无法调用当前国际站地址，改为新加坡区后可以调用；当时使用新加坡 VPN 节点。

### 2.3 `run_pilot.py`

- 从 `prompt_dev.jsonl` 中随机采样了 50 条数据（15 S, 15 C, 20 I）。
- 让 Jev 和四个 LLM 各跑这 50 条，一共进行 50 次 jev 调用和 200 次 llm 调用。
- 每条结果存下来，最少记录样本 ID、模型、预测标签、耗时和输入／输出 token。Jev 额外保存三类概率和 confidence。请求失败或返回无法解析的标签时，单独记为失败，避免算成 I。
- 看三项结果。 看每个模型的标签输出是否稳定、实际单条成本是多少，以及在 Jev 低置信度的样本里，纠正了几条、改错了几条。

一个测试，大致看一下预算需要多少。实验结果是 cut 掉 glm，一个模型占据了 80% 的开销。

实验结果在 `code/pilot_results`

### 2.4 v1 threshold 验证

选择一个适合的 threshold 作为 low confidence 的判断，在这个临界值将 jev 判断转交 llm

代码实现思路：
- jev 跑 2k 条数据（来自 `data/vitaminc_splits/threshold_validation.jsonl`），对所有 prediction 结果的 confidence 做排序，得到前 20% 低的 confidence 的数据（是因为我们以 20% 为成本上限，最多 20% 的数据转交 llm 继续判断）
- 在这一个 split 的数据集中我们得到了 396 条前 20% 低 confidence 的数据。注意，这里的 confidence 的数据是离散的，不连续且存在重复的数据，因此后续我们画出来的表也是生硬的。
- 对这些 396 条数据，跑遍 3 个 model，得到每一个 model 在每一个 threshold 下的预测 accuracy（指 jev+llm 的 accuracy）
- 绘表，每一个 model，和几个 model 的平均值

![](assets/threshold_curve.png)

v1 在验证集上以三个级联 accuracy 的平均值选出原始 confidence 阈值 **0.48**，对应转交 **209/2000（10.45%）**，并非 48% 的转交率。该点 JEV only 为 86.40%，JEV + Qwen 为 87.35%，JEV + DeepSeek 为 86.50%，JEV + Kimi 为 86.40%。Qwen 自身的最高验证点是 0.49/0.50（87.40%）。

上图曲线是 **JEV + LLM 的级联准确率**；LLM 各只跑了 396 条，未测完整验证集单模型基线。收益是否跨页面泛化、以及是否节省包含 JEV 在内的总费用，仍需要独立验证。

### 2.5 v1 final eval

在跑这个代码前，我们认为 test 的数据集量太大了没有必要（原先 5k 条），于是我们修改了 `data/vitaminc_splits/test.jsonl`，改为 2k 条数据。具体操作是直接对这个 test 集进行了数据覆盖，没有修改 split 代码文件的代码内容。

v1 当时使用写死的阈值运行。当前脚本已升级为 v2，运行命令见后文；v1 结果文件和图表原样保留。

这个阶段 threshold 固定为 0.48，合计 jev 转发了 216 条 qa。

合计我们跑了 2k 条 jev 调用，2k 条 qwen, kimi, dpsk 调用，result 信息在 `final_eval_results/summary.csv` 中，在这批 VitaminC-real 样本及这套请求配置下，单模型 accuracy 依次为 JEV、Qwen、DeepSeek、Kimi；这个排序不代表通用模型能力。

![alt text](assets/eval_final_results.png)

| 方法 | 准确率 | Macro-F1 | 平均延迟 | P95 延迟 | LLM 费用估算 |
|---|---:|---:|---:|---:|---:|
| **Jev only** | **84.90%** | **82.19%** | **1.39 秒** | **1.49 秒** | — |
| Qwen only | 83.20% | 80.86% | 2.12 秒 | 2.51 秒 | $0.684 |
| **Jev + Qwen** | **84.65%** | **82.05%** | 1.62 秒 | 3.48 秒 | $0.076 |
| DeepSeek only | 77.95% | 74.52% | 2.37 秒 | 3.68 秒 | $0.045–0.091 |
| Jev + DeepSeek | 84.05% | 81.28% | 1.64 秒 | 3.65 秒 | $0.005–0.010 |
| Kimi only | 77.80% | 75.94% | 3.08 秒 | 5.02 秒 | $0.831 |
| Jev + Kimi | 83.60% | 81.09% | 1.74 秒 | 4.33 秒 | $0.093 |

结论：在这次 VitaminC 测试中，低 confidence 转交改善了纯 llm 的准确率和平均延迟，并大幅减少了 llm 调用，但没有观察到它进一步提高 jev 的accuracy

### 2.6 v1 审计：验证收益没有在测试集复现

threshold 的级联曲线与最终单模型排名不是同一统计口径。真正需要解释的是，验证集的小幅级联收益未在 test 重现。

<u>最明确的问题是验证集的页面分布严重偏斜；没有发现评分算错或两阶段请求配置不一致。</u>

完整证据见 [v1 审计报告](code/audit_results/analysis_zh.md)。

我们重新检查了数据集，发现了一个比较明显的数据分布问题：验证集 2k 条中，有 1424 条来自同一个页面（意大利疫情），占 71.2%；Qwen 在阈值 0.48 下净纠正的 19 条中，18 条来自这个页面，其他页面合计只增加 1 条。这是验证收益缺乏代表性的直接证据。

测试集则分布在 316 个页面上。标签比例基本一致，也未发现四个实验子集之间的页面或样本重叠。同时我们仔细检查了一下个别数据集，存在原标签需要人工复核的案例。例如证据“销量超过 1,400 万”，不能严格推出 claim“少于 1,600 万”，但 gold 标为 S。类似疑点同时可能有利于不同模型，目前不能据此判断整体错标率。

低置信度样本中，验证集上 Qwen 比 JEV 准；测试集上 JEV 是 56.48%，Qwen 是 54.17%。JEV 低置信度意味着样本困难，并不保证 LLM 更擅长这些样本。

测试集上 JEV 比 Qwen 单模型高 1.70 个百分点；但级联比 JEV 低 0.25 个百分点，按页面 bootstrap 的 95% 区间为 −1.16～+0.57 个百分点，不足以认定级联稳定变差。


## 3 v2 研究问题与实施方式

研究问题：**在页面分布更分散的数据上，低置信度转交能否稳定改善准确率，或者以较少的 LLM 调用保持接近的表现？**

v2 将页面覆盖作为采样约束，评估对象与 v1 不同；两版分数分别报告，不把差值直接归因于模型或路由提升。全程使用原始 gold，不按模型表现筛样本或改标签。

### 3.1 数据、隔离与复现

- 原始清理规则与 v1 一致，仅使用 real 样本，移除冲突文本对及同标签重复。
- prompt_dev 200 条和 calibration 2,000 条按字节保留，其完整连通组不进入新 validation。calibration 仍只预留，路由使用 raw confidence。
- 新 validation 为 2,000 条，S/C/I = 996/575/429；新 test 为 2,000 条，S/C/I = 1002/575/423。配额沿用清理后官方来源池的比例。
- 新 test 排除旧 test 已评估的页面、case、文本对、ID 及关联的完整连通组。
- validation/test 每个页面最多 20 条。固定 seed `20261109`，在全体可用页面上用整数最大流分配标签配额和页面容量，再按 page/label 随机抽取行。容量不足时报错，不自动放宽规则。
- 输出连通组 ID、输入/输出哈希、标签数量、页面分布及隔离检查；原始样本和 v1 结果不被覆盖。

```bash
python data/split_vitaminc.py --mode v2 \
  --dev data/vitaminc/dev.jsonl --test data/vitaminc/test.jsonl \
  --previous data/vitaminc_splits --out data/vitaminc_splits_v2
```

重建切分前先解压第 0 节的 v1 数据包，以提供保留开发集和旧 test 排除清单。输出目录必须为空。使用 `--mode v1 --seed 20261015 --test-size 5000` 可在另一个新目录复现原始 v1 切分；原来手动缩减 test 的步骤不属于该算法。

### 3.2 完整验证基线、阈值与分组验证

四个模型各预测完整 validation 2,000 条。保持 v1 的模型名、提示词、thinking 和 token 上限；额外分组元数据不发送给模型。默认每模型 8 个并发请求，通过 `--workers` 可调整；本轮实际收集在初始 8 并发后改用 16 并发续跑，运行设置保存在结果目录的 `collection_runs.json`。请求并发会影响延迟，因此 v1/v2 延迟不可直接当作模型速度变化。

每个 LLM 独立选择阈值，以级联正确数最大为目标，同分优先少调用。规则是 `confidence <= threshold`，包括不转交候选（JSON 中为 null）；相同 confidence 的样本整体转交，选择集合上最多转交 20%。测试集及留出折如实报告实际转交率，不据此重新调阈值。

按 page/case 连通组固定划分五折，贪心平衡样本数与 S/C/I 数量；每次在四折选阈值、剩余一折评估。折外分数与完整验证集上选择后的表观分数分别保存。最终阈值用完整 validation 选定，写入包含模型配置、数据哈希和规则的结果文件。

### 3.3 独立 test、统计与失败处理

final eval 必须通过 `--thresholds` 输入完整 v2 阈值产物，校验模型、提示词与请求地址一致，并记录文件 SHA256。禁止根据 test 重新选阈值。

输出七种场景（JEV only、三个 LLM only、三个级联）的 accuracy、Macro-F1、混淆矩阵、各标签表现、纠正/改错、调用量、token、费用和延迟，并做 10,000 次按页面配对 bootstrap（seed `20261109`）。区间以当前样本和原标签为条件，不消除标签噪声或模型随机性。

SQLite 逐次保存请求；同一输出目录的数据/模型/阈值配置改变会拒绝续跑。网络、限流或 5xx 默认最多重试两次；明确的 HTTP 400 `data_inspection_failed` 属于服务商对单条内容的审查拒绝，保留原错误并记为终态，不改写输入、不反复重试。其他错误停止发起新请求并保存已在途结果。无效 LLM 标签或上述服务商拒绝在单模型中计错，在级联中回退 JEV；无效 JEV 返回停止实验。重跑相同命令跳过已有终态结果。

费用沿用 v1 保存的价格假设，JEV 单价未知时总费用保持空值。场景费用和延迟只累计终态请求；实际请求日志另记录中断/重试。若服务商拒绝未返回 token 用量，涉及该请求的费用估算保持空值，不假定其费用为零。级联延迟为 JEV 与被转交 LLM 的耗时之和，不含重试、排队与编排。

级联的调用量、费用和延迟由同一批预测离线重建，表示按固定策略部署时的场景估算。本轮为得到单模型基线而全量请求了所有模型，不能把低转交率当成本轮实验的实际请求节省。

### 3.4 从仓库根目录运行

批量脚本使用 Python 3.9+、标准库、NumPy 和 Matplotlib。单条 `test_jev.py` 另外使用 TypeSafe SDK，不是批量运行的依赖。

```bash
python -m pip install numpy matplotlib
set -a
source .env
set +a

# 只查看预算，不调用 API，也不写文件
python code/run_threshold.py --data data/vitaminc_splits_v2/threshold_validation.jsonl --plan

# 收集完整 validation，并生成阈值曲线、五折图和阈值产物
python code/run_threshold.py --data data/vitaminc_splits_v2/threshold_validation.jsonl

# 阈值产物生成后，检查 test 调用预算并运行
python code/run_final_eval.py --data data/vitaminc_splits_v2/test.jsonl \
  --thresholds code/threshold_results_v2/selected_threshold.json --plan
python code/run_final_eval.py --data data/vitaminc_splits_v2/test.jsonl \
  --thresholds code/threshold_results_v2/selected_threshold.json
```

两个脚本均支持 `--report-only` 从完整检查点离线重建图表、`--model jev|qwen|deepseek|kimi` 分批收集，以及 `--workers`、`--timeout`、`--retries`。单独收集部分模型只生成进度记录，全部模型完成后才生成最终报告。默认输出到独立的 v2 结果目录。

首次请求预算：validation 8,000 次、test 8,000 次，共 **16,000 次**，不含失败重试；不重跑 pilot。阈值扫描、分组验证及画图复用同一批预测，不增加 API 调用。

离线验收、打包和文档回填命令（均不调用模型 API）：

```bash
python -B -m unittest discover -s tests -v
python code/verify_v2.py
python data/package_vitaminc.py
python code/update_v2_readme.py
```

`verify_v2.py` 用独立实现检查数据、阈值选择、每个留出折和最终指标；未完成阶段明确显示 PENDING。`update_v2_readme.py` 仅更新下方标记的结果区块，从实际产物读取状态和分数。

### 3.5 v2 实际进度与结果

<!-- V2_RESULTS_START -->
数据切分已完成并通过复现与隔离检查。

| 子集 | 条数 | S / C / I | 页面数 | 最大页面条数 |
|---|---:|---|---:|---:|
| threshold_validation | 2000 | 996 / 575 / 429 | 277 | 20（1.0%） |
| test | 2000 | 1002 / 575 / 423 | 291 | 20（1.0%） |

[v2 划分报告](data/vitaminc_splits_v2/split_report.json)。

独立数据包：[vitaminc_v2.zip](data/vitaminc_v2.zip)，包含原始数据及 v2 四份切分；[SHA256 清单](data/vitaminc_v2_archive_manifest.json)。旧包继续保留。

```bash
unzip data/vitaminc_v2.zip -d data
```

- validation：已保存 8000/8000 个终态响应；收集完成，累计非终态失败尝试 10 次。
- test：已保存 8000/8000 个终态响应；收集完成，累计非终态失败尝试 11 次。

**validation 实际结果**（完整集最优值与折外分数分开）：

JEV only：80.45%。

| LLM | LLM only | 最终阈值 | 转交条数 | 完整集级联 accuracy | 折外级联 accuracy |
|---|---:|---:|---:|---:|---:|
| qwen3.8-max | 79.45% | 0.43 | 226 | 81.15% | 80.55% |
| deepseek-v4.1-flash-1017 | 74.30% | 0.3 | 91 | 80.65% | 80.15% |
| kimi-k3.1-pro | 75.45% | 0.09 | 6 | 80.55% | 80.45% |

![v2 阈值与净收益](code/threshold_results_v2/threshold_curve.png)

![v2 按连通组的五折验证](code/threshold_results_v2/cross_validation.png)

[固定阈值产物](code/threshold_results_v2/selected_threshold.json)。

**独立 test 实际结果**：

| 方法 | Accuracy | Macro-F1 | 转交条数（比例） | 纠正 / 改错 | 对 JEV 的差值及 95% 区间（百分点） |
|---|---:|---:|---:|---|---|
| jev_only | 83.90% | 81.75% | — | — / — | +0.00 [+0.00, +0.00] |
| llm_only / qwen3.8-max | 81.85% | 80.16% | — | — / — | -2.05 [-3.57, -0.52] |
| cascade / qwen3.8-max | 83.75% | 81.79% | 185（9.25%） | 34 / 37 | -0.15 [-1.05, +0.75] |
| llm_only / deepseek-v4.1-flash-1017 | 77.40% | 74.48% | — | — / — | -6.50 [-8.17, -4.86] |
| cascade / deepseek-v4.1-flash-1017 | 83.00% | 80.80% | 80（4.00%） | 7 / 25 | -0.90 [-1.45, -0.36] |
| llm_only / kimi-k3.1-pro | 77.65% | 76.14% | — | — / — | -6.25 [-8.01, -4.50] |
| cascade / kimi-k3.1-pro | 83.75% | 81.62% | 4（0.20%） | 0 / 3 | -0.15 [-0.34, +0.00] |

服务商内容审查拒绝：qwen3.8-max：1 条。这些样本保留在分母中，单模型计错，若被转交则回退 JEV；本轮三个级联合计实际回退 0 条。未报告 token 用量的请求不假定免费，相应费用估算留空。

![v2 最终评估](code/final_eval_results_v2/final_eval_results.png)

完整成本、延迟及调用量见 [summary.csv](code/final_eval_results_v2/summary.csv)，口径见 [报告元数据](code/final_eval_results_v2/report_metadata.json)。

- JEV + qwen3.8-max：区间包含零，尚不能判定稳定增益或损害。
- JEV + deepseek-v4.1-flash-1017：当前样本的区间支持负收益。
- JEV + kimi-k3.1-pro：区间包含零，尚不能判定稳定增益或损害。

本轮三个固定阈值级联的 test accuracy 均未超过 JEV only。分散页面后仍未观察到稳定的准确率提升，低 confidence 本身不足以确定 LLM 能否带来净纠正。

区间包含零不等于已经证明统计等价或非劣效；若要声称“接近的表现”，应在下一轮预先定义可接受的下降幅度。少量 LLM 调用是相对 LLM only 的场景比较；相对 JEV only，级联增加了调用。

上述判断仅适用于 v2 的页面采样、原标签和固定请求配置；不将 v1/v2 的分数差归因为模型能力变化。
<!-- V2_RESULTS_END -->

### 3.6 后续实验思路

- 在新的开发数据上检查各标签、页面和置信度区间的纠正与改错，重点研究“LLM 相对 JEV 的预期净收益”，而不仅是 JEV 自身是否低置信度。分析后的策略仍需在独立页面上验证。
- 若启用预留 calibration，先单独检验 JEV confidence 与正确率的关系；校准 JEV 的正确概率并不自动解决哪个模型更适合该样本的问题。
- 若研究目标是减少 LLM 调用并保持接近的准确率，先明确比较对象、可接受的下降幅度和调用预算，再用分组验证选择策略。JEV 价格明确前不声称整体费用更低。
- v2 test 已经使用，只作为本轮固定策略的最终评估。后续修改提示词、模型、校准或路由规则，应重新安排开发数据和未评估页面；不依据此 test 的表现回调阈值。
