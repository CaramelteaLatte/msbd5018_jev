# RAGTruth QA：真实生成回答的幻觉检测

本实验检验 JEV 对真实生成回答的证据一致性判断，以及低置信度转交给 LLM 是否可能带来收益。输入材料和候选回答均来自 RAGTruth；本轮不实现自己的检索器，也不让核验模型重新生成答案。

来源：[官方仓库](https://github.com/ParticleMedia/RAGTruth)、[ACL 2024 论文](https://aclanthology.org/2024.acl-long.585/)。固定数据版本为 `c103204b9ce28d6bbad859304bf30de72b8ed8fe`，下载后逐文件验证 SHA256，保存上游 MIT 许可证。数据可用脚本重新下载，展开的数据及逐次请求日志不进入 Git。

## 1. 任务和标注口径

一条评估样本是 `question + passages + candidate_answer`。回答可能包含多条事实，也可能同时包含矛盾和无依据的信息，因此使用整段回答的二分类，而不是将原始标注强行转成 VitaminC 的 S/C/I：

- **H**：至少存在一处人工标注的幻觉片段，包含与材料矛盾、缺乏材料支持两类。
- **N**：不存在人工标注的幻觉片段；这不等于证明回答有用、完整或符合外部世界的事实。

`implicit_true` 表示内容可能真实但未被参考材料支持。主实验遵循官方严格的证据一致性口径，将这些片段包含在 H 中；同时用完全相同的预测，另报排除此类片段后的敏感性评分。这是评分定义的敏感性分析，不是重新提示模型得到的另一套结果。

主集合采用官方基线脚本的 `quality=good` 条件。QA 原始数据有 989 个问题、5,934 段回答；其中 143 段 `incorrect_refusal`、24 段 `truncated` 进入 `excluded_quality.jsonl`，不被改写为无幻觉标签。5,767 段进入主集合。所有幻觉片段的字符偏移均已验证对应原回答文字。

核验模型只接收三个输入字段。人工标签、片段位置、原回答生成模型和 `quality` 均不进入请求。JEV 的输出为 H/N、对应概率和 confidence；三个 LLM 仅输出 H/N，使用原先模型标识、`enable_thinking=false`、`max_completion_tokens=32`。本实验不会计算未预测过的片段定位 F1。

## 2. 来源隔离与试跑

使用 seed `20261112`，在官方 train 的问题之间切分；同一 `source_id` 的不同模型回答整体分配到同一个集合。额外检查规范化问题及证据的重复；出现重复时停止，不能直接按独立问题切分。官方 test 专用于固定策略的独立评估。

| 集合 | 独立问题数 | 回答数 | H / N | 本轮用途 |
|---|---:|---:|---:|---|
| prompt_dev | 48 | 277 | 96 / 181 | 提示词与接口开发 |
| pilot（prompt_dev 的子集） | 48 | 48 | 17 / 31 | 四个核验模型真实试跑 |
| validation | 200 | 1,157 | 367 / 790 | 正式阈值选择与五折验证 |
| test | 150 | 875 | 160 / 715 | 官方 test 的 good 子集，独立最终评估 |
| train_reserved | 591 | 3,458 | 1,083 / 2,375 | 预留，未训练或调用 |

pilot 每个问题只选择一个回答，根据生成模型的已有入选数量进行平衡，不查看幻觉标签或任何核验预测。此次六种生成模型各 8 段。多个生成模型提供的回答不是核验模型本轮生成的。

pilot 的首次请求为 `48 × 4 = 192`，全程最多 200 次请求尝试（含已记录失败），默认四并发，无自动重试。网络错误停止提交新请求并保存已在途结果，重跑相同命令跳过已有终态记录，且仍检查总次数上限。服务商明确内容拒绝保留为失败终态；单模型评分计错，探索性级联遇到无效 LLM 预测回退 JEV。无效 JEV 响应使收集停止。

## 3. 运行与复现

从仓库根目录执行；依赖 Python 3.9+、NumPy 和 Matplotlib。

```bash
# 下载固定版本并切分；空目录或克隆后仅含 split_report.json 的目录可用
python data/prepare_ragtruth.py

# 只检查输入和预算，不调用模型
python code/run_ragtruth_pilot.py --data data/ragtruth_qa/pilot.jsonl --plan

# 读取本地密钥，实际收集四个核验模型的输出
set -a
source .env
set +a
python code/run_ragtruth_pilot.py --data data/ragtruth_qa/pilot.jsonl

# 使用完整检查点离线重建，不增加模型调用
python code/run_ragtruth_pilot.py --data data/ragtruth_qa/pilot.jsonl --report-only

# 独立复算数据和逐条预测
python code/verify_ragtruth.py
python -B -m unittest discover -s tests -v
```

支持 `--model jev|qwen|deepseek|kimi` 分批收集、`--out` 指定输出目录、`--workers` 调整并发。200 次上限针对同一输出目录；不要通过新建目录重复执行同一试跑。数据或请求配置不匹配时拒绝混用旧检查点。

重新克隆后，数据准备脚本会核对已跟踪的 `split_report.json`，仅在参数和生成结果完全一致时恢复展开的数据；已有展开文件的目录不会覆盖，重复切分请使用新的 `--out`。请求检查点仅保存在本机，克隆仓库不会获得缓存，因此在另一台机器执行收集命令会产生新的 API 请求。

## 4. 报告内容及解释

pilot 已完成 192 次真实 API 请求，无失败或重试。实际指标与图表见 [pilot 报告](../code/ragtruth_pilot_results/pilot_report.md)。独立复算通过。正式阶段流程见第 5、6 节。

结果写入 `code/ragtruth_pilot_results/`：

- `summary.csv`、`metrics.json`：四个模型的 accuracy、Macro-F1、H 类 precision/recall/F1、混淆矩阵、无效输出数量、token 和请求耗时。
- `majority_baseline.json`：全部预测 N 的基线。本批可得到 64.58% accuracy，却不能检出任何 H；因此正式实验优先考察 H 类 F1。
- `by_generator.csv`：按原回答生成模型分组；每组仅 8 条，只用于检查明显异常。
- `sensitivity_without_implicit_true.csv`：改变上述标注口径后的敏感性评分。
- `exploratory_routing.csv`、`pilot_results.png`：完整同置信度组、转交最多 20% 的探索性结果，展示纠正/改错；不从 pilot 发布正式阈值。
- 本地 `predictions.jsonl`、`disagreements.jsonl` 和 SQLite：逐条复算、人工检查及断点续跑。

原始证据与回答不截断。模型实际返回的 token 用量单独记录，不用字符数冒充 token。费用字段沿用明确标记的历史价格假设，不是当前账单；JEV 单价未知，不给出完整总费用。报告中耗时是并发收集时的终态请求耗时，不含重试与排队。

这批 48 个问题只验证接入、标签和探索性行为；不据此声明模型能力排名、最优阈值或正式测试增益。核验输出不会自动修正原回答；转交改善的是幻觉判断。

## 5. 正式实验协议

1. 已复核 pilot 的误报、漏报和 `implicit_true` 敏感案例，保留现有提示词和原始标签；[案例笔记](ragtruth_pilot_review.md)仅为助手审阅，不是新的人工作业标注。[冻结协议](../code/ragtruth_protocol.json)保存完整请求配置、输入哈希、评分规则及审阅文件哈希；变化时拒绝混用检查点。
2. 四模型预测完整 validation 1,157 段。主选择指标预定为 **H 类 F1**，并列时先比较 accuracy，再优先较少 LLM 调用；保留不转交候选。同 confidence 整组处理，选择阶段转交率最多 20%。按 `source_id` 做五折阈值验证。
3. 从完整 validation 固定各 LLM 的阈值，生成含配置及数据哈希的产物。不可直接使用 VitaminC 的 0.48 或 v2 阈值。
4. 四模型各预测独立 test 875 段，报告准确率、H 类 precision/recall/F1、实际转交率、纠正/改错、无效输出和成本/延迟。按问题进行 10,000 次配对 bootstrap；同一问题的多个回答一起重采样。
5. 如需评价完整 RAG，再接入固定语料库的检索和回答生成，并为新回答安排独立人工标注；本实验不承担检索效果评估。

正式 validation + test 已完成 `(1157 + 875) × 4 = 8,128` 个模型/样本结果；validation 无失败重试，test 另有 50 次连接失败尝试，共 8,178 次请求尝试，pilot 未重复运行。阈值扫描、五折分析、bootstrap、敏感性评分及画图均离线复用。原始 confidence 候选用散点表示，不做拟合。[正式验证报告](../code/ragtruth_validation_results/validation_report.md)、[独立测试报告](../code/ragtruth_final_results/final_report.md)及 README 第 4.5 节已写入实际结果；数据、评分、阈值与 bootstrap 独立复算通过。

五折以 source_id 整组分配，固定 seed，平衡回答数与 H/N 数量。每次只在四折上按 H F1、accuracy、较少调用选阈值，在留出折评估；总体折外指标由各条留出预测合并计算，不简单平均五个 F1。最后用完整 validation 选定部署阈值，单独展示该集合的表观最优分数。无转交用 JSON null 表示，confidence=0 仍是实际数值阈值。

bootstrap 以问题为单位配对抽样，同一问题的多个回答整体保留；每次重新计算响应加权的 accuracy 和 H F1，报告差值的百分位 95% 区间。区间不覆盖原标注噪声或重新请求模型的随机性，也不是多模型比较的校正区间。区间包含零不能当作等价或非劣效证明。按原回答生成模型的分组统计仅作诊断。

## 6. 正式运行、复算与产物

从仓库根目录运行，先按第 3 节加载本地环境变量。默认每模型 8 并发；本轮正式收集指定 16 并发，并写入 `collection_runs.json`。默认每次运行对瞬时连接错误、408、429 或 5xx 最多重试两次；其他错误停止新增请求并保存已在途结果。服务商明确内容拒绝及无效 LLM 标签计为终态：单模型计错，级联回退 JEV。无效 JEV 停止实验。

```bash
# 只检查预算，无 API 请求
python code/run_ragtruth.py --stage validation --data data/ragtruth_qa/validation.jsonl --plan

# 正式 validation：收集四模型、生成五折分析和固定阈值
python code/run_ragtruth.py --stage validation --data data/ragtruth_qa/validation.jsonl --workers 16
python code/verify_ragtruth_formal.py --stage validation

# 正式 test：必须传入上述阈值文件；其 SHA256 写入运行配置
python code/run_ragtruth.py --stage test --data data/ragtruth_qa/test.jsonl \
  --thresholds code/ragtruth_validation_results/selected_threshold.json --plan
python code/run_ragtruth.py --stage test --data data/ragtruth_qa/test.jsonl \
  --thresholds code/ragtruth_validation_results/selected_threshold.json --workers 16

# 从完整缓存重建，无 API 请求；validation 同样支持 --report-only
python code/run_ragtruth.py --stage test --data data/ragtruth_qa/test.jsonl \
  --thresholds code/ragtruth_validation_results/selected_threshold.json --report-only
python code/verify_ragtruth_formal.py
python code/update_ragtruth_readme.py
```

正式脚本也支持 `--model` 分模型运行；全部四模型完成前不生成最终阈值或结果报告。数据、协议或测试阈值文件变化时拒绝续跑；已有终态不重复请求。`--plan` 显示缓存和剩余请求数。克隆仓库不会获得本地 SQLite 缓存，在另一台机器执行收集会产生新请求。

- `code/ragtruth_validation_results/`：`selected_threshold.json`、`threshold_sweep.csv`、`folds.json`、`cross_validation.csv`、`oof_metrics.json`、阈值散点图及五折图。
- `code/ragtruth_final_results/`：七种方案的 `summary.csv`、`metrics.json`、`bootstrap.json`、`by_generator.csv`、`sensitivity_without_implicit_true.csv`、最终结果图及混淆矩阵图。
- 两阶段均保存调用记录、配置和数据哈希，`attempt_summary.json` 汇总失败类型及无效终态；逐条预测 JSONL、原始请求日志和 SQLite 仅保存在本机。`verify_ragtruth_formal.py` 从逐条响应独立复算评分、阈值最优选择、留出折与 bootstrap 区间。

级联调用、成本和延迟是基于完整预测重建的部署场景。本轮实际调用了所有模型，不能把级联的低转交率解释为此次实验已节省的请求。JEV 单价未知时完整费用为空；缺少 token 用量也不会假定免费。累计实际失败/重试见 progress 和本地 attempts；表格中的 token、费用与延迟按终态请求统计，不含重试、排队和编排。
