#!/usr/bin/env python3
"""Publish local, completed RAGTruth aggregates into README; never sends requests."""
import csv
import json
from pathlib import Path
import experiment_v2 as shared

ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'code/ragtruth_validation_results'
T=ROOT/'code/ragtruth_final_results'
START='<!-- RAGTRUTH_RESULTS_START -->'
END='<!-- RAGTRUTH_RESULTS_END -->'

THRESHOLD_GUIDE = """**这张图怎么看（RAGTruth validation，1,157 段回答）：** 三列从左到右对应 Qwen、DeepSeek、Kimi。九个小图的横轴都是尝试的 confidence 阈值；一个点表示采用该阈值后的整体结果，不是一条回答。confidence 是离散值，所以没有拟合或连接这些散点。

| 位置 | 纵轴与图形的含义 | 读图重点 |
|---|---|---|
| 第一行 | 分数（%）。蓝色圆点是级联 H F1，橙色三角是级联 accuracy；灰色横虚线表示 JEV only，彩色横点线表示该 LLM only，两种指标分别在图例中标明 | 同一种指标要与自己的基线比较。选阈值主要看 H F1，不是挑最高的 accuracy 点 |
| 第二行 | 被转交回答占全部验证回答的比例（%）；横虚线是 20% 选择预算 | 在同一个阈值下，是否转交只由 JEV confidence 决定，所以三列的转交点位置相同 |
| 第三行 | 回答条数。蓝色圆点是纠正，橙色叉号是改错，绿色三角是净变化；零线表示正确总数没变化 | 纠正多于改错，accuracy 才提高；H F1 还取决于改动的是 H 还是 N，不能只凭净变化选 F1 最优阈值 |

每列标题标明最终选中的策略。`no routing` 表示完全不转交，不能把它当作横轴上的 0；阈值等于 0 时，仍会转交 confidence 恰好等于 0 的回答。上方表格是完整验证集选阈值后的分数，接下来还要看五折和独立 test。
"""

CV_GUIDE = """**这张五折图怎么看：** 三幅图的横轴都是留出折编号 1～5；每折有 40 个独立问题、231～232 段回答。蓝、橙、绿分别表示 Qwen、DeepSeek、Kimi。为了避免点重叠，三个模型的点在同一折附近稍微左右错开，不代表小数编号的折。

- **左图纵轴：** 在该份留出题上，级联的 H F1 减去 JEV 的 H F1，单位是百分点。高于 0 表示这一折检出效果有改善，低于 0 表示下降。
- **中图纵轴：** 同一批留出题的 accuracy 差值，也是百分点。F1 与 accuracy 的变化不一定同方向，要分开读。
- **右图纵轴：** 从另外四折选出的 confidence 阈值。标为 None 的 −0.04 位置只用于摆放“不转交”的点，不是真实负阈值。

每个点都在自己没参与选阈值的那份题上评分。各折既有正收益也有负收益，说明收益会随问题来源变化；不能只挑表现最好的一折。表格中的折外 F1 是合并全部留出预测后重算的，不是把五个 F1 简单平均。
"""

FINAL_GUIDE = """**这张图怎么看（RAGTruth 独立 test，875 段回答）：**

| 位置 | 横轴 | 纵轴、点与线的含义 | 读图重点 |
|---|---|---|---|
| 左上 | 七种方案，JEV、三个 LLM only 和三个级联 | 分数（%）；蓝柱是 accuracy，橙柱是 H F1 | 同色柱子之间比较，越高越好。高 accuracy 不一定代表能找出幻觉，所以同时看 H F1 |
| 右上 | H F1 相对 JEV 的差值（百分点） | 六种对比方案；圆点是本次差值，横线是 95% bootstrap 区间 | 竖虚线 0 表示与 JEV 一样；右侧为提高，左侧为降低；区间跨 0 时，本轮不足以确定稳定差异 |
| 左下 | accuracy 相对 JEV 的差值（百分点） | 与右上相同的六种方案、圆点和区间 | 比较的是准确率，不要把这里的区间当作 H F1 区间。横线是重采样区间，不是拟合线 |
| 右下 | 三个级联；标签同时列出实际转交条数和比例 | 发生变化的回答条数；蓝柱是纠正，橙柱是改错 | 蓝柱比橙柱高才是多答对题。两柱相加不一定等于转交总数，因为很多回答转交前后都对或都错 |

Qwen 级联的点与区间都落在 0，是因为验证阶段已经选中完全不转交，实际预测逐条等于 JEV；不是额外测试证明了两个不同模型等价。测试只应用固定策略，不在这张图上反过来挑新阈值。
"""

CONFUSION_GUIDE = """**这张混淆矩阵怎么看：** 七个小图分别对应七种方案，最后两个空位只是排版预留，不表示漏跑模型。每个小图的**横轴是模型预测**：H（有幻觉）、N（无幻觉）、Invalid（无效输出）；**纵轴是原数据标签**：H 或 N。每个格子的数字是回答条数，不是百分比。

| 格子位置 | 含义 |
|---|---|
| H 行、H 列 | 正确找出的幻觉 |
| H 行、N 列 | 有幻觉却没发现，即漏报 |
| N 行、H 列 | 原标注没有幻觉却被报为有，即误报 |
| N 行、N 列 | 正确判断为没有幻觉 |
| 任一行、Invalid 列 | 没有给出合法 H/N，单模型评分计错；级联按规则回退 JEV |

每幅图所有格子相加为 875；H 行合计 160，N 行合计 715。颜色越深表示该小图中数量越多，但不同小图分别缩放颜色，因此跨模型比较时要读数字，不能只比较深浅。理想情况是 H/H 与 N/N 格子大，误报、漏报和 Invalid 格子小。
"""


def read(path): return json.loads(path.read_text())

def cutoff(value): return '不转交（null）' if value is None else f'{value:g}'

def interval(value):
    return f"{value['difference']*100:+.2f} [{value['ci95_low']*100:+.2f}, {value['ci95_high']*100:+.2f}]"


def cost_section():
    from plot_ragtruth_costs import NAMES, load_report, cost_range
    rows, metadata = load_report(T, pricing='historical')
    billing = metadata['jev_billing']
    n = metadata['sample_count']
    per_call = metadata['jev_effective_average_usd_per_request']
    lines = [
        '**JEV、纯 LLM 和级联的开销对比（历史单价假设，美元估算）**', '',
        f'本节比较 RAGTruth 独立 test 的同一批 {n} 条回答，固定验证阶段已选择的阈值。LLM 费用使用记录的实际 token 数乘以原实验保存的历史美元单价；JEV 使用用户提供的 ${billing["reported_total_cost_usd"]:.3f} / {billing["reported_requests"]:,} 次平均分摊。这是一项价格情景分析，不是实际账单节省结论。', '',
        f'JEV 平均每次约 ${per_call:.8f}；875 次 test 请求约 ${per_call*n:.5f}。所有级联都包含这 875 次 JEV。JEV 分摊不是已验证的固定每次报价；LLM 历史单价见 [report_metadata.json](code/ragtruth_final_results/report_metadata.json)，DeepSeek 保留两种单价假设。', '',
        '![RAGTruth QA: historical-price USD scenario, calls and latency](code/ragtruth_final_results/cost_comparison.png)', '',
        '**这张图怎么看：** 三幅图的每一行对应同一个方案；横轴均从 0 开始。', '',
        '| 小图 | 横轴、柱子与标记 | 如何理解 |', '|---|---|---|',
        f'| A，左侧：总费用估算 | 处理全部 {n} 条的费用（美元）；灰色为 JEV 分摊，蓝色为 LLM 估算，两段相加 | 越短表示在这些价格假设下越便宜。DeepSeek 的横短线是价格假设范围，不是统计置信区间；不是实付费用 |',
        f'| B，中间：调用次数 | 处理全部 {n} 条的终态请求数；灰色为 JEV，蓝色为 LLM，右侧数字为总数 | 级联先全部调用 JEV，再追加被转交的 LLM 请求；减少 LLM 次数不一定减少总请求数 |',
        '| C，右侧：每条回答耗时 | 秒；蓝柱为平均值，橙色菱形为 P95，即约 95% 回答不超过的耗时 | 级联按 JEV 加上被转交 LLM 的请求耗时重建，不含重试、排队和编排；不是整轮实验运行时间 |', '',
        '| 方案 | JEV 调用 | LLM 调用 | JEV 分摊（美元） | LLM 估算（美元） | 总费用估算（美元） |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {NAMES[row['scenario']]} | {row['jev_calls']} | {row['llm_calls']} | ${row['jev_cost']:.5f} | {cost_range(row, 'llm_cost_')} | {cost_range(row)} |")
    lines += ['', '**相对对应纯 LLM，在相同历史价格假设下：**', '',
              '| 级联 | 估算总费用降低 | LLM 调用减少 |', '|---|---:|---:|']
    for row in rows:
        if row['scenario'].startswith('cascade/'):
            low, high = row['total_cost_reduction_low_percent'], row['total_cost_reduction_high_percent']
            reduction = f'{low:.2f}%' if low == high else f'{low:.2f}%～{high:.2f}%'
            lines.append(f"| {NAMES[row['scenario']]} | {reduction} | {1-int(row['llm_calls'])/n:.2%} |")
    lines += ['', 'DeepSeek 降幅分别按低价与低价、高价与高价比较，不混用价格端点。Qwen 完全不转交，实际就是 JEV only；其较低估算费用不能当作非零转交策略有效的证据。', '',
        '**本节结论仅在上述定价假设下成立。** 成本降低的比较对象是对应纯 LLM；与纯 JEV 比，额外转交仍增加费用。为获得单模型基线，本轮实际全量请求四个模型；这里是固定策略的部署场景重建，不是本次收集过程已经实现的费用节省。LLM 历史估算排除失败重试，JEV 报告总额中的重试计费无法单独拆分。', '',
        '美元结果见 [cost_comparison.csv](code/ragtruth_final_results/cost_comparison.csv)，假设与来源哈希见 [cost_comparison_metadata.json](code/ragtruth_final_results/cost_comparison_metadata.json)。实际账单复核另存于 [复核记录](docs/ragtruth_billing_reconciliation.md)，其成本结论与本节价格情景不同，不将两者混用。', '',
        '从仓库根目录重画（默认历史美元单价，无新增 API 请求）：', '', '```bash', 'python code/plot_ragtruth_costs.py', '```', '']
    return lines


def tradeoff_section():
    path = T/'tradeoff_comparisons.csv'
    metadata = read(T/'tradeoff_metadata.json')
    if metadata['source_sha256']['summary.csv'] != shared.digest(T/'summary.csv'):
        raise ValueError('Tradeoff analysis is stale; run analyze_ragtruth_tradeoffs.py')
    rows = list(csv.DictReader(path.open()))
    lines = ['**历史单价假设下，成本、准确率和延迟如何取舍？**', '',
        '以上美元情景显示级联可以降低估算成本；效果与延迟仍使用原实验记录，必须相对同一个基线比较。不能用相对纯 JEV 的较小分数差，搭配相对纯 LLM 的费用下降，来构造同一个收益结论。', '',
        '以下统一与**对应纯 LLM**比较，差值为级联减去纯 LLM；正的 accuracy 差值更好，负的延迟变化表示更快。', '',
        '| 级联 | Accuracy：纯 LLM → 级联 | Accuracy 差值 [95% 区间]（百分点） | 平均延迟：纯 LLM → 级联（秒） | P95：纯 LLM → 级联（秒） |',
        '|---|---:|---|---|---|']
    for r in rows:
        if not r['baseline'].startswith('llm_only/'):
            continue
        values = {k:float(v) for k,v in r.items() if k not in ('cascade','baseline')}
        v = values
        delta = f"{v['accuracy_difference_pp']:+.2f} [{v['accuracy_ci95_low_pp']:+.2f}, {v['accuracy_ci95_high_pp']:+.2f}]"
        mean = f"{v['mean_latency_ms_baseline']/1000:.3f} → {v['mean_latency_ms_cascade']/1000:.3f}（{v['mean_latency_ms_change_percent']:+.2f}%）"
        tail = f"{v['p95_latency_ms_baseline']/1000:.3f} → {v['p95_latency_ms_cascade']/1000:.3f}（{v['p95_latency_ms_change_percent']:+.2f}%）"
        lines.append(f"| {r['cascade']} | {v['accuracy_baseline']:.2%} → {v['accuracy_cascade']:.2%} | {delta} | {mean} | {tail} |")
    lines += ['', '区间是本次补充分析：固定原预测与阈值，对 150 个 source_id 做 10,000 次配对 bootstrap（seed 20261112），直接比较每种级联与对应 LLM；不是把已有的两个相对 JEV 区间相减。区间未做多重比较校正。延迟仅报告记录重建值，未重新实跑级联或计算延迟置信区间。', '',
        '- JEV + Qwen：不转交，实际就是 JEV；不能作为非零转交策略有效的证据。',
        '- JEV + DeepSeek：accuracy 下降 4.46 个百分点，平均延迟只减少 1.29%，P95 增加 16.37%。',
        '- JEV + Kimi：accuracy 下降 2.86 个百分点，平均延迟减少 40.77%，P95 减少 10.66%；在该历史价格情景下估算总成本降低 73.35%，体现了成本、速度与准确率的取舍。', '',
        'DeepSeek、Kimi 相对纯 LLM 的 accuracy 差值区间均低于 0。H F1 也分别下降 3.25 和 2.14 个百分点，对应 95% 区间为 [-5.26, -1.27]、[-3.59, -0.76]。不能把这些差值说成没有下降；“下降不太多”需要预先定义业务可接受界限，本轮没有预设，不能事后宣称非劣效。', '',
        '若改与纯 JEV 比，DeepSeek、Kimi 级联的 accuracy 分别只下降 0.80、0.34 个百分点，但平均延迟分别增加 17.66%、66.67%，总成本也增加；这同样不是全面改善。', '',
        '**可报告的结论：** 在本次 RAGTruth QA 独立测试中，按历史 LLM 单价与 JEV 平均费用分摊估算，JEV + Kimi 相对纯 Kimi 的总成本降低 73.35%，重建的平均请求延迟降低 40.77%，P95 降低 10.66%；accuracy 从 70.51% 降至 67.66%，损失 2.86 个百分点。这表明在该定价情景下，可以用明确的准确率损失换取较低的估算成本与延迟。它不是实际账单省钱或准确率无损的证明，也不能外推成所有级联都能同时改善成本与延迟。', '',
        '后续若研究“以较少损失换取成本下降”，应先固定同一个比较对象、可接受的 accuracy/H F1 下降幅度与延迟要求，再用开发数据选择满足这些约束的策略，并用新的独立测试验证。当前 test 不用于反向挑选支持结论的新阈值。', '',
        '[直接配对比较](code/ragtruth_final_results/tradeoff_comparisons.csv) · [统计口径](code/ragtruth_final_results/tradeoff_metadata.json)。从仓库根目录复算：`python code/analyze_ragtruth_tradeoffs.py`（需要本地 predictions.jsonl；不调用 API）。', '']
    return lines


def main():
    vp,tp=read(V/'progress.json'),read(T/'progress.json')
    if not (vp['complete'] and tp['complete']):
        raise ValueError('Both formal stages must be complete before publishing final results')
    artifact=read(V/'selected_threshold.json')
    vc,tc=read(V/'run_config.json'),read(T/'run_config.json')
    if tc['thresholds_sha256']!=shared.digest(V/'selected_threshold.json'):
        raise ValueError('Test used a different threshold artifact')
    vm,tm,oof=read(V/'metrics.json'),read(T/'metrics.json'),read(V/'oof_metrics.json')
    boot=read(T/'bootstrap.json')
    summaries={r['scenario']:r for r in csv.DictReader((T/'summary.csv').open())}
    validation=[
        '正式 validation：200 个问题、1,157 段回答（H=367、N=790），四模型共 4,628 个终态结果。阈值以 H 类 F1 最大为目标；同分依次比较 accuracy、较少调用。完整 validation 上选择后的分数与五折折外分数分别报告。', '',
        f"JEV only：accuracy **{vm['jev_only']['accuracy']:.2%}**，H F1 **{vm['jev_only']['hallucination_f1']:.2%}**。", '',
        '| LLM | LLM only H F1 | 最终阈值 | 验证集转交 | 完整集级联 H F1 | 折外级联 H F1 | 完整集 / 折外 accuracy |',
        '|---|---:|---|---:|---:|---:|---:|']
    for alias,model in shared.LLMS.items():
        chosen=artifact['models'][model]
        full=vm['cascade/'+alias]
        validation.append(f"| {alias} | {vm['llm_only/'+alias]['hallucination_f1']:.2%} | {cutoff(chosen['confidence_threshold'])} | {chosen['routed']}/1157（{chosen['routing_rate']:.2%}） | {full['hallucination_f1']:.2%} | {oof[model]['hallucination_f1']:.2%} | {full['accuracy']:.2%} / {oof[model]['accuracy']:.2%} |")
    validation += ['', '[固定阈值](code/ragtruth_validation_results/selected_threshold.json) · [验证指标](code/ragtruth_validation_results/summary.csv) · [逐折结果](code/ragtruth_validation_results/cross_validation.csv)', '',
        '![RAGTruth 离散阈值、转交率与纠正改错](code/ragtruth_validation_results/threshold_curve.png)', '', THRESHOLD_GUIDE.strip(), '',
        '![RAGTruth 按问题五折验证](code/ragtruth_validation_results/cross_validation.png)', '', CV_GUIDE.strip()]
    # Explain the null policy from actual sweep rows rather than treating null as zero.
    if artifact['models'][shared.LLMS['qwen']]['confidence_threshold'] is None:
        points=[r for r in csv.DictReader((V/'threshold_sweep.csv').open()) if r['model']==shared.LLMS['qwen']]
        numeric=[r for r in points if r['confidence_threshold']!='']
        best=max(numeric,key=lambda r:(float(r['hallucination_f1']),float(r['accuracy']),-int(r['routed']))) if numeric else None
        validation += ['', '**为什么 JEV + Qwen 不转交？** Qwen 已预测全部验证样本，我们用缓存逐个比较允许的转交阈值，也把“不转交”作为候选。选择规则先看 H F1，再看 accuracy，最后优先少调用。']
        if best:
            validation += ['', '| Qwen 的验证策略 | 转交条数 | H F1 | 纠正 / 改错 |', '|---|---:|---:|---:|',
                f"| 不转交 | 0 | {vm['jev_only']['hallucination_f1']:.2%} | 0 / 0 |",
                f"| 最好的非零转交方案：confidence ≤ {float(best['confidence_threshold']):g} | {best['routed']} | {float(best['hallucination_f1']):.2%} | {best['corrected']} / {best['harmed']} |"]
            if all(float(r['hallucination_f1'])<vm['jev_only']['hallucination_f1'] for r in numeric):
                validation += ['', f"本轮 {len(numeric)} 个符合 20% 预算的数值阈值，H F1 都低于不转交。因此选择 null，并在 test 保持这个决定。不是漏调 Qwen，也不是必须用满 20% 预算。"]
    testing=[
        '独立 test：150 个问题、875 段回答（H=160、N=715），四模型共 3,500 个终态结果。使用完整 validation 固定的阈值，未按测试集重新选择。H 是幻觉阳性类；下列区间来自 10,000 次按 source_id 配对 bootstrap，单位为百分点。', '',
        '| 方法 | Accuracy | Macro-F1 | H Precision | H Recall | H F1 | H F1 相对 JEV 差值 [95% 区间] |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name,row in tm.items():
        testing.append(f"| {name} | {row['accuracy']:.2%} | {row['macro_f1']:.2%} | {row['hallucination_precision']:.2%} | {row['hallucination_recall']:.2%} | {row['hallucination_f1']:.2%} | {interval(boot[name]['hallucination_f1'])} |")
    majority=read(T/'majority_baseline.json')
    testing += ['', f"全部预测 N 的 test accuracy 为 {majority['accuracy']:.2%}，H F1 为 0；因此不能仅凭 accuracy 评价幻觉检出效果。验证与测试的 H 比例不同，分数变化不能直接解释为方法变化。", '',
                '单模型无效输出：'+ '；'.join(f"{name}：{row['invalid_predictions']} 条" for name,row in tm.items() if not name.startswith('cascade/'))+'。无效输出保留在评分分母中。']
    testing += ['', '| 级联 | test 转交 | 纠正 / 改错 | 无效 LLM 回退 | Accuracy 相对 JEV 差值 [95% 区间] |',
                '|---|---:|---:|---:|---|']
    for alias in shared.LLMS:
        name='cascade/'+alias
        row=summaries[name]
        testing.append(f"| JEV + {alias} | {row['routed']}/875（{float(row['routing_rate']):.2%}） | {row['corrected']} / {row['harmed']} | {row['fallbacks']} | {interval(boot[name]['accuracy'])} |")
    testing += ['', '![RAGTruth 独立测试指标与区间](code/ragtruth_final_results/final_results.png)', '', FINAL_GUIDE.strip(), '',
                '![RAGTruth 七种方案的混淆矩阵](code/ragtruth_final_results/confusion_matrices.png)', '', CONFUSION_GUIDE.strip(), '',
                '[混淆矩阵图](code/ragtruth_final_results/confusion_matrices.png) · [完整指标、调用与成本](code/ragtruth_final_results/summary.csv) · [bootstrap](code/ragtruth_final_results/bootstrap.json) · [按原回答生成模型分组](code/ragtruth_final_results/by_generator.csv) · [排除 implicit_true 的敏感性评分](code/ragtruth_final_results/sensitivity_without_implicit_true.csv)', '']
    testing += cost_section()
    testing += tradeoff_section()
    sensitivity={r['scenario']:r for r in csv.DictReader((T/'sensitivity_without_implicit_true.csv').open())}
    testing += ['**如果不把 implicit_true 算作幻觉，结果会怎样？** 这项检查保留同一批预测和固定阈值，只改变评分标签：某段回答如果只有“材料没写、但可能真实”的片段，就从 H 改按 N 评分。test 的 H 从 160 段变为 134 段，回答总数仍为 875。', '',
                '| 方案 | 主口径 H F1 | 排除 implicit_true 后的 H F1 |', '|---|---:|---:|']
    for name,row in tm.items():
        testing.append(f"| {name} | {row['hallucination_f1']:.2%} | {float(sensitivity[name]['hallucination_f1']):.2%} |")
    testing += ['', '这些分数评的是不同标签定义，不能把下降说成模型突然变差。本轮排除 implicit_true 后，过多报 H 的问题仍在；这不是重新训练或重新写提示词后得到的效果。', '', '**本轮结论：**', '']
    for alias in shared.LLMS:
        ci=boot['cascade/'+alias]['hallucination_f1']
        model=shared.LLMS[alias]
        conclusion=('验证阶段选择不转交，实际策略与 JEV only 完全相同，差值及区间为零' if artifact['models'][model]['confidence_threshold'] is None else
                    '区间支持 H F1 正收益' if ci['ci95_low']>0 else '区间支持 H F1 负收益' if ci['ci95_high']<0 else '区间包含零，尚不能判定 H F1 稳定改善或下降')
        testing.append(f'- JEV + {alias}：{conclusion}。')
    if all(tm['cascade/'+a]['hallucination_f1']<=tm['jev_only']['hallucination_f1'] for a in shared.LLMS):
        testing += ['', '**本轮未观察到固定阈值级联提高测试集 H 类 F1。** 单模型整体表现较好，也不保证在 JEV 的低置信度子集上更好。']
    j=tm['jev_only']
    testing += ['', f"JEV 在 test 上误报 {j['false_positive']} 条、漏报 {j['false_negative']} 条，H recall 为 {j['hallucination_recall']:.2%}、precision 为 {j['hallucination_precision']:.2%}。后续应在开发数据上复核误报与标注边界，并研究 LLM 相对 JEV 的净纠正能力。此 test 已用于最终评估，后续改策略需要新的独立评估数据。"]
    over=[f"{a} 为 {float(summaries['cascade/'+a]['routing_rate']):.2%}" for a in shared.LLMS if float(summaries['cascade/'+a]['routing_rate'])>.2]
    if over:
        testing += ['', '测试集实际转交超过验证阶段的 20% 选择预算：'+ '；'.join(over)+'。这反映固定 confidence 阈值在不同样本分布下的转交率变化，本轮未按 test 重新压低阈值。']
    invalid=read(T/'attempt_summary.json')['invalid_terminal']
    if invalid:
        testing += ['', '无效终态详情见 [attempt_summary.json](code/ragtruth_final_results/attempt_summary.json)。', '']
        for r in invalid:
            testing.append(f"- {r['model']} / sample {r['sample_id']}：未返回合法 H/N；finish_reason={r['finish_reason']}，output_tokens={r['output_tokens']}。单模型计错，被转交时回退 JEV；未为该条改变配置或重新请求。")
    testing += ['', '区间包含零不等于证明等价或非劣效。上述结论以当前 QA 样本和原人工标注为条件；模型误报可能涉及推断边界或标注覆盖疑点，不能仅凭模型一致判错就修改 gold。敏感性分析仅改变评分标签，不重调提示词或阈值。', '',
                '实验为获得单模型基线而全量调用四个模型；级联调用量、费用和延迟由缓存离线重建，表示固定策略的部署场景。主图采用历史 LLM 美元单价和 JEV 平均请求费用分摊，总费用情景见 cost_comparison.csv；这是条件性的成本估算，不是实际支出结论。延迟按终态请求累计，不含重试、排队和编排。验证阶段 20% 是选择约束，独立测试如实报告实际转交率。']
    budget=[f"正式 validation/test 已完成 **8,128 个终态结果**；另记录验证阶段 {vp['nonterminal_attempts']} 次、测试阶段 {tp['nonterminal_attempts']} 次非终态尝试。pilot 的 192 次请求未重复运行。收集并发与重试设置见各目录的 `collection_runs.json`，失败类型和无效终态见 `attempt_summary.json`。", '',
            '[冻结协议](code/ragtruth_protocol.json) · [pilot 案例审阅](docs/ragtruth_pilot_review.md)。两阶段数据、配置与固定阈值通过哈希关联；代码可从逐条本地检查点独立复算结果。']
    block='\n'.join([START,*budget,'',*validation,'',*testing,END])
    path=ROOT/'readme.md'
    text=path.read_text()
    old='pilot 为 48 个开发问题、48 段回答，六种原回答生成模型各 8 段；H=17、N=31。四个核验模型实际完成 **192 次 API 请求，无失败或重试**。已预留的 validation 和 test 尚未调用模型；正式阶段预计 `(1,157 + 875) × 4 = 8,128` 次首次请求，不含失败重试。'
    if START in text:
        left,rest=text.split(START,1)
        _,right=rest.split(END,1)
        text=left+block+right
    else:
        if old not in text: raise ValueError('README insertion anchor missing')
        text=text.replace(old,'pilot 为 48 个开发问题、48 段回答，六种原回答生成模型各 8 段；H=17、N=31。四个核验模型已完成 192 次 pilot 请求，无失败或重试。\n\n'+block)
    text=text.replace('已完成数据接入和小规模真实 API 试跑。使用 [RAGTruth 官方数据]', '已完成数据接入、pilot、正式阈值验证和独立测试。使用 [RAGTruth 官方数据]')
    text=text.replace('正式 validation 和 test 尚未调用模型。','正式 validation/test 结果见第 4.5 节。')
    text=text.replace('这批样本只用于开发检查，尚未发布正式阈值或 final eval 结论。','这批 pilot 样本只用于开发检查，正式阈值与独立测试结果单独列在上方。')
    shared.pilot.atomic_text(path,text)
    for out,title,lines in ((V,'RAGTruth QA 正式验证',validation),(T,'RAGTruth QA 独立测试',testing)):
        body='\n'.join([f'# {title}','',*budget,'',*lines])+'\n'
        body=body.replace('(code/','(../../code/').replace('(docs/','(../../docs/')
        shared.pilot.atomic_text(out/('validation_report.md' if out==V else 'final_report.md'),body)
    print('README and standalone formal reports updated from completed artifacts; no API calls.')


if __name__=='__main__': main()
