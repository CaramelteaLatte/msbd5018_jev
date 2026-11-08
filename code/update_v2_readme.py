#!/usr/bin/env python3
"""Update only the marked v2 results block, using actual local artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    splits = root / 'data/vitaminc_splits_v2'
    validation = root / 'code/threshold_results_v2'
    test = root / 'code/final_eval_results_v2'
    report = read(splits / 'split_report.json')
    if not report:
        raise ValueError('Create the v2 split first')
    for name, expected in report['output_sha256'].items():
        if hashlib.sha256((splits / (name + '.jsonl')).read_bytes()).hexdigest() != expected:
            raise ValueError('Dataset differs from the split report')
    lines = ['数据切分已完成并通过复现与隔离检查。', '',
             '| 子集 | 条数 | S / C / I | 页面数 | 最大页面条数 |',
             '|---|---:|---|---:|---:|']
    for name in ('threshold_validation', 'test'):
        row = report['subsets'][name]
        labels = ' / '.join(str(row['labels'][g]) for g in 'SCI')
        lines.append(f"| {name} | {row['rows']} | {labels} | {row['pages']} | {row['largest_page_rows']}（{row['largest_page_rows'] / row['rows']:.1%}） |")
    lines += ['', '[v2 划分报告](data/vitaminc_splits_v2/split_report.json)。', '']
    if (root / 'data/vitaminc_v2.zip').exists():
        lines += ['独立数据包：[vitaminc_v2.zip](data/vitaminc_v2.zip)，包含原始数据及 v2 四份切分；[SHA256 清单](data/vitaminc_v2_archive_manifest.json)。旧包继续保留。', '',
                  '```bash', 'unzip data/vitaminc_v2.zip -d data', '```', '']
    for title, directory in [('validation', validation), ('test', test)]:
        progress = read(directory / 'progress.json')
        if progress:
            counts = progress['terminal_by_model']
            total = sum(counts.values())
            expected = progress['expected_per_model'] * len(counts)
            lines.append(f"- {title}：已保存 {total}/{expected} 个终态响应；{'收集完成' if progress['complete'] else '尚未收集完成'}，累计非终态失败尝试 {progress['nonterminal_attempts']} 次。")
        else:
            lines.append(f'- {title}：待运行。')
    threshold = read(validation / 'selected_threshold.json')
    if threshold and threshold.get('complete'):
        if threshold['validation_data_sha256'] != report['output_sha256']['threshold_validation']:
            raise ValueError('Threshold results belong to another validation dataset')
        lines += ['', '**validation 实际结果**（完整集最优值与折外分数分开）：', '',
                  f"JEV only：{threshold['baselines']['jev-1.14.0']['accuracy_all']:.2%}。", '',
                  '| LLM | LLM only | 最终阈值 | 转交条数 | 完整集级联 accuracy | 折外级联 accuracy |',
                  '|---|---:|---:|---:|---:|---:|']
        for model, selected in threshold['selected'].items():
            cutoff = selected['confidence_threshold']
            lines.append(f"| {model} | {threshold['baselines'][model]['accuracy_all']:.2%} | {'不转交' if cutoff is None else cutoff} | {selected['routed']} | {selected['cascade_accuracy']:.2%} | {threshold['out_of_fold'][model]['accuracy_all']:.2%} |")
        lines += ['', '![v2 阈值与净收益](code/threshold_results_v2/threshold_curve.png)', '',
                  '![v2 按连通组的五折验证](code/threshold_results_v2/cross_validation.png)', '',
                  '[固定阈值产物](code/threshold_results_v2/selected_threshold.json)。']
    else:
        lines += ['', '尚无完整 validation 预测，因此尚未生成真实阈值、验证指标及图表。']
    metadata = read(test / 'report_metadata.json')
    if metadata and metadata.get('complete'):
        if metadata['data_sha256'] != report['output_sha256']['test']:
            raise ValueError('Final results belong to another test dataset')
        if metadata['thresholds_sha256'] != hashlib.sha256((validation / 'selected_threshold.json').read_bytes()).hexdigest():
            raise ValueError('Final results use a different threshold artifact')
        with (test / 'summary.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        lines += ['', '**独立 test 实际结果**：', '',
                  '| 方法 | Accuracy | Macro-F1 | 转交条数（比例） | 纠正 / 改错 | 对 JEV 的差值及 95% 区间（百分点） |',
                  '|---|---:|---:|---:|---|---|']
        for row in rows:
            label = row['method'] + (' / ' + row['llm_model'] if row['llm_model'] else '')
            diff, lo, hi = [100 * float(row[k]) for k in ('accuracy_diff_vs_jev', 'accuracy_diff_ci95_low', 'accuracy_diff_ci95_high')]
            routed = f"{row['routed_samples']}（{float(row['routing_rate']):.2%}）" if row['routing_rate'] else '—'
            lines.append(f"| {label} | {float(row['accuracy_all']):.2%} | {float(row['macro_f1_all']):.2%} | {routed} | {row['corrected_vs_jev'] or '—'} / {row['harmed_vs_jev'] or '—'} | {diff:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
        refusals = [f"{row['llm_model']}：{row['llm_provider_refusals']} 条" for row in rows
                    if row['method'] == 'llm_only' and int(row.get('llm_provider_refusals') or 0)]
        if refusals:
            fallbacks = sum(int(row['llm_fallbacks']) for row in rows if row['method'] == 'cascade')
            lines += ['', '服务商内容审查拒绝：' + '；'.join(refusals) + f'。这些样本保留在分母中，单模型计错，若被转交则回退 JEV；本轮三个级联合计实际回退 {fallbacks} 条。未报告 token 用量的请求不假定免费，相应费用估算留空。']
        lines += ['', '![v2 最终评估](code/final_eval_results_v2/final_eval_results.png)', '',
                  '完整成本、延迟及调用量见 [summary.csv](code/final_eval_results_v2/summary.csv)，口径见 [报告元数据](code/final_eval_results_v2/report_metadata.json)。', '']
        for row in rows:
            if row['method'] != 'cascade':
                continue
            lo, hi = float(row['accuracy_diff_ci95_low']), float(row['accuracy_diff_ci95_high'])
            conclusion = '当前样本的区间支持正收益' if lo > 0 else '当前样本的区间支持负收益' if hi < 0 else '区间包含零，尚不能判定稳定增益或损害'
            lines.append(f"- JEV + {row['llm_model']}：{conclusion}。")
        baseline = next(r for r in rows if r['method'] == 'jev_only')
        if all(float(r['accuracy_all']) <= float(baseline['accuracy_all']) for r in rows if r['method'] == 'cascade'):
            lines += ['', '本轮三个固定阈值级联的 test accuracy 均未超过 JEV only。分散页面后仍未观察到稳定的准确率提升，低 confidence 本身不足以确定 LLM 能否带来净纠正。']
        lines += ['', '区间包含零不等于已经证明统计等价或非劣效；若要声称“接近的表现”，应在下一轮预先定义可接受的下降幅度。少量 LLM 调用是相对 LLM only 的场景比较；相对 JEV only，级联增加了调用。', '',
                  '上述判断仅适用于 v2 的页面采样、原标签和固定请求配置；不将 v1/v2 的分数差归因为模型能力变化。']
    else:
        lines += ['', '独立 test 尚无完整结果；不预填分数、图表或增益结论。']
    text = (root / 'readme.md').read_text(encoding='utf-8')
    start, end = '<!-- V2_RESULTS_START -->', '<!-- V2_RESULTS_END -->'
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('README must contain exactly one v2 results block')
    before, rest = text.split(start)
    _, after = rest.split(end)
    (root / 'readme.md').write_text(before + start + '\n' + '\n'.join(lines) + '\n' + end + after, encoding='utf-8')
    print('Updated README v2 block from actual artifacts.')


if __name__ == '__main__':
    main()
