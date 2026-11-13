#!/usr/bin/env python3
"""Compare the first VitaminC pilot using saved CSV/manifest only; no API calls."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {'jev-1.14.0': 'JEV', 'qwen3.8-max': 'Qwen', 'deepseek-v4.1-flash-1017': 'DeepSeek',
           'kimi-k3.1-pro': 'Kimi', 'glm-5.3-prime': 'GLM'}
GLM = 'glm-5.3-prime'


def configure_font(english=False):
    if not english:
        for name in ('PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', 'Arial Unicode MS', 'Heiti SC', 'SimHei'):
            try:
                font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            except ValueError:
                continue
            plt.rcParams['font.family'] = name
            return True
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return False


def plot(rows, manifest, chinese):
    def label(zh, en):
        return zh if chinese else en

    if not rows or set(r['model'] for r in rows) != set(manifest['models']):
        raise ValueError('CSV model list differs from pilot manifest')
    n = int(rows[0]['expected'])
    for r in rows:
        if int(r['expected']) != n or int(r['successful']) + int(r['failed']) != n:
            raise ValueError('Expected complete, equally sized pilot results')
        if abs(float(r['accuracy_all_50']) - int(r['correct']) / n) > .0001:
            raise ValueError('Accuracy must use all samples, including invalid outputs')
    names = [ALIASES.get(r['model'], r['model']) for r in rows]
    colors = ['#c75b39' if r['model'] == GLM else '#347b91' for r in rows]
    y = list(range(len(rows)))
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 15, 'axes.labelsize': 11,
                         'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.subplots_adjust(left=.10, right=.97, top=.84, bottom=.21, hspace=.40, wspace=.34)

    def bars(ax, values, title, xlabel, formatter, limit):
        ax.barh(y, values, color=colors, height=.58, zorder=3)
        ax.set_title(title, loc='left', pad=12, weight='bold')
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, limit)
        for i, value in enumerate(values):
            ax.text(value + limit*.014, i, formatter(value, i), va='center', fontsize=11)

    bars(axes[0, 0], [float(r['accuracy_all_50'])*100 for r in rows],
         label('A  准确率：越高越好', 'A  Accuracy | higher is better'),
         label('全部样本中判断正确的比例（%）', 'Correct predictions over all samples (%)'),
         lambda v, i: f"{v:.0f}%  ({rows[i]['correct']}/{n})", 100)
    bars(axes[0, 1], [float(r['avg_latency_ms'])/1000 for r in rows],
         label('B  平均每次耗时：越低越快', 'B  Mean request latency | lower is faster'),
         label('平均请求耗时（秒，仅有效响应）', 'Mean request latency (seconds; valid responses)'),
         lambda v, i: label(f'{v:.2f} 秒', f'{v:.2f} s'),
         max(float(r['avg_latency_ms'])/1000 for r in rows)*1.25)

    ax = axes[1, 0]
    max_cost = max(float(r['estimated_cost_usd_high']) for r in rows if r['estimated_cost_usd_high'])
    limit = max_cost * 1.34
    ax.set_title(label(f'C  {n} 次请求的 LLM 费用估算', f'C  Estimated LLM cost for {n} requests'),
                 loc='left', pad=12, weight='bold')
    ax.set_xlabel(label('历史价格口径（美元；不是当前报价）', 'USD | historical price assumptions'))
    ax.set_xlim(0, limit)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.2f}'))
    for i, r in enumerate(rows):
        if not r['estimated_cost_usd_low'] or not r['estimated_cost_usd_high']:
            ax.text(limit*.025, i, label('单价未知，不计为 $0', 'Unknown price; not $0'), va='center', color='#666666')
            continue
        low, high = float(r['estimated_cost_usd_low']), float(r['estimated_cost_usd_high'])
        ax.barh(i, low, height=.58, color=colors[i], zorder=3)
        if high > low:
            ax.plot([low, high], [i, i], color=colors[i], linewidth=4, marker='|', markersize=10, zorder=4)
        value = f'${low:.5f}' if low == high else f'${low:.5f} - ${high:.5f}'
        ax.text(high + limit*.014, i, value, va='center', fontsize=10, parse_math=False)

    bars(axes[1, 1], [float(r['avg_output_tokens']) for r in rows],
         label('D  平均输出量：帮助理解耗时与费用', 'D  Mean output tokens per response'),
         label('每次输出 token 数（仅有效响应，不是准确率）', 'Output tokens per valid response (not a quality score)'),
         lambda v, i: f'{v:.2f}', max(float(r['avg_output_tokens']) for r in rows)*1.25)

    for ax in axes.flat:
        ax.set_yticks(y, names)
        ax.set_ylim(len(rows)-.5, -.5)
        ax.grid(axis='x', color='#dfe5e9', linewidth=.8, zorder=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#c6cdd3')
        ax.tick_params(axis='y', length=0, pad=9)
        ax.set_axisbelow(True)
    fig.suptitle(label('第一次 VitaminC pilot：五个模型的表现、耗时与费用',
                       'First VitaminC pilot | quality, request latency and cost'),
                 fontsize=21, weight='bold', y=.97)
    fig.text(.5, .91, label(f'同一批 {n} 条样本；每个模型各判断一次。橙红色突出显示 GLM。',
                           f'Same {n} samples per model. GLM highlighted in orange.'),
             ha='center', color='#425466', fontsize=12)
    glm = next(r for r in rows if r['model'] == GLM)
    config = manifest['chat_options'][GLM]
    others = [manifest['chat_options'][r['model']] for r in rows if r['model'] in manifest['chat_options'] and r['model'] != GLM]
    if not config['enable_thinking'] or any(c['enable_thinking'] or c['max_completion_tokens'] != 32 for c in others):
        raise ValueError('Pilot request settings changed; review figure notes')
    notes = [
        label(f"配置：GLM 开启 thinking，输出上限 {config['max_completion_tokens']} tokens；其余三个 LLM 关闭 thinking，上限 32。JEV 使用独立接口。",
              f"Settings: GLM thinking ON, max {config['max_completion_tokens']} tokens; other three LLMs thinking OFF, max 32. JEV uses its own API."),
        label(f"统计：准确率以全部 {n} 条为分母；耗时与输出量只统计有效响应（GLM {glm['successful']} 条，其他模型 {n} 条）。",
              f"Accuracy includes all {n} samples; latency/output means use valid responses only (GLM {glm['successful']}; others {n})."),
        label('费用：包含有用量记录的无效响应；DeepSeek 短线表示历史价格假设范围，不是置信区间。',
              'Costs include invalid responses with recorded usage; the DeepSeek range is a price assumption, not a confidence interval.'),
        label('来源：code/pilot_results/summary.csv 与 manifest.json；开发试跑结果，不代表独立测试或相同配置下的通用模型排名。',
              'Source: code/pilot_results/summary.csv and manifest.json. Development pilot; not final evaluation or a matched-settings ranking.'),
    ]
    for i, note in enumerate(notes):
        fig.text(.035, .135-i*.030, note, fontsize=10, color='#425466', va='top')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT/'code/pilot_results/summary.csv')
    parser.add_argument('--manifest', type=Path, default=ROOT/'code/pilot_results/manifest.json')
    parser.add_argument('--out', type=Path, default=ROOT/'code/pilot_results/pilot_comparison.png')
    language = parser.add_mutually_exclusive_group()
    language.add_argument('--english', dest='english', action='store_true', help='Use English labels (default)')
    language.add_argument('--chinese', dest='english', action='store_false', help='Use Chinese labels when a Chinese font is available')
    parser.set_defaults(english=True)
    args = parser.parse_args()
    with args.summary.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    fig = plot(rows, json.loads(args.manifest.read_text()), configure_font(args.english))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(f'Saved {args.out}; used cached table only, no API calls.')


if __name__ == '__main__':
    main()
