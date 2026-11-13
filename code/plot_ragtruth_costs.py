#!/usr/bin/env python3
"""Plot RAGTruth test deployment costs from saved aggregates; never call APIs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / 'code/ragtruth_final_results'
NAMES = {
    'jev_only': 'JEV only',
    'llm_only/qwen': 'Qwen only',
    'cascade/qwen': 'JEV + Qwen',
    'llm_only/deepseek': 'DeepSeek only',
    'cascade/deepseek': 'JEV + DeepSeek',
    'llm_only/kimi': 'Kimi only',
    'cascade/kimi': 'JEV + Kimi',
}


def load_report(results, pricing='historical'):
    if pricing not in ('historical', 'billed'):
        raise ValueError('Unknown pricing basis')
    with (results / 'summary.csv').open(newline='', encoding='utf-8') as handle:
        raw = list(csv.DictReader(handle))
    metadata = json.loads((results / 'report_metadata.json').read_text())
    if metadata['stage'] != 'test' or not metadata['complete']:
        raise ValueError('A completed independent test report is required')
    rows = {row['scenario']: row for row in raw}
    if len(raw) != len(NAMES) or set(rows) != set(NAMES):
        raise ValueError('Expected JEV, three LLM baselines and three frozen cascades')
    for row in rows.values():
        if int(row['samples']) != metadata['sample_count']:
            raise ValueError('Scenarios must cover the same test samples')
        for prefix in ('jev', 'llm'):
            count = int(row[prefix + '_calls'])
            for token in ('input_tokens', 'output_tokens'):
                if int(row[prefix + '_calls_with_' + token]) != count:
                    raise ValueError('Token usage is incomplete; do not treat missing cost as zero')
        uses_jev = int(row['jev_calls']) > 0
        if uses_jev and any(row['estimated_total_cost_usd_' + b] for b in ('low', 'high')):
            raise ValueError('JEV pricing changed; review the unknown-cost figure notes')
        rates = metadata['rates_usd_per_million_tokens'].get(row['model'])
        for i, bound in enumerate(('low', 'high')):
            cost = float(row['llm_estimated_cost_usd_' + bound])
            expected = ((int(row['llm_input_tokens_observed']) * rates[i][0]
                         + int(row['llm_output_tokens_observed']) * rates[i][1]) / 1_000_000
                        if int(row['llm_calls']) else 0.)
            if not math.isclose(cost, expected, abs_tol=1e-12):
                raise ValueError('Saved cost disagrees with recorded tokens and historical rates')
    billing_path = results / 'jev_billing.json'
    billing = json.loads(billing_path.read_text())
    if billing['currency'] != 'USD' or billing['reported_requests'] <= 0 or billing['reported_total_cost_usd'] < 0:
        raise ValueError('Invalid reported JEV billing')
    if sum(billing['matched_terminal_requests'].values()) != billing['reported_requests']:
        raise ValueError('Reported JEV request count does not match the stage totals')
    if billing['matched_terminal_requests']['test'] != metadata['sample_count'] or billing['model'] != rows['jev_only']['model']:
        raise ValueError('Reported JEV billing is for a different model or test size')
    source_paths = [results / 'summary.csv', results / 'report_metadata.json', billing_path]
    llm_bill = None
    fx = 1.
    if pricing == 'billed':
        llm_path = results / 'llm_billing.json'
        llm_bill = json.loads(llm_path.read_text())
        if not llm_bill['scope_confirmed_by_user'] or llm_bill['currency'] != 'CNY':
            raise ValueError('Confirmed RAGTruth CNY billing is required')
        if llm_bill['requests_per_model'] != billing['reported_requests'] or llm_bill['matched_terminal_requests'] != billing['matched_terminal_requests']:
            raise ValueError('JEV and LLM billing scopes differ')
        displayed_sum = sum(llm_bill['model_costs_cny'].values())
        rounding_bound = .0005 * (len(llm_bill['model_costs_cny']) + 1)
        if abs(displayed_sum - llm_bill['reported_total_cost_cny']) > rounding_bound + 1e-12:
            raise ValueError('Per-model billing differs from the total beyond three-decimal display rounding')
        fx = llm_bill['fx']['cny_per_usd']
        if fx <= 0:
            raise ValueError('FX rate must be positive')
        source_paths.append(llm_path)
    for stage, folder in (('pilot', 'ragtruth_pilot_results'), ('validation', 'ragtruth_validation_results'), ('test', results.name)):
        progress_path = results.parent / folder / 'progress.json'
        progress = json.loads(progress_path.read_text())
        if progress['terminal_by_model'][billing['model']] != billing['matched_terminal_requests'][stage]:
            raise ValueError('JEV billing stage counts disagree with saved progress')
        if llm_bill and any(progress['terminal_by_model'][model] != llm_bill['matched_terminal_requests'][stage]
                            for model in llm_bill['model_costs_cny']):
            raise ValueError('LLM billing stage counts disagree with saved progress')
        source_paths.append(progress_path)
    per_call = billing['reported_total_cost_usd'] / billing['reported_requests']
    for row in rows.values():
        j = int(row['jev_calls']) * per_call
        row['jev_allocated_cost_usd'] = j
        row['cost_currency'] = 'USD' if pricing == 'historical' else 'CNY'
        row['pricing_basis'] = pricing
        row['jev_cost'] = j * fx
        for bound in ('low', 'high'):
            row['llm_cost_' + bound] = (float(row['llm_estimated_cost_usd_' + bound]) if pricing == 'historical' else
                llm_bill['model_costs_cny'][row['model']] * int(row['llm_calls']) / llm_bill['requests_per_model']
                if int(row['llm_calls']) else 0.)
            row['total_cost_' + bound] = row['jev_cost'] + row['llm_cost_' + bound]
    for row in rows.values():
        if row['scenario'].startswith('cascade/'):
            baseline = rows[row['scenario'].replace('cascade/', 'llm_only/')]
            reductions = [(1 - row['total_cost_' + bound] / baseline['total_cost_' + bound]) * 100 for bound in ('low', 'high')]
            row['total_cost_reduction_low_percent'] = min(reductions)
            row['total_cost_reduction_high_percent'] = max(reductions)
    metadata['jev_billing'] = billing
    if llm_bill:
        metadata['llm_billing'] = llm_bill
    metadata['pricing_basis'] = pricing
    metadata['cost_currency'] = 'USD' if pricing == 'historical' else 'CNY'
    metadata['jev_effective_average_usd_per_request'] = per_call
    metadata['cost_source_sha256'] = {str(p.relative_to(results.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in source_paths}
    return [rows[name] for name in NAMES], metadata


def cost_range(row, prefix='total_cost_', symbol='$'):
    low, high = (float(row[prefix + bound]) for bound in ('low', 'high'))
    return f'{symbol}{low:.5f}' if low == high else f'{symbol}{low:.5f}–{symbol}{high:.5f}'


def plot_costs(rows, metadata):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 14, 'axes.labelsize': 11,
                         'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.subplots_adjust(left=.095, right=.98, top=.78, bottom=.28, wspace=.35)
    y = np.arange(len(rows))
    names = [NAMES[row['scenario']] for row in rows]
    n = metadata['sample_count']
    for ax in axes:
        ax.set_yticks(y, names if ax is axes[0] else [''] * len(rows))
        ax.set_ylim(len(rows) - .5, -.5)
        ax.tick_params(axis='y', length=0)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.grid(axis='x', color='#dfe5e9')
        ax.set_axisbelow(True)
        for low, high in ((.5, 2.5), (4.5, 6.5)):
            ax.axhspan(low, high, color='#f3f6f8', zorder=0)

    historical = metadata['pricing_basis'] == 'historical'
    symbol = '$' if historical else '¥'
    currency = metadata['cost_currency']
    jcost = np.array([row['jev_cost'] for row in rows])
    low = [row['total_cost_low'] for row in rows]
    high = [row['total_cost_high'] for row in rows]
    axes[0].barh(y, jcost, height=.52, color='#84929b', label='JEV allocation')
    axes[0].barh(y, [row['llm_cost_low'] for row in rows], left=jcost,
                 height=.52, color='#347b91', label='LLM estimate' if historical else 'LLM allocation')
    for i, row in enumerate(rows):
        if high[i] > low[i]:
            axes[0].plot([low[i], high[i]], [i, i], color='#1a4654', lw=3, marker='|', ms=10)
        axes[0].text(high[i] + max(high)*.018, i, cost_range(row, symbol=symbol), va='center', fontsize=10, parse_math=False)
    axes[0].set_xlim(0, max(high) * 1.38)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{symbol}{value:.2f}'))
    axes[0].set_title('A  Estimated total cost', loc='left', weight='bold', pad=48)
    axes[0].set_xlabel(f'JEV + LLM cost ({currency} per {n} responses)\n' +
                      ('Historical LLM prices; not billed savings' if historical else 'Each model bill allocated by request count'))
    axes[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=10)

    jc = np.array([int(row['jev_calls']) for row in rows])
    lc = np.array([int(row['llm_calls']) for row in rows])
    axes[1].barh(y, jc, height=.52, color='#84929b', label='JEV calls')
    axes[1].barh(y, lc, left=jc, height=.52, color='#347b91', label='LLM calls')
    for i, (j, l) in enumerate(zip(jc, lc)):
        if j:
            axes[1].text(j / 2, i, str(j), ha='center', va='center', color='white')
        if l:
            axes[1].text(j + l / 2, i, str(l), ha='center', va='center', color='white')
        axes[1].text(j + l + 20, i, str(j + l), va='center', fontsize=10)
    axes[1].set_xlim(0, max(jc + lc) * 1.18)
    axes[1].set_title('B  Calls under the fixed policy', loc='left', weight='bold', pad=48)
    axes[1].set_xlabel(f'Terminal calls per {n} responses\nStacked segments; number at right = total')
    axes[1].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=10)

    mean = np.array([float(row['mean_latency_ms']) / 1000 for row in rows])
    p95 = np.array([float(row['p95_latency_ms']) / 1000 for row in rows])
    axes[2].barh(y, mean, height=.52, color='#347b91', label='Mean')
    axes[2].scatter(p95, y, color='#c75b39', marker='D', s=28, zorder=3, label='P95')
    for i, (m, p) in enumerate(zip(mean, p95)):
        axes[2].text(m / 2, i, f'{m:.2f}', ha='center', va='center', color='white', fontsize=10)
        axes[2].text(p + .10, i, f'{p:.2f}', va='center', fontsize=10)
    axes[2].set_xlim(0, max(p95) * 1.19)
    axes[2].set_title('C  Response latency', loc='left', weight='bold', pad=48)
    axes[2].set_xlabel('Seconds per response\nCascade = JEV + routed LLM request time')
    axes[2].legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, fontsize=10)
    fig.suptitle('RAGTruth QA | ' + ('Historical-price cost scenario' if historical else 'Bill-allocated cost comparison'),
                 y=.96, fontsize=22, weight='bold')
    fig.text(.5, .90, f'Independent test: {n} responses | Frozen validation thresholds | Same scenario order in all panels',
             ha='center', fontsize=13, color='#425466')
    billing = metadata['jev_billing']
    if historical:
        price_notes = [
            f"JEV: reported ${billing['reported_total_cost_usd']:.3f} / {billing['reported_requests']:,} requests, allocated by count. LLM: recorded tokens times historical USD price assumptions.",
            'Hypothetical pricing scenario, not an actual-spend conclusion. DeepSeek ranges are price assumptions, not confidence intervals.',
        ]
    else:
        llm_bill = metadata['llm_billing']
        price_notes = [
            f"Bills for {billing['reported_requests']:,} requests/model: JEV ${billing['reported_total_cost_usd']:.3f}; Qwen ¥0.507; DeepSeek ¥0.073; Kimi ¥0.516. User-confirmed RAGTruth scope.",
            f"FX: {llm_bill['fx']['date']} official central parity, 1 USD = {llm_bill['fx']['cny_per_usd']} CNY (SAFE/CFETS); not a verified card settlement rate.",
        ]
    notes = price_notes + [
        'Calls and cascade latency are reconstructed from cached full-model predictions; they are not measured savings from this experiment.',
        'Latency excludes retries, queueing and orchestration. P95 is a latency percentile, not an uncertainty interval. No new API calls.',
        ('Historical LLM estimates exclude failed retries; JEV retry billing is not separable. Qwen cascade equals JEV only.' if historical else
         'Costs assume the same mean charge per request within each model; subset/retry charges are not separately known. Qwen cascade equals JEV only.'),
    ]
    for i, note in enumerate(notes):
        fig.text(.04, .16 - i * .029, note, fontsize=11, color='#425466', parse_math=False)
    return fig


def write_plot(results=DEFAULT_RESULTS, out=None, pricing='historical'):
    rows, metadata = load_report(results, pricing)
    fig = plot_costs(rows, metadata)
    stem = 'cost_comparison' if pricing == 'historical' else 'billed_cost_comparison'
    path = out or results / (stem + '.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    fields = ['scenario', 'samples', 'jev_calls', 'llm_calls', 'mean_latency_ms', 'p95_latency_ms',
              'cost_currency', 'pricing_basis', 'jev_allocated_cost_usd', 'jev_cost', 'llm_cost_low', 'llm_cost_high',
              'total_cost_low', 'total_cost_high', 'total_cost_reduction_low_percent', 'total_cost_reduction_high_percent']
    with (results / (stem + '.csv')).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    provenance = {k: metadata[k] for k in ('stage', 'sample_count', 'data_sha256', 'thresholds_sha256',
                  'pricing_basis', 'cost_currency', 'jev_billing', 'jev_effective_average_usd_per_request', 'cost_source_sha256')}
    if pricing == 'historical':
        provenance['rates_usd_per_million_tokens'] = metadata['rates_usd_per_million_tokens']
        provenance['method'] = 'JEV reported aggregate mean times calls, plus recorded LLM tokens at historical USD rates. A pricing scenario, not actual billed savings.'
    else:
        provenance['llm_billing'] = metadata['llm_billing']
        provenance['method'] = 'Each reported model bill divided by 2080 times scenario calls; JEV converted at billing-date central parity.'
    provenance['cost_reduction'] = 'Positive means lower scenario cost versus the corresponding pure LLM; negative means higher. Low/low and high/high price comparisons; not a confidence interval.'
    (results / (stem + '_metadata.json')).write_text(json.dumps(provenance, indent=2) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--pricing', choices=('historical', 'billed'), default='historical',
                        help='Historical USD pricing scenario (default), or separate CNY billing reconciliation')
    args = parser.parse_args()
    print(f'Saved {write_plot(args.results, args.out, args.pricing)}; cached records only, no API calls.')


if __name__ == '__main__':
    main()
