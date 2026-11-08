#!/usr/bin/env python3
"""v2 independent test evaluation using an explicit, frozen validation artifact."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sqlite3
import sys

import experiment_v2 as ev


def load_thresholds(path):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if report.get('version') != 'v2' or report.get('complete') is not True or report.get('split') != 'threshold_validation':
        raise ValueError('Expected a complete v2 validation threshold artifact')
    if report.get('request_configs') != ev.request_configs():
        raise ValueError('Threshold model/prompt/request configuration differs from this run')
    if (report.get('jev_url'), report.get('bailian_url')) != (ev.pilot.JEV_URL, ev.pilot.BAILIAN_URL):
        raise ValueError('Threshold provider endpoints differ from this run')
    if set(report.get('selected', {})) != set(ev.LLMS.values()):
        raise ValueError('Threshold artifact must contain all three LLMs')
    cutoffs = {}
    for model in ev.LLMS.values():
        cutoff = report['selected'][model]['confidence_threshold']
        if cutoff is not None and (type(cutoff) not in (int, float) or not math.isfinite(cutoff) or not 0 <= cutoff <= 1):
            raise ValueError(f'Invalid cutoff for {model}')
        cutoffs[model] = cutoff
    return report, cutoffs


def summarize(method, model, samples, results, cutoff, jev_rates=None, jev_per_1000=None):
    jev = results[ev.JEV]
    llm = results[model] if model else {}
    predictions, latencies, llm_calls = [], [], []
    use_jev = method != 'llm_only'
    for sample in samples:
        sid = sample['sample_id']
        routed = method == 'cascade' and ev.is_routed(jev[sid], cutoff)
        elapsed = jev[sid]['latency_ms'] if use_jev else 0
        if method == 'llm_only' or routed:
            llm_calls.append(llm[sid])
            elapsed += llm[sid]['latency_ms']
        if method == 'llm_only':
            pred = ev.prediction(llm[sid])
        elif method == 'cascade':
            pred = ev.cascade_prediction(jev[sid], llm[sid], cutoff)
        else:
            pred = ev.prediction(jev[sid])
        predictions.append(pred)
        latencies.append(elapsed)
    metrics = ev.score(samples, predictions)
    row = dict(method=method, jev_model=ev.JEV if use_jev else None, llm_model=model,
               confidence_cutoff=cutoff if method == 'cascade' else None,
               **{k: v for k, v in metrics.items() if k not in ('per_label', 'confusion')})
    routing = ev.routing_metrics(samples, jev, llm, cutoff) if method == 'cascade' else {}
    row.update(routed_samples=routing.get('routed'), routing_rate=routing.get('routing_rate'),
               llm_fallbacks=routing.get('llm_fallbacks'), corrected_vs_jev=routing.get('corrected'),
               harmed_vs_jev=routing.get('harmed'), api_calls=(len(samples) if use_jev else 0) + len(llm_calls),
               mean_e2e_latency_ms=sum(latencies) / len(samples),
               p95_e2e_latency_ms=sorted(latencies)[math.ceil(.95 * len(samples)) - 1],
               total_sample_latency_seconds=sum(latencies) / 1000)
    jev_usage = ev.usage_and_cost([jev[s['sample_id']] for s in samples] if use_jev else [], jev_rates, jev_per_1000)
    llm_usage = ev.usage_and_cost(llm_calls, ev.pilot.RATES.get(model))
    for provider, usage in [('jev', jev_usage), ('llm', llm_usage)]:
        row.update({provider + '_' + k: v for k, v in usage.items()})
    for bound in ('low', 'high'):
        key = 'estimated_cost_usd_' + bound
        a, b = jev_usage[key], llm_usage[key]
        row[key] = a + b if a is not None and b is not None else None
    baseline = [ev.prediction(jev[s['sample_id']]) for s in samples]
    interval = ev.bootstrap(samples, baseline, predictions)
    row.update(accuracy_diff_vs_jev=interval['difference_accuracy'], accuracy_diff_ci95_low=interval['ci95_low'],
               accuracy_diff_ci95_high=interval['ci95_high'])
    return row, metrics, predictions, interval


def plot_reports(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['JEV' if r['method'] == 'jev_only' else ('JEV + ' if r['method'] == 'cascade' else '') +
              next(k for k, v in ev.LLMS.items() if v == r['llm_model']) for r in rows]
    colors = ['#526175' if r['method'] == 'jev_only' else '#167c80' if r['method'] == 'cascade' else '#4774b4' for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    y = list(range(len(rows)))
    axes[0, 0].barh(y, [r['accuracy_all'] * 100 for r in rows], color=colors)
    for i, row in enumerate(rows):
        axes[0, 0].text(row['accuracy_all'] * 100 + .4, i, f"{row['accuracy_all']:.2%}", va='center', fontsize=9)
    axes[0, 0].set_xlim(0, 105)
    axes[0, 0].set_xlabel('Test accuracy (%)')
    axes[0, 0].set_title('Standalone and fixed-threshold cascades')
    for i, row in enumerate(rows):
        d = row['accuracy_diff_vs_jev'] * 100
        lo, hi = row['accuracy_diff_ci95_low'] * 100, row['accuracy_diff_ci95_high'] * 100
        axes[0, 1].plot([lo, hi], [i, i], color=colors[i], lw=2)
        axes[0, 1].scatter([d], [i], color=colors[i])
    axes[0, 1].axvline(0, color='#777777', ls='--')
    axes[0, 1].set_xlabel('Accuracy difference vs JEV (percentage points)')
    axes[0, 1].set_title('95% paired page-bootstrap intervals')
    axes[1, 0].scatter([r['mean_e2e_latency_ms'] / 1000 for r in rows], y, label='Mean', color=colors)
    axes[1, 0].scatter([r['p95_e2e_latency_ms'] / 1000 for r in rows], y, label='P95', marker='x', color=colors)
    axes[1, 0].set_xlabel('Reconstructed terminal-request latency (seconds)')
    axes[1, 0].legend()
    for i, row in enumerate(rows):
        low, high = row['llm_estimated_cost_usd_low'], row['llm_estimated_cost_usd_high']
        if low is not None and high is not None:
            axes[1, 1].plot([low, high], [i, i], color=colors[i], lw=3, marker='o')
        else:
            axes[1, 1].text(0, i, 'Token usage incomplete', va='center', fontsize=8, color='#666666')
    axes[1, 1].set_xlabel('LLM scenario cost estimate (USD); JEV excluded')
    axes[1, 1].set_title('Stored v1 price assumptions; terminal responses only')
    for ax in axes.flat:
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('VitaminC v2 independent test | no threshold retuning on test')
    fig.savefig(out / 'final_eval_results.png', dpi=160)
    plt.close(fig)


def generate_reports(out, samples, results, cutoffs, config, jev_rates=None, jev_per_1000=None):
    rows, detailed, per_label, intervals, predictions = [], {}, [], {}, {}
    scenarios = [('jev_only', None)] + [(method, model) for model in ev.LLMS.values() for method in ('llm_only', 'cascade')]
    for method, model in scenarios:
        key = method + (':' + model if model else '')
        row, metrics, pred, interval = summarize(method, model, samples, results, cutoffs.get(model), jev_rates, jev_per_1000)
        rows.append(row)
        detailed[key] = metrics
        intervals[key] = interval
        predictions[key] = pred
        for label, values in metrics['per_label'].items():
            per_label.append(dict(method=method, model=model, label=label, **values))
    ev.write_csv(out / 'summary.csv', rows)
    ev.write_csv(out / 'per_label.csv', per_label)
    ev.write_json(out / 'metrics.json', detailed)
    ev.write_json(out / 'bootstrap.json', intervals)
    ev.pilot.atomic_text(out / 'predictions.jsonl', ''.join(json.dumps(dict(sample_id=s['sample_id'], gold=s['gold'], page=s['page'],
                                predictions={k: v[i] for k, v in predictions.items()}), ensure_ascii=False) + '\n' for i, s in enumerate(samples)))
    report = dict(**config, complete=True,
                  scoring_note='All test samples count. Invalid LLM output or explicit provider data_inspection_failed refusal is wrong standalone and falls back to JEV when routed. Refusals are retained without rewriting inputs or resubmitting.',
                  cost_note='Stored v1 pilot price assumptions, not current verified prices. Scenario costs count terminal calls only; JEV unknown pricing keeps total cost null.',
                  latency_note='Sum of terminal request durations; excludes retries, backoff, queues and orchestration. Measurements use concurrent collection.',
                  uncertainty_note='10000 paired page-cluster bootstrap repetitions, seed 20261109; conditional on this sample and its original labels.',
                  llm_rates_usd_per_million_tokens={m: ev.pilot.RATES[m] for m in ev.LLMS.values()},
                  jev_rates_usd_per_million_tokens=jev_rates, jev_usd_per_1000_calls=jev_per_1000)
    plot_reports(out, rows)
    ev.write_json(out / 'report_metadata.json', report)
    print(json.dumps([{'method': r['method'], 'llm': r['llm_model'], 'accuracy': r['accuracy_all'], 'cutoff': r['confidence_cutoff']} for r in rows], indent=2), flush=True)
    return rows


def run(args):
    ev.validate_args(args)
    for rate in (args.jev_rates or [args.jev_usd_per_1000_calls]):
        if rate is not None and (not math.isfinite(rate) or rate < 0):
            raise ValueError('Price rates must be finite and nonnegative')
    threshold_report, cutoffs = load_thresholds(args.thresholds)
    samples = ev.load_samples(args.data, 'test')
    config = ev.make_config(args.data, samples, 'test', thresholds_sha256=ev.digest(args.thresholds),
                            validation_data_sha256=threshold_report['validation_data_sha256'], cutoffs=cutoffs,
                            bootstrap_repetitions=10000, bootstrap_seed=ev.SEED)
    if args.plan:
        ev.plan(samples, args)
        print(json.dumps({'cutoffs': cutoffs}, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = ev.collect_stage(args, samples, config, args.out / 'final_eval_checkpoint.sqlite3')
        if results is not None:
            jev_rates = (tuple(args.jev_rates),) * 2 if args.jev_rates else None
            generate_reports(args.out, samples, results, cutoffs, config, jev_rates, args.jev_usd_per_1000_calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    ev.add_collection_args(parser, 'code/final_eval_results_v2')
    parser.add_argument('--thresholds', type=Path, required=True, help='complete v2 selected_threshold.json from validation')
    prices = parser.add_mutually_exclusive_group()
    prices.add_argument('--jev-rates', nargs=2, type=float, metavar=('INPUT', 'OUTPUT'))
    prices.add_argument('--jev-usd-per-1000-calls', type=float)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print('Stopped; resume with the same arguments.', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
