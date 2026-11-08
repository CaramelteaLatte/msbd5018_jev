#!/usr/bin/env python3
"""v2: complete validation baselines, per-model cutoffs and grouped out-of-fold evaluation."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import experiment_v2 as ev


def plot_reports(out, report, sweeps, folds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex='col', layout='constrained')
    for col, (alias, model) in enumerate(ev.LLMS.items()):
        points = sweeps[model]
        xs = [p['confidence_threshold'] for p in points if p['confidence_threshold'] is not None]
        numeric = [p for p in points if p['confidence_threshold'] is not None]
        acc, routing, net = axes[:, col]
        acc.axhline(report['baselines'][ev.JEV]['accuracy_all'] * 100, color='#444444', ls='--', label='JEV only')
        acc.axhline(report['baselines'][model]['accuracy_all'] * 100, color='#a66b32', ls=':', label=f'{alias} only (full validation)')
        if xs:
            acc.step(xs, [p['cascade_accuracy'] * 100 for p in numeric], where='post', color='#167c80', label=f'JEV + {alias}')
            routing.step(xs, [p['routing_rate'] * 100 for p in numeric], where='post', color='#167c80')
            for key, color in [('corrected', '#187c66'), ('harmed', '#b64d3c'), ('net', '#333333')]:
                net.step(xs, [p[key] for p in numeric], where='post', color=color, label=key)
        else:
            acc.text(.5, .5, 'No confidence group fits the budget', transform=acc.transAxes, ha='center')
        cutoff = report['selected'][model]['confidence_threshold']
        acc.set_title(f"{alias}: selected {'no routing' if cutoff is None else format(cutoff, '.4g')}")
        if cutoff is not None:
            for ax in (acc, routing, net):
                ax.axvline(cutoff, color='#b64d3c', ls=':', lw=1)
        acc.set_ylabel('Full-validation accuracy (%)')
        routing.set_ylabel('Routed (%)')
        routing.axhline(report['max_routing_percent'], color='#666666', ls='--')
        net.axhline(0, color='#999999', lw=.8)
        net.set_ylabel('Samples')
        net.set_xlabel('Raw JEV confidence cutoff (<=)')
        acc.legend(fontsize=7)
        net.legend(fontsize=7)
        for ax in (acc, routing, net):
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('v2 validation | apparent full-set selection (not out-of-fold estimates)')
    fig.savefig(out / 'threshold_curve.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for alias, model in ev.LLMS.items():
        rows = [r for r in folds if r['model'] == model]
        x = [r['fold'] for r in rows]
        axes[0].plot(x, [100 * r['net'] / r['held_samples'] for r in rows], marker='o', label=alias)
        # A separate y position explicitly denotes None; never treat None as cutoff 0.
        axes[1].plot(x, [r['confidence_threshold'] if r['confidence_threshold'] is not None else -.05 for r in rows], marker='o', label=alias)
    axes[0].axhline(0, color='#777777', ls='--')
    axes[0].set_ylabel('Held-out accuracy gain vs JEV (pp)')
    axes[1].set_ylabel('Training-fold selected cutoff')
    axes[1].axhline(-.05, color='#aaaaaa', ls=':')
    axes[1].text(1, -.05, 'No routing', va='bottom', fontsize=8)
    for ax in axes:
        ax.set_xticks(range(1, 6))
        ax.set_xlabel('Held-out group fold')
        ax.legend()
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('v2 grouped cross-validation | threshold selected on the other four folds')
    fig.savefig(out / 'cross_validation.png', dpi=160)
    plt.close(fig)


def generate_reports(out, samples, results, config):
    baselines = {m: ev.score(samples, [ev.prediction(results[m][s['sample_id']]) for s in samples]) for m in ev.MODELS}
    sweeps, selected = {}, {}
    for model in ev.LLMS.values():
        sweeps[model], selected[model] = ev.threshold_sweep(samples, results[ev.JEV], results[model], config['max_routing_percent'])
    assignment, folds, oof = ev.cross_validate(samples, results, config['max_routing_percent'], config['fold_seed'])
    report = dict(version='v2', complete=True, split='threshold_validation', validation_size=len(samples),
                  validation_data_sha256=config['data_sha256'], score='raw Jev confidence',
                  rule='Route when confidence <= threshold. Null means never route.',
                  objective='Per-model maximum correct count; ties prefer fewer calls, including no routing.',
                  max_routing_percent=config['max_routing_percent'], selected=selected,
                  baselines=baselines, out_of_fold=oof, fold_seed=config['fold_seed'],
                  request_configs=config['request_configs'], jev_url=config['jev_url'], bailian_url=config['bailian_url'],
                  statistical_note='Full-validation selected scores are apparent estimates. OOF thresholds use other groups only. The 20% constraint applies to selection folds, not held-out/test routing rates.')
    ev.write_csv(out / 'threshold_sweep.csv', [dict(model=m, **p) for m in ev.LLMS.values() for p in sweeps[m]])
    ev.write_csv(out / 'cross_validation.csv', folds)
    ev.write_json(out / 'fold_assignments.json', dict(seed=config['fold_seed'], group_to_fold={g: f + 1 for g, f in assignment.items()}))
    rows = []
    for model, metrics in baselines.items():
        rows.append(dict(model=model, method='standalone_validation', samples=len(samples), accuracy=metrics['accuracy_all'], macro_f1=metrics['macro_f1_all'], threshold=None))
    for model in ev.LLMS.values():
        predictions = [ev.cascade_prediction(results[ev.JEV][s['sample_id']], results[model][s['sample_id']], selected[model]['confidence_threshold']) for s in samples]
        metrics = ev.score(samples, predictions)
        rows.append(dict(model=model, method='selected_on_full_validation', samples=len(samples), accuracy=metrics['accuracy_all'], macro_f1=metrics['macro_f1_all'], threshold=selected[model]['confidence_threshold']))
        rows.append(dict(model=model, method='out_of_fold', samples=len(samples), accuracy=oof[model]['accuracy_all'], macro_f1=oof[model]['macro_f1_all'], threshold='fold-specific'))
    ev.write_csv(out / 'summary.csv', rows)
    plot_reports(out, report, sweeps, folds)
    # Publish the artifact only after all reports succeed; the test run consumes its exact hash.
    ev.write_json(out / 'selected_threshold.json', report)
    print(json.dumps({'selected': selected, 'out_of_fold_accuracy': {m: v['accuracy_all'] for m, v in oof.items()}}, indent=2), flush=True)
    return report


def run(args):
    ev.validate_args(args)
    if not 0 <= args.max_routing_percent <= 100:
        raise ValueError('Routing budget must be 0..100')
    samples = ev.load_samples(args.data, 'threshold_validation')
    config = ev.make_config(args.data, samples, 'threshold_validation', max_routing_percent=args.max_routing_percent, fold_seed=ev.SEED, folds=5)
    if args.plan:
        ev.plan(samples, args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = ev.collect_stage(args, samples, config, args.out / 'threshold_checkpoint.sqlite3')
        if results is not None:
            generate_reports(args.out, samples, results, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    ev.add_collection_args(parser, 'code/threshold_results_v2')
    parser.add_argument('--max-routing-percent', type=int, default=20)
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
