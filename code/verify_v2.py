#!/usr/bin/env python3
"""Independent offline recomputation of v2 files (no imports of scoring/selection code)."""
from collections import Counter
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3

JEV = 'jev-1.14.0'
LLMS = ('qwen3.8-max', 'deepseek-v4.1-flash-1017', 'kimi-k3.1-pro')


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def keys(rows):
    norm = lambda s: ' '.join(s.split())
    return dict(id={r['unique_id'] for r in rows}, case={r['case_id'] for r in rows},
                page={norm(r['page']).replace('_', ' ') for r in rows},
                pair={(norm(r['claim']), norm(r['evidence'])) for r in rows})


def load_checkpoint(path, data):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        config = json.loads(db.execute('SELECT manifest FROM config').fetchone()[0])
        records = [json.loads(b) for b, in db.execute('SELECT body FROM attempts WHERE terminal=1')]
    indexed = {}
    for row in records:
        key = (row['model'], row['sample_id'])
        assert key not in indexed, ('duplicate terminal', key)
        assert row['gold'] == data[row['sample_id']]['gold']
        if row['status'] == 'ok':
            assert row['correct'] == (row['prediction'] == row['gold'])
        indexed[key] = row
    return config, indexed


def predict(record):
    return record['prediction'] if record['status'] == 'ok' else None


def route(data, indexed, model, cutoff):
    predicted, routed, corrected, harmed, fallbacks = [], 0, 0, 0, 0
    for sid, sample in data.items():
        j, llm = indexed[JEV, sid], indexed[model, sid]
        pred = predict(j)
        if cutoff is not None and j['confidence'] <= cutoff:
            routed += 1
            if llm['status'] == 'ok':
                pred = predict(llm)
                corrected += int(predict(j) != sample['gold'] and pred == sample['gold'])
                harmed += int(predict(j) == sample['gold'] and pred != sample['gold'])
            else:
                fallbacks += 1
        predicted.append(pred)
    correct = sum(p == r['gold'] for p, r in zip(predicted, data.values()))
    return dict(predictions=predicted, routed=routed, corrected=corrected, harmed=harmed,
                llm_fallbacks=fallbacks, cascade_correct=correct, cascade_accuracy=correct / len(data),
                net=corrected - harmed, routing_rate=routed / len(data))


def select(data, indexed, model, budget):
    candidates = [None] + sorted({indexed[JEV, sid]['confidence'] for sid in data})
    feasible = []
    for cutoff in candidates:
        metrics = route(data, indexed, model, cutoff)
        if metrics['routed'] <= len(data) * budget // 100:
            feasible.append((cutoff, metrics))
    return max(feasible, key=lambda x: (x[1]['cascade_correct'], -x[1]['routed']))


def check_metrics(actual, expected, names):
    for name in names:
        assert abs(float(actual[name]) - expected[name]) < 1e-12, (name, actual[name], expected[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    folder = root / 'data/vitaminc_splits_v2'
    report = json.loads((folder / 'split_report.json').read_text())
    subsets = {name: read_rows(folder / (name + '.jsonl')) for name in report['output_sha256']}
    identities = {name: keys(rows) for name, rows in subsets.items()}
    for name, rows in subsets.items():
        assert hashlib.sha256((folder / (name + '.jsonl')).read_bytes()).hexdigest() == report['output_sha256'][name]
        assert len(rows) == len(identities[name]['id']) == len(identities[name]['pair'])
        if name in ('test', 'threshold_validation'):
            assert len(rows) == 2000
            assert max(Counter(' '.join(r['page'].split()).replace('_', ' ') for r in rows).values()) <= 20
            assert all(r['revision_type'] == 'real' for r in rows)
            expected = Counter(S=1002, C=575, I=423) if name == 'test' else Counter(S=996, C=575, I=429)
            assert Counter(r['gold'] for r in rows) == expected
    for a, b in itertools.combinations(identities, 2):
        for kind in identities[a]:
            assert not identities[a][kind] & identities[b][kind], (a, b, kind)
    old_test = keys(read_rows(root / 'data/vitaminc_splits/test.jsonl'))
    for kind in old_test:
        assert not old_test[kind] & identities['test'][kind]
    print('PASS: data hashes, exact counts/quotas, page caps, split isolation, old-test exclusion')

    validation = root / 'code/threshold_results_v2'
    artifact = validation / 'selected_threshold.json'
    if artifact.exists():
        data = {r['unique_id']: r for r in subsets['threshold_validation']}
        config, indexed = load_checkpoint(validation / 'threshold_checkpoint.sqlite3', data)
        selected = json.loads(artifact.read_text())
        assert config['data_sha256'] == selected['validation_data_sha256'] == report['output_sha256']['threshold_validation']
        assert len(indexed) == 4 * len(data)
        for model in LLMS:
            cutoff, expected = select(data, indexed, model, config['max_routing_percent'])
            assert cutoff == selected['selected'][model]['confidence_threshold']
            check_metrics(selected['selected'][model], expected,
                          ('cascade_correct', 'cascade_accuracy', 'routed', 'corrected', 'harmed', 'net', 'llm_fallbacks'))
        with (validation / 'threshold_sweep.csv').open() as h:
            for row in csv.DictReader(h):
                cutoff = float(row['confidence_threshold']) if row['confidence_threshold'] else None
                expected = route(data, indexed, row['model'], cutoff)
                check_metrics(row, expected, ('routed', 'cascade_correct', 'cascade_accuracy', 'corrected', 'harmed'))
        assignment = json.loads((validation / 'fold_assignments.json').read_text())['group_to_fold']
        oof_correct = Counter()
        with (validation / 'cross_validation.csv').open() as h:
            for row in csv.DictReader(h):
                fold, model = int(row['fold']), row['model']
                train = {sid: r for sid, r in data.items() if assignment[r['group_id']] != fold}
                held = {sid: r for sid, r in data.items() if assignment[r['group_id']] == fold}
                assert not keys(list(train.values()))['page'] & keys(list(held.values()))['page']
                cutoff, _ = select(train, indexed, model, config['max_routing_percent'])
                recorded = float(row['confidence_threshold']) if row['confidence_threshold'] else None
                assert cutoff == recorded
                expected = route(held, indexed, model, cutoff)
                check_metrics(row, expected, ('cascade_correct', 'cascade_accuracy', 'routed', 'corrected', 'harmed'))
                oof_correct[model] += expected['cascade_correct']
        for model in LLMS:
            assert oof_correct[model] == selected['out_of_fold'][model]['correct']
        print('PASS: all threshold sweep points, independent optimal cutoffs, group-separated fold selection, OOF totals')
    else:
        print('PENDING: validation artifact (no estimated results substituted)')

    final = root / 'code/final_eval_results_v2'
    summary = final / 'summary.csv'
    if summary.exists():
        data = {r['unique_id']: r for r in subsets['test']}
        config, indexed = load_checkpoint(final / 'final_eval_checkpoint.sqlite3', data)
        assert config['data_sha256'] == report['output_sha256']['test']
        assert config['thresholds_sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert len(indexed) == 4 * len(data)
        with summary.open() as h:
            rows = list(csv.DictReader(h))
        assert len(rows) == 7
        exported = read_rows(final / 'predictions.jsonl')
        assert [r['sample_id'] for r in exported] == list(data)
        detailed = json.loads((final / 'metrics.json').read_text())
        intervals = json.loads((final / 'bootstrap.json').read_text())
        metadata = json.loads((final / 'report_metadata.json').read_text())
        assert metadata['complete'] and metadata['thresholds_sha256'] == config['thresholds_sha256']
        # Recompute the paired page resamples independently of experiment_v2.
        import numpy as np
        pages = [' '.join(r['page'].split()).replace('_', ' ') for r in data.values()]
        page_index = {page: i for i, page in enumerate(dict.fromkeys(pages))}
        membership = np.array([page_index[p] for p in pages])
        sizes = np.bincount(membership)
        draws = np.random.default_rng(20261109).integers(0, len(sizes), size=(10000, len(sizes)))
        denominators = sizes[draws].sum(axis=1)
        gold = [r['gold'] for r in data.values()]
        baseline_correct = np.array([predict(indexed[JEV, sid]) == r['gold'] for sid, r in data.items()])
        for row in rows:
            scenario = row['method'] + (':' + row['llm_model'] if row['llm_model'] else '')
            if row['method'] == 'cascade':
                expected = route(data, indexed, row['llm_model'], config['cutoffs'][row['llm_model']])
                predictions = expected['predictions']
                for key, source in [('routed_samples', 'routed'), ('corrected_vs_jev', 'corrected'),
                                    ('harmed_vs_jev', 'harmed'), ('llm_fallbacks', 'llm_fallbacks')]:
                    assert int(row[key]) == expected[source]
            else:
                model = JEV if row['method'] == 'jev_only' else row['llm_model']
                predictions = [predict(indexed[model, sid]) for sid in data]
            pairs = list(zip([r['gold'] for r in data.values()], predictions))
            assert [r['predictions'][scenario] for r in exported] == predictions
            assert [r['gold'] for r in exported] == gold
            confusion = {label: {p: sum(g == label and pred == (None if p == 'INVALID' else p)
                                       for g, pred in pairs) for p in ('S', 'C', 'I', 'INVALID')}
                         for label in 'SCI'}
            assert detailed[scenario]['confusion'] == confusion
            correct = sum(a == b for a, b in pairs)
            f1 = []
            for label in 'SCI':
                denominator = sum(a == label for a, _ in pairs) + predictions.count(label)
                f1.append(2 * sum(a == b == label for a, b in pairs) / denominator if denominator else 0.)
            check_metrics(row, dict(correct=correct, accuracy_all=correct / len(data), macro_f1_all=sum(f1) / 3),
                          ('correct', 'accuracy_all', 'macro_f1_all'))
            difference = np.array([a == b for a, b in pairs], dtype=int) - baseline_correct.astype(int)
            page_difference = np.bincount(membership, weights=difference)
            low, high = np.quantile(page_difference[draws].sum(axis=1) / denominators, [.025, .975])
            check_metrics(row, dict(accuracy_diff_vs_jev=float(difference.mean()),
                                   accuracy_diff_ci95_low=float(low), accuracy_diff_ci95_high=float(high)),
                          ('accuracy_diff_vs_jev', 'accuracy_diff_ci95_low', 'accuracy_diff_ci95_high'))
            interval = intervals[scenario]
            assert interval['repetitions'] == 10000 and interval['seed'] == 20261109
            check_metrics(interval, dict(difference_accuracy=float(difference.mean()), ci95_low=float(low), ci95_high=float(high)),
                          ('difference_accuracy', 'ci95_low', 'ci95_high'))
        print('PASS: independent final accuracy, Macro-F1, routing, corrections/harm, frozen threshold hash')
        print('PASS: exported predictions, confusion matrices, all 10000-resample page-bootstrap intervals')
    else:
        print('PENDING: independent test results')


if __name__ == '__main__':
    main()
