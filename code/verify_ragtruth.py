#!/usr/bin/env python3
"""Independently check RAGTruth provenance, isolation and cached pilot scores."""
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_score(samples, predictions, label_key):
    assert len(samples) == len(predictions)
    gold = [s[label_key] for s in samples]
    confusion = {g: {p: sum(a == g and b == (None if p == 'INVALID' else p) for a, b in zip(gold, predictions))
                     for p in ('H', 'N', 'INVALID')} for g in ('H', 'N')}
    f1 = []
    for label in ('H', 'N'):
        denominator = gold.count(label) + predictions.count(label)
        f1.append(2 * confusion[label][label] / denominator if denominator else 0.)
    tp = confusion['H']['H']
    return dict(accuracy=sum(a == b for a, b in zip(gold, predictions)) / len(gold), macro_f1=sum(f1) / 2,
                hallucination_f1=f1[0], hallucination_precision=tp / predictions.count('H') if 'H' in predictions else 0.,
                hallucination_recall=tp / gold.count('H') if 'H' in gold else 0.,
                invalid_predictions=predictions.count(None), confusion=confusion)


def main():
    raw, prepared = ROOT / 'data/ragtruth_raw', ROOT / 'data/ragtruth_qa'
    report = json.loads((prepared / 'split_report.json').read_text())
    for name, expected in report['input_sha256'].items():
        assert digest(raw / name) == expected, name
    originals = {r['id']: r for r in rows(raw / 'dataset/response.jsonl')}
    sources = {s['source_id']: s for s in rows(raw / 'dataset/source_info.jsonl')}
    subsets = {name: rows(prepared / (name + '.jsonl')) for name in report['output_sha256']}
    for name, records in subsets.items():
        assert digest(prepared / (name + '.jsonl')) == report['output_sha256'][name]
        for row in records:
            original = originals[row['sample_id']]
            assert row['annotations'] == original['labels']
            assert row['gold'] == ('H' if original['labels'] else 'N')
            assert row['gold_without_implicit_true'] == ('H' if any(not a.get('implicit_true', False) for a in original['labels']) else 'N')
            assert row['response'] == original['response']
            source = sources[row['source_id']]
            assert source['task_type'] == 'QA'
            assert row['question'] == source['source_info']['question'] and row['evidence'] == source['source_info']['passages']
            assert original['split'] == ('test' if name == 'test' else row['official_split'])
            assert (row['quality'] != 'good') == (name == 'excluded_quality')
            for span in row['annotations']:
                assert row['response'][span['start']:span['end']] == span['text']
        if name in report['subsets']:
            expected = report['subsets'][name]
            assert len(records) == expected['responses']
            assert len({r['source_id'] for r in records}) == expected['sources']
            assert Counter(r['gold'] for r in records) == expected['labels']
    for a, b in itertools.combinations(('prompt_dev', 'validation', 'test', 'train_reserved'), 2):
        for key in ('source_id', 'question', 'evidence'):
            normalize = lambda row: ' '.join(row[key].split()).casefold()
            assert not {normalize(s) for s in subsets[a]} & {normalize(s) for s in subsets[b]}, (a, b, key)
    pilot = subsets['pilot']
    assert len({s['source_id'] for s in pilot}) == len(pilot)
    assert {s['sample_id'] for s in pilot} <= {s['sample_id'] for s in subsets['prompt_dev']}
    assert all(s['official_split'] == 'train' for s in pilot)
    expected_test = {r['id'] for r in originals.values() if r['split'] == 'test' and r['quality'] == 'good'
                     and sources[r['source_id']]['task_type'] == 'QA'}
    assert {s['sample_id'] for s in subsets['test']} == expected_test
    print('PASS: upstream/output hashes, original spans and labels, source/question/context isolation, reserved official test')
    out = ROOT / 'code/ragtruth_pilot_results'
    if not (out / 'report_metadata.json').exists():
        print('PENDING: complete real API pilot results')
        return
    metadata = json.loads((out / 'report_metadata.json').read_text())
    with sqlite3.connect(f'file:{out / "pilot_checkpoint.sqlite3"}?mode=ro', uri=True) as db:
        config = json.loads(db.execute('SELECT manifest FROM config').fetchone()[0])
        attempted = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
        terminal = [json.loads(body) for body, in db.execute('SELECT body FROM attempts WHERE terminal=1')]
    assert attempted <= 200 and len(terminal) == 4 * len(pilot)
    assert config['data_sha256'] == digest(prepared / 'pilot.jsonl')
    assert config['split_report_sha256'] == digest(prepared / 'split_report.json')
    assert metadata['complete'] and metadata['selected_threshold'] is None
    indexed = {(r['model'], r['sample_id']): r for r in terminal}
    assert len(indexed) == len(terminal)
    exported = rows(out / 'predictions.jsonl')
    assert [r['sample_id'] for r in exported] == [s['sample_id'] for s in pilot]
    detailed = json.loads((out / 'metrics.json').read_text())
    for filename, gold_key in [('summary.csv', 'gold'), ('sensitivity_without_implicit_true.csv', 'gold_without_implicit_true')]:
        with (out / filename).open() as handle:
            table = list(csv.DictReader(handle))
        assert len(table) == 4
        for row in table:
            records = [indexed[row['model'], s['sample_id']] for s in pilot]
            predictions = [r['prediction'] if r['status'] == 'ok' else None for r in records]
            expected = independent_score(pilot, predictions, gold_key)
            for key, value in expected.items():
                if key != 'confusion':
                    assert abs(float(row[key]) - value) < 1e-12, (row['model'], key)
            if gold_key == 'gold':
                assert detailed[row['model']]['confusion'] == expected['confusion']
                assert [r['predictions'][row['model']] for r in exported] == predictions
                assert int(row['calls']) == len(pilot)
                for field in ('input_tokens', 'output_tokens'):
                    observed = [r[field] for r in records if type(r.get(field)) is int and r[field] >= 0]
                    assert int(row[field + '_observed']) == sum(observed)
                    assert int(row['calls_with_' + field]) == len(observed)
                assert abs(float(row['mean_latency_ms']) - sum(r['latency_ms'] for r in records) / len(records)) < 1e-9
    jev_model = 'jev-1.14.0'
    with (out / 'exploratory_routing.csv').open() as handle:
        sweep = list(csv.DictReader(handle))
    for row in sweep:
        cutoff = float(row['confidence_threshold']) if row['confidence_threshold'] else None
        predicted, routed, corrected, harmed, fallback = [], 0, 0, 0, 0
        for s in pilot:
            j, other = indexed[jev_model, s['sample_id']], indexed[row['model'], s['sample_id']]
            prediction = j['prediction']
            if cutoff is not None and j['confidence'] <= cutoff:
                routed += 1
                if other['status'] == 'ok':
                    prediction = other['prediction']
                else:
                    fallback += 1
            predicted.append(prediction)
            corrected += int(prediction == s['gold'] and j['prediction'] != s['gold'])
            harmed += int(prediction != s['gold'] and j['prediction'] == s['gold'])
        assert routed <= len(pilot) * 20 // 100
        expected = independent_score(pilot, predicted, 'gold')
        for key, value in dict(routed=routed, corrected=corrected, harmed=harmed, net=corrected-harmed, fallbacks=fallback,
                               accuracy=expected['accuracy'], hallucination_f1=expected['hallucination_f1']).items():
            assert abs(float(row[key]) - value) < 1e-12
    print(f'PASS: {len(terminal)} cached API outcomes, {attempted} attempts, independent H/N metrics, sensitivity scores and exported predictions')
    print('PASS: token/latency summaries and exploratory routing points recomputed from individual responses')


if __name__ == '__main__':
    main()
