#!/usr/bin/env python3
"""Offline, post-hoc cascade comparisons against both JEV and the matching LLM."""
import csv
import hashlib
import json
from pathlib import Path

import ragtruth_experiment as ex
import ragtruth_eval as rt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'code/ragtruth_final_results'


def main():
    prediction_path = OUT / 'predictions.jsonl'
    samples = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    with (OUT / 'summary.csv').open(newline='') as handle:
        summaries = {row['scenario']: row for row in csv.DictReader(handle)}
    thresholds = ROOT / 'code/ragtruth_validation_results/selected_threshold.json'
    config = json.loads((OUT / 'run_config.json').read_text())
    assert config['thresholds_sha256'] == hashlib.sha256(thresholds.read_bytes()).hexdigest()
    assert len(samples) == config['sample_count'] == len({row['sample_id'] for row in samples})
    for name, row in summaries.items():
        score = rt.score(samples, [s['predictions'][name] for s in samples])
        assert score['correct'] == int(row['correct'])
        assert abs(score['hallucination_f1'] - float(row['hallucination_f1'])) < 1e-12
    comparisons = []
    for alias in ('qwen', 'deepseek', 'kimi'):
        name = 'cascade/' + alias
        cascade = summaries[name]
        candidate = [s['predictions'][name] for s in samples]
        for baseline_name in ('jev_only', 'llm_only/' + alias):
            baseline = summaries[baseline_name]
            intervals = ex.bootstrap(samples, [s['predictions'][baseline_name] for s in samples], candidate)
            row = dict(cascade=name, baseline=baseline_name, samples=len(samples),
                       routed=int(cascade['routed']), routing_rate=float(cascade['routing_rate']))
            for metric in ('accuracy', 'hallucination_f1'):
                row[metric + '_baseline'] = float(baseline[metric])
                row[metric + '_cascade'] = float(cascade[metric])
                for key, value in intervals[metric].items():
                    row[metric + '_' + key + '_pp'] = value * 100
            for metric in ('mean_latency_ms', 'p95_latency_ms'):
                b, c = float(baseline[metric]), float(cascade[metric])
                row[metric + '_baseline'] = b
                row[metric + '_cascade'] = c
                row[metric + '_change_percent'] = (c / b - 1) * 100
            comparisons.append(row)
    with (OUT / 'tradeoff_comparisons.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(comparisons)
    metadata = dict(
        analysis='Post-hoc descriptive comparison; thresholds and predictions unchanged',
        data_sha256=config['data_sha256'], thresholds_sha256=config['thresholds_sha256'],
        samples=len(samples), source_groups=len({s['source_id'] for s in samples}),
        bootstrap=dict(repetitions=10000, seed=ex.SEED, unit='source_id',
                       method='paired cluster percentile; response-weighted accuracy and H F1'),
        notes=['Differences are cascade minus the explicitly named baseline.',
               'Latency is reconstructed from cached terminal requests; no real cascade timing experiment or latency confidence intervals.',
               'No acceptable loss margin was prespecified; these results do not establish non-inferiority.',
               'Intervals are pointwise; no multiple-comparison correction.'],
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (prediction_path, OUT / 'summary.csv', OUT / 'run_config.json')})
    (OUT / 'tradeoff_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Saved direct cascade comparisons against JEV and matching LLM; no API calls.')


if __name__ == '__main__':
    main()
