"""Frozen RAGTruth QA validation/test protocol, cached scoring and source-level inference."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import sqlite3

import experiment_v2 as shared
import ragtruth_eval as rt

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'code/ragtruth_protocol.json'
SEED = 20261112
POLICY = dict(primary='hallucination_f1', tie_breakers=['accuracy', 'fewest_llm_calls'],
              comparison='confidence <= threshold', max_routing_percent=20,
              tied_confidence='entire group', no_routing_threshold=None,
              folds=5, fold_unit='source_id', seed=SEED, bootstrap_repetitions=10000,
              bootstrap_unit='source_id', invalid_llm='standalone wrong; cascade fallback to JEV')


def expected_protocol(report):
    return dict(version=rt.VERSION, protocol_version=1, request_configs=rt.request_configs(),
                jev_url=shared.pilot.JEV_URL, bailian_url=shared.pilot.BAILIAN_URL,
                upstream_commit=report['upstream_commit'], data_sha256=report['output_sha256'],
                label_policy=report['label_policy'], quality_policy=report['quality_policy'], selection=POLICY,
                pilot_review_sha256=shared.digest(ROOT / 'docs/ragtruth_pilot_review.md'))


def load_data(path, stage):
    if stage not in ('validation', 'test'):
        raise ValueError('Expected formal validation or test')
    path = Path(path)
    report = json.loads((path.parent / 'split_report.json').read_text())
    if report['dataset_version'] != rt.VERSION or shared.digest(path) != report['output_sha256'][stage]:
        raise ValueError('Formal data hash/version differs from split report')
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = set()
    for s in samples:
        if (s['dataset_version'] != rt.VERSION or s['experiment_split'] != stage
                or s['official_split'] != ('train' if stage == 'validation' else 'test')
                or s['quality'] != 'good' or s['source_id'] != s['group_id'] or s['sample_id'] in ids):
            raise ValueError('Invalid source/ID/split in formal data')
        if any(not isinstance(s.get(k), str) or not s[k].strip()
               for k in ('question', 'evidence', 'response', 'source_id', 'sample_id')):
            raise ValueError('Missing input or identifier')
        for a in s['annotations']:
            if not (0 <= a['start'] < a['end'] <= len(s['response']) and
                    s['response'][a['start']:a['end']] == a['text']):
                raise ValueError('Invalid annotation span')
        if (s['gold'] != ('H' if s['annotations'] else 'N') or s['gold_without_implicit_true'] !=
                ('H' if any(not a.get('implicit_true', False) for a in s['annotations']) else 'N')):
            raise ValueError('Gold differs from original annotations')
        ids.add(s['sample_id'])
    stats = report['subsets'][stage]
    if (len(samples) != stats['responses'] or len({s['source_id'] for s in samples}) != stats['sources']
            or dict(Counter(s['gold'] for s in samples)) != stats['labels']):
        raise ValueError('Formal subset counts differ from split report')
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol != expected_protocol(report):
        raise ValueError('Frozen protocol differs from data, review or request settings; create a new experiment version')
    return samples, report, protocol


def load_thresholds(path, protocol):
    artifact = json.loads(Path(path).read_text())
    for key, value in dict(version=rt.VERSION, protocol_sha256=shared.digest(PROTOCOL_PATH),
                           validation_data_sha256=protocol['data_sha256']['validation'],
                           request_configs=protocol['request_configs'], selection=POLICY).items():
        if artifact.get(key) != value:
            raise ValueError(f'Threshold artifact mismatch: {key}')
    if set(artifact['models']) != set(shared.LLMS.values()):
        raise ValueError('Threshold model list mismatch')
    for choice in artifact['models'].values():
        t = choice['confidence_threshold']
        if t is not None and (type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1):
            raise ValueError('Invalid threshold')
        n, calls = choice['samples'], choice['routed']
        if n < 1 or type(calls) is not int or not 0 <= calls <= n * 20 // 100 or (t is None and calls != 0):
            raise ValueError('Threshold selection budget invalid')
    return artifact


def flat(metrics):
    return {k: v for k, v in metrics.items() if k not in ('per_label', 'confusion')}


def attempt_report(checkpoint, out):
    """Public failure accounting without raw answer text, request bodies or credentials."""
    with sqlite3.connect(f'file:{checkpoint}?mode=ro', uri=True) as db:
        rows = [(m, terminal, json.loads(body)) for m, terminal, body in
                db.execute('SELECT model,terminal,body FROM attempts ORDER BY id')]
    failures = Counter((m, r.get('http_status'), r.get('error_type'), r.get('provider_error_code'))
                       for m, terminal, r in rows if not terminal)
    invalid = [dict(model=m, **{k: r.get(k) for k in ('sample_id', 'gold', 'http_status', 'error_type',
                'provider_error_code', 'finish_reason', 'input_tokens', 'output_tokens')})
               for m, terminal, r in rows if terminal and r['status'] != 'ok']
    shared.write_json(out / 'attempt_summary.json', dict(total_attempts=len(rows),
        terminal_outcomes=sum(t for _, t, _ in rows), nonterminal_attempts=sum(not t for _, t, _ in rows),
        nonterminal_errors=[dict(model=m, http_status=h, error_type=e, provider_error_code=c, attempts=n)
                            for (m, h, e, c), n in failures.items()], invalid_terminal=invalid))


def predictions(samples, results, model, cutoff=None, cascade=False):
    if cascade:
        return [shared.cascade_prediction(results[shared.JEV][s['sample_id']], results[model][s['sample_id']], cutoff)
                for s in samples]
    return [shared.prediction(results[model][s['sample_id']]) for s in samples]


def routing(samples, results, model, cutoff):
    base = predictions(samples, results, shared.JEV)
    pred = predictions(samples, results, model, cutoff, True)
    metrics = rt.score(samples, pred)
    routed = [s for s in samples if shared.is_routed(results[shared.JEV][s['sample_id']], cutoff)]
    corrected = sum(b != s['gold'] and p == s['gold'] for s, b, p in zip(samples, base, pred))
    harmed = sum(b == s['gold'] and p != s['gold'] for s, b, p in zip(samples, base, pred))
    return dict(confidence_threshold=cutoff, **flat(metrics), routed=len(routed), routing_rate=len(routed)/len(samples),
                corrected=corrected, harmed=harmed, net=corrected-harmed,
                fallbacks=sum(shared.prediction(results[model][s['sample_id']]) is None for s in routed),
                h_tp=metrics['per_label']['H']['true_positive'], h_support=metrics['per_label']['H']['support'],
                h_predicted=metrics['per_label']['H']['predicted'])


def selection_key(row):
    denominator = row['h_support'] + row['h_predicted']
    return (Fraction(2 * row['h_tp'], denominator) if denominator else Fraction(0), row['correct'], -row['routed'])


def sweep(samples, results, model):
    groups = Counter()
    for s in samples:
        row = results[shared.JEV][s['sample_id']]
        if (shared.prediction(row) not in rt.LABELS or type(row.get('confidence')) not in (float, int)
                or not math.isfinite(row['confidence']) or not 0 <= row['confidence'] <= 1):
            raise ValueError('Selection requires valid JEV labels and finite confidence')
        groups[row['confidence']] += 1
    thresholds, used = [None], 0
    for threshold, count in sorted(groups.items()):
        if used + count > len(samples) * POLICY['max_routing_percent'] // 100:
            break
        used += count
        thresholds.append(threshold)
    rows = [routing(samples, results, model, t) for t in thresholds]
    return rows, max(rows, key=selection_key)


def grouped_folds(samples):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample['source_id']].append(sample)
    if len(groups) < 5:
        raise ValueError('At least five independent sources required')
    order = sorted(groups)
    random.Random(SEED).shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))
    totals = Counter(s['gold'] for s in samples)
    sizes, labels, assignment = [0]*5, [Counter() for _ in range(5)], {}
    for group in order:
        hist, n = Counter(s['gold'] for s in groups[group]), len(groups[group])
        def loss(fold):
            return ((2*sizes[fold]*n+n*n)/len(samples)**2 +
                    sum((2*labels[fold][g]*hist[g]+hist[g]**2)/max(1, totals[g])**2 for g in rt.LABELS),
                    sizes[fold], fold)
        fold = min(range(5), key=loss)
        assignment[group] = fold
        sizes[fold] += n
        labels[fold].update(hist)
    return assignment


def cross_validate(samples, results):
    assignment, rows = grouped_folds(samples), []
    oof = {m: {} for m in shared.LLMS.values()}
    for fold in range(5):
        train = [s for s in samples if assignment[s['source_id']] != fold]
        held = [s for s in samples if assignment[s['source_id']] == fold]
        baseline = rt.score(held, predictions(held, results, shared.JEV))
        for model in shared.LLMS.values():
            _, best = sweep(train, results, model)
            row = routing(held, results, model, best['confidence_threshold'])
            rows.append(dict(model=model, fold=fold+1, train_samples=len(train), held_sources=len({s['source_id'] for s in held}),
                             train_routed=best['routed'], train_h_f1=best['hallucination_f1'],
                             baseline_h_f1=baseline['hallucination_f1'], baseline_accuracy=baseline['accuracy'],
                             delta_h_f1=row['hallucination_f1']-baseline['hallucination_f1'],
                             delta_accuracy=row['accuracy']-baseline['accuracy'], **row))
            for s, pred in zip(held, predictions(held, results, model, best['confidence_threshold'], True)):
                oof[model][s['sample_id']] = pred
    metrics = {m: rt.score(samples, [oof[m][s['sample_id']] for s in samples]) for m in shared.LLMS.values()}
    return assignment, rows, metrics, oof


def bootstrap(samples, baseline, candidate, repetitions=10000):
    """Paired source clusters; recompute micro accuracy and H F1 for every resample."""
    import numpy as np
    groups = defaultdict(lambda: np.zeros(8, dtype=np.int64))
    for s, b, p in zip(samples, baseline, candidate):
        h = s['gold'] == 'H'
        groups[s['source_id']] += [1, h, b == s['gold'], h and b == 'H', b == 'H',
                                 p == s['gold'], h and p == 'H', p == 'H']
    values = np.array([groups[g] for g in sorted(groups)])
    def differences(v):
        base_d, other_d = v[..., 1]+v[..., 4], v[..., 1]+v[..., 7]
        base_f = np.divide(2*v[..., 3], base_d, out=np.zeros_like(base_d, dtype=float), where=base_d != 0)
        other_f = np.divide(2*v[..., 6], other_d, out=np.zeros_like(other_d, dtype=float), where=other_d != 0)
        return np.stack(((v[..., 5]-v[..., 2])/v[..., 0], other_f-base_f), axis=-1)
    rng, estimates = np.random.default_rng(SEED), []
    for start in range(0, repetitions, 256):
        indices = rng.integers(0, len(values), size=(min(256, repetitions-start), len(values)))
        estimates.append(differences(values[indices].sum(axis=1)))
    low, high = np.quantile(np.concatenate(estimates), [.025, .975], axis=0)
    point = differences(values.sum(axis=0))
    return dict(source_groups=len(groups), repetitions=repetitions, seed=SEED,
                method='paired source_id cluster percentile bootstrap; response-weighted metrics',
                **{name: dict(difference=float(point[i]), ci95_low=float(low[i]), ci95_high=float(high[i]))
                   for i, name in enumerate(('accuracy', 'hallucination_f1'))})


def scenario_reports(out, samples, results, choices, config, with_bootstrap=False):
    import numpy as np
    metrics, rows, sensitivity, by_generator, intervals = {}, [], [], [], {}
    base = predictions(samples, results, shared.JEV)
    exported = [dict(sample_id=s['sample_id'], source_id=s['source_id'], generator_model=s['generator_model'],
                     gold=s['gold'], gold_without_implicit_true=s['gold_without_implicit_true'],
                     confidence=results[shared.JEV][s['sample_id']]['confidence'], predictions={}) for s in samples]
    scenarios = [('jev_only', shared.JEV, False)] + [(f'llm_only/{a}', m, False) for a, m in shared.LLMS.items()] + [
        (f'cascade/{a}', m, True) for a, m in shared.LLMS.items()]
    for name, model, is_cascade in scenarios:
        cutoff = choices[model]['confidence_threshold'] if is_cascade else None
        pred = predictions(samples, results, model, cutoff, is_cascade)
        metrics[name] = rt.score(samples, pred)
        route = routing(samples, results, model, cutoff) if is_cascade else None
        used_jev, used_llm, delays = [], [], []
        for s in samples:
            sid = s['sample_id']
            jr = results[shared.JEV][sid]
            lr = results[model][sid]
            uses_j = is_cascade or model == shared.JEV
            uses_l = model != shared.JEV and (not is_cascade or shared.is_routed(jr, cutoff))
            if uses_j:
                used_jev.append(jr)
            if uses_l:
                used_llm.append(lr)
            delays.append((jr['latency_ms'] if uses_j else 0)+(lr['latency_ms'] if uses_l else 0))
        ju = shared.usage_and_cost(used_jev)
        lu = shared.usage_and_cost(used_llm, shared.pilot.RATES.get(model))
        row = dict(scenario=name, model=model, confidence_threshold=cutoff, **flat(metrics[name]),
                   routed=route['routed'] if route else None, routing_rate=route['routing_rate'] if route else None,
                   corrected=route['corrected'] if route else None, harmed=route['harmed'] if route else None,
                   net=route['net'] if route else None, fallbacks=route['fallbacks'] if route else None,
                   jev_calls=len(used_jev), llm_calls=len(used_llm),
                   mean_latency_ms=float(np.mean(delays)), p95_latency_ms=float(np.percentile(delays, 95)),
                   **{f'jev_{k}': v for k, v in ju.items() if k not in ('calls', 'estimated_cost_usd_low', 'estimated_cost_usd_high')},
                   **{f'llm_{k}': v for k, v in lu.items() if k != 'calls'},
                   estimated_total_cost_usd_low=None if used_jev else lu['estimated_cost_usd_low'],
                   estimated_total_cost_usd_high=None if used_jev else lu['estimated_cost_usd_high'])
        rows.append(row)
        sensitivity.append(dict(scenario=name, **flat(rt.score(samples, pred, 'gold_without_implicit_true'))))
        if with_bootstrap:
            intervals[name] = bootstrap(samples, base, pred)
        for generator in sorted({s['generator_model'] for s in samples}):
            indices = [i for i, s in enumerate(samples) if s['generator_model'] == generator]
            by_generator.append(dict(scenario=name, generator_model=generator,
                                     **flat(rt.score([samples[i] for i in indices], [pred[i] for i in indices]))))
        for e, p in zip(exported, pred):
            e['predictions'][name] = p
    shared.write_json(out / 'metrics.json', metrics)
    shared.write_json(out / 'majority_baseline.json', rt.score(samples, ['N']*len(samples)))
    shared.write_csv(out / 'summary.csv', rows)
    shared.write_csv(out / 'sensitivity_without_implicit_true.csv', sensitivity)
    shared.write_csv(out / 'by_generator.csv', by_generator)
    if with_bootstrap:
        shared.write_json(out / 'bootstrap.json', intervals)
    shared.pilot.atomic_text(out / 'predictions.jsonl', ''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in exported))
    shared.write_json(out / 'report_metadata.json', dict(**config, complete=True,
        rates_usd_per_million_tokens=shared.pilot.RATES,
        cost_note='Historical VitaminC price assumptions, not current billing. JEV unknown; totals including JEV null. Missing token usage never free.',
        latency_note='Concurrent terminal request durations; cascade sums JEV plus routed LLM. Excludes retry, queue and orchestration.',
        scope='Fixed existing RAG answers; response-level detection, not own retrieval/generation or span localization.',
        sensitivity_note='Same fixed predictions/thresholds rescored excluding implicit_true; no retuning.',
        scenario_note='Full model collection costs differ from reconstructed deployment scenarios; terminal usage excludes retries.'))
    return rows, metrics, intervals


def validation_report(out, samples, results, config, protocol):
    all_points, choices = [], {}
    for model in shared.LLMS.values():
        points, choices[model] = sweep(samples, results, model)
        all_points.extend(dict(model=model, **p) for p in points)
    folds, fold_rows, oof_metrics, oof = cross_validate(samples, results)
    shared.write_csv(out / 'threshold_sweep.csv', all_points)
    shared.write_csv(out / 'cross_validation.csv', fold_rows)
    shared.write_json(out / 'folds.json', dict(seed=SEED, source_id_to_fold=folds,
        folds=[dict(fold=f+1, sources=sum(v == f for v in folds.values()),
                    samples=sum(folds[s['source_id']] == f for s in samples),
                    labels=dict(Counter(s['gold'] for s in samples if folds[s['source_id']] == f))) for f in range(5)]))
    shared.write_json(out / 'oof_metrics.json', oof_metrics)
    shared.pilot.atomic_text(out / 'oof_predictions.jsonl', ''.join(json.dumps(dict(sample_id=s['sample_id'],
        source_id=s['source_id'], fold=folds[s['source_id']]+1, predictions={m: oof[m][s['sample_id']] for m in oof}))+'\n' for s in samples))
    artifact = dict(version=rt.VERSION, protocol_sha256=config['protocol_sha256'],
                    validation_data_sha256=config['data_sha256'], request_configs=protocol['request_configs'],
                    selection=POLICY, models=choices)
    shared.write_json(out / 'selected_threshold.json', artifact)
    rows, metrics, _ = scenario_reports(out, samples, results, choices, config)
    from ragtruth_plots import validation_plots
    validation_plots(out, all_points, fold_rows, metrics)
    return rows


def final_report(out, samples, results, config, artifact):
    rows, metrics, intervals = scenario_reports(out, samples, results, artifact['models'], config, True)
    from ragtruth_plots import final_plots
    final_plots(out, rows, metrics, intervals)
    return rows
