"""Formal protocol tests with synthetic predictions and a mocked transport; no network."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import experiment_v2 as shared
import ragtruth_eval as rt
import ragtruth_experiment as ex


def fixture(n=30):
    samples = [dict(sample_id=str(i), source_id=str(i//3), group_id=str(i//3),
                    gold='H' if i%3 == 0 else 'N', gold_without_implicit_true='H' if i%6 == 0 else 'N',
                    generator_model='g'+str(i%3)) for i in range(n)]
    results = {m: {} for m in shared.MODELS}
    for m in shared.MODELS:
        for i, s in enumerate(samples):
            pred = s['gold'] if i > 3 or m != shared.JEV else ('N' if s['gold'] == 'H' else 'H')
            results[m][s['sample_id']] = dict(status='ok', prediction=pred, correct=pred == s['gold'],
                gold=s['gold'], confidence=0.0 if i < 3 else .1 if i < 6 else .9,
                latency_ms=100, input_tokens=10, output_tokens=1)
    return samples, results


class FormalProtocol(unittest.TestCase):
    def test_boundary_zero_ties_budget_and_invalid_fallback(self):
        samples, results = fixture()
        model = shared.LLMS['qwen']
        results[model]['0'].update(status='error', prediction=None)
        rows, best = ex.sweep(samples, results, model)
        self.assertEqual([r['confidence_threshold'] for r in rows], [None, 0., .1])
        self.assertEqual([r['routed'] for r in rows], [0, 3, 6])
        self.assertEqual(rows[1]['fallbacks'], 1)
        self.assertEqual(best['confidence_threshold'], .1)
        self.assertEqual(ex.predictions(samples, results, model, 0., True)[0], results[shared.JEV]['0']['prediction'])
        # A tie group above budget is not split or sampled.
        for r in results[shared.JEV].values(): r['confidence'] = .2
        rows, best = ex.sweep(samples, results, model)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(best['confidence_threshold'])

    def test_primary_f1_ties_accuracy_then_fewer_calls(self):
        a = dict(h_tp=2, h_support=4, h_predicted=4, correct=8, routed=2)
        b = dict(h_tp=1, h_support=4, h_predicted=1, correct=9, routed=0)
        self.assertGreater(ex.selection_key(a), ex.selection_key(b))
        self.assertGreater(ex.selection_key(dict(a, correct=9)), ex.selection_key(a))
        self.assertGreater(ex.selection_key(dict(a, routed=0)), ex.selection_key(a))
        samples, results = fixture()
        results[shared.LLMS['qwen']] = results[shared.JEV]
        _, best = ex.sweep(samples, results, shared.LLMS['qwen'])
        self.assertIsNone(best['confidence_threshold'])

    def test_nonfinite_jev_rejected(self):
        samples, results = fixture()
        results[shared.JEV]['0']['confidence'] = float('nan')
        with self.assertRaises(ValueError): ex.sweep(samples, results, shared.LLMS['qwen'])

    def test_grouped_folds_reproducible_balanced_and_oof(self):
        samples, results = fixture()
        assignment, rows, metrics, oof = ex.cross_validate(samples, results)
        self.assertEqual(assignment, ex.grouped_folds(list(reversed(samples))))
        counts = Counter(assignment[s['source_id']] for s in samples)
        self.assertEqual(set(counts.values()), {6})
        self.assertEqual(len(rows), 15)
        for m in shared.LLMS.values():
            self.assertEqual(metrics[m], rt.score(samples, [oof[m][s['sample_id']] for s in samples]))
        self.assertTrue(all(r['train_routed'] <= r['train_samples']*20//100 for r in rows))

    def test_cluster_bootstrap_matches_raw_resampling_oracle(self):
        import numpy as np
        samples, results = fixture(12)
        baseline = ex.predictions(samples, results, shared.JEV)
        candidate = ex.predictions(samples, results, shared.LLMS['qwen'])
        candidate[3] = None
        actual = ex.bootstrap(samples, baseline, candidate, repetitions=300)
        groups = sorted({s['source_id'] for s in samples})
        indices = np.random.default_rng(ex.SEED).integers(0, len(groups), size=(300, len(groups)))
        diffs = []
        for draws in indices:
            chosen = [i for group in draws for i,s in enumerate(samples) if s['source_id'] == groups[group]]
            def metric(pred):
                pairs = [(samples[i]['gold'], pred[i]) for i in chosen]
                correct = sum(g == p for g,p in pairs)/len(pairs)
                den = sum(g == 'H' for g,p in pairs)+sum(p == 'H' for g,p in pairs)
                f = 2*sum(g == p == 'H' for g,p in pairs)/den if den else 0
                return correct, f
            b, c = metric(baseline), metric(candidate)
            diffs.append([c[i]-b[i] for i in range(2)])
        low, high = np.quantile(diffs, [.025, .975], axis=0)
        for i,name in enumerate(('accuracy','hallucination_f1')):
            self.assertAlmostEqual(actual[name]['ci95_low'], low[i])
            self.assertAlmostEqual(actual[name]['ci95_high'], high[i])
        identical = ex.bootstrap(samples, baseline, baseline, repetitions=20)
        self.assertEqual(identical['hallucination_f1']['ci95_high'], 0)

    def test_threshold_rejects_request_and_data_changes(self):
        protocol = json.loads(ex.PROTOCOL_PATH.read_text())
        samples, results = fixture()
        choices = {m: ex.sweep(samples, results, m)[1] for m in shared.LLMS.values()}
        artifact = dict(version=rt.VERSION, protocol_sha256=shared.digest(ex.PROTOCOL_PATH),
            validation_data_sha256=protocol['data_sha256']['validation'], request_configs=protocol['request_configs'],
            selection=ex.POLICY, models=choices)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'thresholds.json'
            shared.write_json(path,artifact)
            self.assertEqual(ex.load_thresholds(path,protocol),artifact)
            for field in ('request_configs','validation_data_sha256','protocol_sha256'):
                shared.write_json(path,dict(artifact,**{field:'changed'}))
                with self.assertRaises(ValueError): ex.load_thresholds(path,protocol)

    def test_resume_has_zero_duplicate_requests_and_rejects_config_change(self):
        samples, results = fixture(3)
        calls = []
        def fake(opener, model, key, sample, timeout):
            calls.append((model,sample['sample_id']))
            return results[model][sample['sample_id']]
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'TYPESAFE_API_KEY':'fake','DASHSCOPE_API_KEY':'fake'}):
            args = argparse.Namespace(out=Path(folder), workers=2, timeout=1, retries=0, model=None, no_proxy=True, report_only=False)
            config = dict(version=rt.VERSION, stage='validation', sample_count=len(samples))
            db = args.out/'cache.sqlite3'
            shared.collect_stage(args,samples,config,db,request_one=fake)
            shared.collect_stage(args,samples,config,db,request_one=fake)
            args.report_only = True
            shared.collect_stage(args,samples,config,db,request_one=fake)
            self.assertEqual(len(calls),12)
            self.assertEqual(len(set(calls)),12)
            with self.assertRaises(ValueError):
                shared.collect_stage(args,samples,dict(config,data_sha256='changed'),db,request_one=fake)

    @unittest.skipUnless(all((ROOT/'data/ragtruth_qa'/f'{stage}.jsonl').exists() for stage in ('validation','test')),
                         'Expand the pinned QA data with data/prepare_ragtruth.py for this integration check')
    def test_real_frozen_data_counts(self):
        for stage,n in (('validation',1157),('test',875)):
            samples,_,_ = ex.load_data(ROOT/'data/ragtruth_qa'/f'{stage}.jsonl',stage)
            self.assertEqual(len(samples),n)


if __name__ == '__main__':
    unittest.main()
