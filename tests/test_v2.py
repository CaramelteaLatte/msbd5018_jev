"""Offline behavioral checks; synthetic model responses never call APIs."""
from collections import Counter
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import experiment_v2 as ev
import run_final_eval as final
import run_threshold as threshold

spec = importlib.util.spec_from_file_location('splitter', ROOT / 'data/split_vitaminc.py')
splitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)


def sample(i, gold=None):
    return dict(sample_id=str(i), gold=gold or ev.LABELS[i % 3], page=f'page-{i // 3}',
                group_id=f'group-{i // 3}', case_id=f'case-{i}', claim=f'claim {i}', evidence=f'evidence {i}')


def response(s, pred, confidence=.1):
    return dict(sample_id=s['sample_id'], gold=s['gold'], status='ok', prediction=pred,
                correct=pred == s['gold'], confidence=confidence,
                latency_ms=100, input_tokens=10, output_tokens=1)


class Sampling(unittest.TestCase):
    def test_flow_can_reallocate_earlier_label_to_meet_other_label(self):
        def row(uid, page, label):
            return dict(unique_id=uid, page=page, label=label)
        groups = [[row('a', 'shared', 'SUPPORTS'), row('b', 'shared', 'REFUTES')],
                  [row('c', 'support-only', 'SUPPORTS')]]
        for seed in range(8):
            selected = splitter.capped_select(groups, 2, Counter(SUPPORTS=1, REFUTES=1), random.Random(seed), 1)
            self.assertEqual(Counter(r['label'] for r in selected), Counter(SUPPORTS=1, REFUTES=1))
            self.assertEqual(len({r['page'] for r in selected}), 2)
        first = splitter.capped_select(groups, 2, Counter(SUPPORTS=1, REFUTES=1), random.Random(5), 1)
        self.assertEqual(first, splitter.capped_select(groups, 2, Counter(SUPPORTS=1, REFUTES=1), random.Random(5), 1))

    def test_impossible_capacity_fails_without_relaxing(self):
        rows = [[dict(unique_id=str(i), page='one', label='SUPPORTS') for i in range(4)]]
        with self.assertRaisesRegex(ValueError, 'No constraints were relaxed'):
            splitter.capped_select(rows, 3, Counter(SUPPORTS=3), random.Random(1), 2)


class Routing(unittest.TestCase):
    def test_boundary_zero_and_null(self):
        row = response(sample(0), 'S', 0)
        self.assertTrue(ev.is_routed(row, 0))
        self.assertFalse(ev.is_routed(row, None))
        row['confidence'] = .48
        self.assertTrue(ev.is_routed(row, .48))
        self.assertFalse(ev.is_routed(row, .479))

    def test_tied_group_cannot_cross_budget(self):
        samples = [sample(i) for i in range(10)]
        jev = {s['sample_id']: response(s, s['gold'], .1 if i < 3 else .9) for i, s in enumerate(samples)}
        points, best = ev.threshold_sweep(samples, jev, jev, 20)
        self.assertEqual(len(points), 1)
        self.assertIsNone(best['confidence_threshold'])

    def test_equal_accuracy_prefers_no_calls(self):
        samples = [sample(i) for i in range(10)]
        jev = {s['sample_id']: response(s, s['gold'], i / 10) for i, s in enumerate(samples)}
        _, best = ev.threshold_sweep(samples, jev, jev, 20)
        self.assertIsNone(best['confidence_threshold'])

    def test_invalid_llm_is_wrong_alone_but_falls_back(self):
        s = sample(0, 'S')
        jev = response(s, 'S')
        bad = dict(status='error', prediction='C', correct=False)
        self.assertIsNone(ev.prediction(bad))
        self.assertEqual(ev.cascade_prediction(jev, bad, .1), 'S')
        m = ev.routing_metrics([s], {'0': jev}, {'0': bad}, .1)
        self.assertEqual((m['llm_fallbacks'], m['harmed'], m['cascade_correct']), (1, 0, 1))

    def test_missing_predictions_stay_in_metric_denominators(self):
        samples = [sample(i) for i in range(3)]
        metrics = ev.score(samples, ['S', None, 'I'])
        self.assertEqual(metrics['accuracy_all'], 2 / 3)
        self.assertEqual(metrics['confusion']['C']['INVALID'], 1)
        self.assertEqual(metrics['macro_f1_all'], 2 / 3)


class Checkpoints(unittest.TestCase):
    def test_explicit_refusal_is_cached_but_other_400s_are_not_terminal(self):
        row = dict(status='error', http_status=400, provider_error_code='data_inspection_failed', prediction=None, correct=None)
        self.assertTrue(ev.terminal_response(ev.LLMS['qwen'], row))
        self.assertFalse(ev.terminal_response(ev.JEV, row))
        self.assertFalse(ev.terminal_response(ev.LLMS['qwen'], dict(row, provider_error_code='invalid_parameter')))
        with sqlite3.connect(':memory:') as db:
            config = {'data': 'same'}
            ev.open_checkpoint(db, config)
            db.execute('INSERT INTO attempts(model,sample_id,terminal,body) VALUES(?,?,0,?)',
                       (ev.LLMS['qwen'], 'a', json.dumps(row)))
            db.commit()
            cached = ev.open_checkpoint(db, config)
            self.assertEqual(len(cached), 1)
            self.assertIsNone(ev.prediction(cached[ev.LLMS['qwen'], 'a']))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0], 1)
            self.assertEqual(len(ev.open_checkpoint(db, config)), 1)

    def test_report_only_never_requests_or_appends_collection_history(self):
        samples = [sample(i) for i in range(3)]
        config = dict(stage='test', sample_count=3)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            checkpoint = out / 'checkpoint.sqlite3'
            history = '[{"workers": 8}]\n'
            (out / 'collection_runs.json').write_text(history)
            with sqlite3.connect(checkpoint) as db:
                ev.open_checkpoint(db, config)
                for model in ev.MODELS:
                    for s in samples:
                        db.execute('INSERT INTO attempts(model,sample_id,terminal,body) VALUES(?,?,1,?)',
                                   (model, s['sample_id'], json.dumps(response(s, s['gold']))))
                db.commit()
            args = SimpleNamespace(out=out, report_only=True)
            with patch.object(ev.pilot, 'run_one', side_effect=AssertionError('API should not run')):
                actual = ev.collect_stage(args, samples, config, checkpoint)
            self.assertEqual(set(actual), set(ev.MODELS))
            self.assertEqual((out / 'collection_runs.json').read_text(), history)

    def test_configuration_changes_are_rejected(self):
        with sqlite3.connect(':memory:') as db:
            ev.open_checkpoint(db, {'data': 'a', 'request': 'x', 'thresholds': 'one'})
            for config in ({'data': 'b', 'request': 'x', 'thresholds': 'one'},
                           {'data': 'a', 'request': 'y', 'thresholds': 'one'},
                           {'data': 'a', 'request': 'x', 'thresholds': 'two'}):
                with self.assertRaises(ValueError):
                    ev.open_checkpoint(db, config)

    def test_resume_reuses_completed_requests_and_preserves_failure(self):
        samples = [sample(i) for i in range(4)]
        with tempfile.TemporaryDirectory() as td, sqlite3.connect(':memory:') as db:
            config = dict(stage='test', sample_count=4)
            completed = ev.open_checkpoint(db, config)
            args = SimpleNamespace(out=Path(td), workers=1, timeout=1, retries=0, no_proxy=True)
            def call(opener, model, key, s, timeout):
                if s['sample_id'] == '2':
                    return dict(sample_id='2', gold=s['gold'], status='error', prediction=None,
                                correct=None, error_type='connection_error')
                return response(s, s['gold'])
            with patch.object(ev.pilot, 'run_one', side_effect=call), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    ev.collect(db, completed, ev.JEV, samples, 'fake', args, config)
            self.assertEqual(len(completed), 2)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM attempts WHERE terminal=0').fetchone()[0], 1)
            with patch.object(ev.pilot, 'run_one', side_effect=lambda o, m, k, s, t: response(s, s['gold'])) as run, contextlib.redirect_stdout(io.StringIO()):
                ev.collect(db, completed, ev.JEV, samples, 'fake', args, config)
                self.assertEqual(run.call_count, 2)
                ev.collect(db, completed, ev.JEV, samples, 'fake', args, config)
                self.assertEqual(run.call_count, 2)


class EndToEndReports(unittest.TestCase):
    def test_offline_validation_artifact_and_fixed_test_reports(self):
        samples = [sample(i) for i in range(30)]
        results = {}
        for index, model in enumerate(ev.MODELS):
            results[model] = {}
            for i, s in enumerate(samples):
                pred = s['gold'] if (i + index) % 4 else ev.LABELS[(ev.LABELS.index(s['gold']) + 1) % 3]
                results[model][s['sample_id']] = response(s, pred, (i % 20) / 20)
        assignments = ev.grouped_folds(samples)
        self.assertEqual(assignments, ev.grouped_folds(list(reversed(samples))))
        self.assertEqual(set(assignments.values()), set(range(5)))
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            config = dict(data_sha256='a' * 64, max_routing_percent=20, fold_seed=ev.SEED,
                          request_configs=ev.request_configs(), jev_url=ev.pilot.JEV_URL, bailian_url=ev.pilot.BAILIAN_URL)
            with contextlib.redirect_stdout(io.StringIO()):
                report = threshold.generate_reports(out, samples, results, config)
            _, cutoffs = final.load_thresholds(out / 'selected_threshold.json')
            assignment, folds, oof = ev.cross_validate(samples, results, 20)
            for model in ev.LLMS.values():
                self.assertEqual(oof[model]['correct'], sum(r['cascade_correct'] for r in folds if r['model'] == model))
                self.assertLessEqual(report['selected'][model]['routed'], 6)
            with contextlib.redirect_stdout(io.StringIO()):
                rows = final.generate_reports(out, samples, results, cutoffs, {})
            self.assertEqual(len(rows), 7)
            for row in rows:
                if row['method'] == 'cascade':
                    expected = rows[0]['correct'] + row['corrected_vs_jev'] - row['harmed_vs_jev']
                    self.assertEqual(row['correct'], expected)
                    self.assertIsNone(row['estimated_cost_usd_low'])
            self.assertTrue((out / 'final_eval_results.png').is_file())
            self.assertTrue((out / 'cross_validation.png').is_file())
            report['request_configs'][ev.JEV]['model'] = 'another-model'
            ev.write_json(out / 'changed.json', report)
            with self.assertRaises(ValueError):
                final.load_thresholds(out / 'changed.json')


if __name__ == '__main__':
    unittest.main()
