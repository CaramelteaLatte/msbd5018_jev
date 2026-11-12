"""RAGTruth data boundaries, binary scoring and API adapters; no live requests."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import experiment_v2 as shared
import ragtruth_eval as rt
from run_ragtruth_pilot import check_budget

spec = importlib.util.spec_from_file_location('prepare_ragtruth', ROOT / 'data/prepare_ragtruth.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def fixture():
    sources, responses = [], []
    for i in range(5):
        sources.append(dict(source_id=str(i), task_type='QA', source_info=dict(question=f'Question {i}', passages=f'Passages {i}')))
        for j in range(2):
            responses.append(dict(id=f'{i}-{j}', source_id=str(i), model=f'generator{j}', quality='good',
                                  split='test' if i == 4 else 'train', temperature=0.7, response='Some text',
                                  labels=[dict(start=0, end=4, text='Some', label_type='Subtle Baseless Info', implicit_true=True)] if j else []))
    return sources, responses


def record(sample, prediction, confidence=.1):
    return dict(status='ok', prediction=prediction, correct=sample['gold'] == prediction,
                gold=sample['gold'], confidence=confidence, latency_ms=100, input_tokens=10, output_tokens=1)


class Dataset(unittest.TestCase):
    def test_source_isolation_reproducibility_and_original_labels(self):
        source, response = fixture()
        splits, _, audit = prep.prepare_rows(source, response, prompt_sources=1, validation_sources=1)
        repeated, _, _ = prep.prepare_rows(list(reversed(source)), list(reversed(response)), prompt_sources=1, validation_sources=1)
        self.assertEqual(splits, repeated)
        self.assertTrue(all(n == 0 for n in audit['source_overlap'].values()))
        self.assertEqual({r['source_id'] for r in splits['test']}, {'4'})
        for rows in splits.values():
            for sample in rows:
                self.assertEqual(sample['gold'], 'H' if sample['annotations'] else 'N')
                self.assertEqual(sample['gold_without_implicit_true'], 'N')
        self.assertEqual(len(splits['pilot']), 1)

    def test_bad_quality_is_audited_not_relabeled(self):
        source, response = fixture()
        response[0]['quality'] = 'incorrect_refusal'
        splits, excluded, audit = prep.prepare_rows(source, response, prompt_sources=1, validation_sources=1)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(audit['excluded_responses'], 1)
        self.assertTrue(all(s['quality'] == 'good' for rows in splits.values() for s in rows))

    def test_offsets_and_duplicate_context_fail(self):
        source, response = fixture()
        response[1]['labels'][0]['end'] = 3
        with self.assertRaisesRegex(ValueError, 'span'):
            prep.prepare_rows(source, response, prompt_sources=1, validation_sources=1)
        source, response = fixture()
        source[1]['source_info']['passages'] = source[0]['source_info']['passages']
        with self.assertRaisesRegex(ValueError, 'Duplicate QA passages'):
            prep.prepare_rows(source, response, prompt_sources=1, validation_sources=1)

    def test_cross_official_split_source_rejected(self):
        source, response = fixture()
        response[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'both train and test'):
            prep.prepare_rows(source, response, prompt_sources=1, validation_sources=1)


class BinaryDetection(unittest.TestCase):
    def test_requests_never_include_gold_or_annotations(self):
        sample = dict(question='Question', evidence='Evidence', response='Answer', gold='SECRET_GOLD',
                      annotations='SECRET_SPAN', generator_model='SECRET_GENERATOR')
        for model in shared.MODELS:
            payload = rt.make_payload(model, sample)
            self.assertNotIn('SECRET_', json.dumps(payload))
        self.assertEqual(set(rt.make_payload(shared.JEV, sample)['questions']['hallucination']['criteria']), {'H', 'N'})

    def test_binary_parser_and_nonfinite_confidence(self):
        response = dict(answers={'hallucination': dict(choice='H', probabilities={'H': .8, 'N': .2}, confidence=.8)})
        output = dict(gold='H')
        rt.parse_response(shared.JEV, response, output)
        self.assertTrue(output['correct'])
        response['answers']['hallucination']['confidence'] = float('nan')
        with self.assertRaises(ValueError):
            rt.parse_response(shared.JEV, response, dict(gold='H'))
        with self.assertRaises(ValueError):
            rt.parse_response(shared.LLMS['qwen'], {'choices': [{'message': {'content': 'S'}}]}, dict(gold='N'))

    def test_real_transport_uses_task_specific_hooks(self):
        sample = dict(sample_id='a', question='Question', evidence='Evidence', response='Answer', gold='H')
        class FakeReply:
            status, headers = 200, {}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{"answers":{"hallucination":{"choice":"H","probabilities":{"H":0.9,"N":0.1},"confidence":0.9}}}'
        class FakeOpener:
            def open(self, request, timeout):
                self.sent = json.loads(request.data)
                return FakeReply()
        opener = FakeOpener()
        result = rt.run_one(opener, shared.JEV, 'fake-key', sample, 1)
        self.assertEqual(result['prediction'], 'H')
        self.assertEqual(set(opener.sent['state']), {'question', 'passages', 'candidate_answer'})

    def test_invalid_prediction_counts_against_full_denominator(self):
        samples = [dict(gold=g) for g in ('H', 'H', 'N', 'N')]
        metrics = rt.score(samples, ['H', None, 'H', 'N'])
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['hallucination_f1'], .5)
        self.assertEqual(metrics['hallucination_recall'], .5)
        self.assertEqual(metrics['confusion']['H']['INVALID'], 1)
        self.assertEqual(metrics['false_negative'], 1)

    def test_sweep_keeps_ties_whole_and_falls_back(self):
        samples = [dict(sample_id=str(i), gold='H') for i in range(10)]
        jev = {s['sample_id']: record(s, 'H', .1 if i < 2 else .8) for i, s in enumerate(samples)}
        results = {model: {sid: dict(row) for sid, row in jev.items()} for model in shared.MODELS}
        results[shared.LLMS['qwen']]['0'].update(status='error', prediction=None)
        points = rt.exploratory_sweep(samples, results)
        row = next(p for p in points if p['model'] == shared.LLMS['qwen'] and p['confidence_threshold'] == .1)
        self.assertEqual((row['routed'], row['fallbacks'], row['accuracy']), (2, 1, 1.0))
        self.assertTrue(all(p['routed'] in (0, 2) for p in points))

    def test_budget_counts_failed_attempts_and_rejects_mixed_config(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.sqlite3'
            samples = [dict(sample_id=str(i)) for i in range(48)]
            config = {'task': 'ragtruth'}
            with sqlite3.connect(path) as db:
                shared.open_checkpoint(db, config)
                for i in range(9):
                    db.execute('INSERT INTO attempts(model,sample_id,terminal,body) VALUES(?,?,0,?)',
                               (shared.JEV, str(i), '{}'))
                db.commit()
            with self.assertRaisesRegex(ValueError, 'budget'):
                check_budget(path, samples, config, None)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                check_budget(path, samples, {'task': 'vitaminc'}, None)


if __name__ == '__main__':
    unittest.main()
