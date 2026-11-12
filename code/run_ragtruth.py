#!/usr/bin/env python3
"""Collect formal RAGTruth validation/test; test requires a frozen validation artifact."""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import sys

import experiment_v2 as shared
import ragtruth_eval as rt
import ragtruth_experiment as ex


def run(args):
    shared.validate_args(args)
    if args.stage == 'test' and args.thresholds is None:
        raise ValueError('Test requires --thresholds from complete validation')
    if args.stage == 'validation' and args.thresholds is not None:
        raise ValueError('Validation selects thresholds; do not pass --thresholds')
    samples, report, protocol = ex.load_data(args.data, args.stage)
    artifact = ex.load_thresholds(args.thresholds, protocol) if args.stage == 'test' else None
    config = dict(version=rt.VERSION, stage=args.stage, sample_count=len(samples),
                  data_sha256=shared.digest(args.data), protocol_sha256=shared.digest(ex.PROTOCOL_PATH),
                  split_report_sha256=shared.digest(args.data.parent / 'split_report.json'),
                  request_configs=protocol['request_configs'], jev_url=protocol['jev_url'], bailian_url=protocol['bailian_url'],
                  thresholds_sha256=shared.digest(args.thresholds) if artifact else None)
    if args.out is None:
        args.out = ex.ROOT / ('code/ragtruth_validation_results' if args.stage == 'validation' else 'code/ragtruth_final_results')
    checkpoint = args.out / f'{args.stage}_checkpoint.sqlite3'
    if args.plan:
        completed = set()
        if checkpoint.exists():
            with sqlite3.connect(f'file:{checkpoint}?mode=ro', uri=True) as db:
                if json.loads(db.execute('SELECT manifest FROM config').fetchone()[0]) != config:
                    raise ValueError('Checkpoint configuration mismatch')
                completed = set(db.execute('SELECT model,sample_id FROM attempts WHERE terminal=1'))
        models = [{'jev': shared.JEV, **shared.LLMS}[m] for m in dict.fromkeys(args.model)] if args.model else shared.MODELS
        print(json.dumps(dict(stage=args.stage, samples=len(samples), sources=len({s['source_id'] for s in samples}),
                              expected_first_requests=len(samples)*len(models), cached=len(completed),
                              pending=sum((m, s['sample_id']) not in completed for m in models for s in samples)), indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = shared.collect_stage(args, samples, config, checkpoint, request_one=rt.run_one)
        if results is None:
            return
        ex.attempt_report(checkpoint, args.out)
        rows = (ex.validation_report(args.out, samples, results, config, protocol) if args.stage == 'validation'
                else ex.final_report(args.out, samples, results, config, artifact))
        print(json.dumps([{k: r[k] for k in ('scenario', 'samples', 'accuracy', 'hallucination_f1', 'routed')} for r in rows], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shared.add_collection_args(parser, str(ex.ROOT / 'code/ragtruth_validation_results'))
    parser.set_defaults(out=None)
    parser.add_argument('--stage', choices=('validation', 'test'), required=True)
    parser.add_argument('--thresholds', type=Path)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print('Stopped; in-flight requests saved. Resume with identical configuration.', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
