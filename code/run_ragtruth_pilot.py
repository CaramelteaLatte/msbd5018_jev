#!/usr/bin/env python3
"""Run a bounded, cached RAGTruth QA pilot; never uses formal validation/test."""
import argparse
import fcntl
import json
from pathlib import Path
import sqlite3
import sys

import experiment_v2 as shared
import ragtruth_eval as rt

ROOT = Path(__file__).resolve().parents[1]
MAX_ATTEMPTS = 200


def check_budget(checkpoint, samples, config, aliases):
    attempted, completed = 0, set()
    if checkpoint.exists():
        with sqlite3.connect(f'file:{checkpoint}?mode=ro', uri=True) as db:
            if json.loads(db.execute('SELECT manifest FROM config').fetchone()[0]) != config:
                raise ValueError('Data or request configuration changed; use a new checkpoint directory')
            attempted = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
            completed = set(db.execute('SELECT model,sample_id FROM attempts WHERE terminal=1'))
    models = [{'jev': shared.JEV, **shared.LLMS}[a] for a in aliases] if aliases else shared.MODELS
    remaining = sum((m, s['sample_id']) not in completed for m in set(models) for s in samples)
    if attempted + remaining > MAX_ATTEMPTS:
        raise ValueError(f'Pilot total-attempt budget {MAX_ATTEMPTS} exceeded: {attempted} already recorded, {remaining} pending. No new requests sent.')
    return dict(max_total_attempts=MAX_ATTEMPTS, recorded_attempts=attempted, pending_first_or_resumed_attempts=remaining,
                complete_pilot_model_samples=4 * len(samples), automatic_retries=0)


def run(args):
    shared.validate_args(args)
    if args.retries != 0:
        raise ValueError('This pilot uses --retries 0 to enforce the 200-attempt budget; resume saved failures explicitly')
    samples, report = rt.load_pilot(args.data)
    config = dict(version=rt.VERSION, stage='pilot', sample_count=len(samples), data_sha256=shared.digest(args.data),
                  split_report_sha256=shared.digest(args.data.parent / 'split_report.json'),
                  upstream_commit=report['upstream_commit'], request_configs=rt.request_configs(),
                  jev_url=shared.pilot.JEV_URL, bailian_url=shared.pilot.BAILIAN_URL,
                  label_policy=report['label_policy'], quality_policy=report['quality_policy'],
                  max_total_attempts=MAX_ATTEMPTS, max_exploratory_routing_percent=20)
    checkpoint = args.out / 'pilot_checkpoint.sqlite3'
    if args.plan:
        print(json.dumps(check_budget(checkpoint, samples, config, args.model), indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.report_only:
            print(json.dumps(check_budget(checkpoint, samples, config, args.model), indent=2), flush=True)
        results = shared.collect_stage(args, samples, config, checkpoint, request_one=rt.run_one)
        if results is not None:
            rows = rt.generate_reports(args.out, samples, results, config)
            print(json.dumps([{k: r[k] for k in ('model', 'samples', 'accuracy', 'hallucination_f1', 'invalid_predictions')}
                              for r in rows], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    shared.add_collection_args(parser, str(ROOT / 'code/ragtruth_pilot_results'))
    parser.set_defaults(retries=0, workers=4)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print('Stopped; in-flight requests saved. Resume with the same arguments.', file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
