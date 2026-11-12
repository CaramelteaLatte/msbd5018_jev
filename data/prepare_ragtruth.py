#!/usr/bin/env python3
"""Download a pinned RAGTruth release and prepare source-disjoint QA subsets."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'c103204b9ce28d6bbad859304bf30de72b8ed8fe'
BASE = f'https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{COMMIT}/'
FILES = {
    'dataset/source_info.jsonl': '0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b',
    'dataset/response.jsonl': 'e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073',
    'LICENSE': 'b7fd7d6bdfe0cbba63c63a310914beb4a4acb8bf08da73849219f45385f5b244',
    'README.md': '3686178d505a38cb20a94827917369bcb9d0aa44cf5154d7ddbac740ee06628d',
}
SPAN_TYPES = {'Evident Baseless Info', 'Subtle Baseless Info', 'Evident Conflict', 'Subtle Conflict'}
SEED = 20261112
VERSION = 'ragtruth-qa-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def normalized(text):
    return ' '.join(text.split()).casefold()


def download(raw):
    manifest = []
    for name, expected in FILES.items():
        path = raw / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(Request(BASE + name, headers={'User-Agent': 'ragtruth-qa-research'}), timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f'Download hash mismatch: {name}')
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_bytes(content)
            temporary.replace(path)
        if digest(path) != expected:
            raise ValueError(f'Local source differs from pinned upstream: {path}')
        manifest.append(dict(path=name, bytes=path.stat().st_size, sha256=expected, url=BASE + name))
    write_json(raw / 'download_manifest.json', dict(repository='https://github.com/ParticleMedia/RAGTruth',
                                                  commit=COMMIT, files=manifest))


def validate_spans(response):
    text, spans = response['response'], response['labels']
    if not isinstance(text, str) or not text.strip() or not isinstance(spans, list):
        raise ValueError(f"Invalid response {response['id']}")
    for span in spans:
        start, end = span['start'], span['end']
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or text[start:end] != span['text'] or span['label_type'] not in SPAN_TYPES):
            raise ValueError(f"Invalid hallucination span in response {response['id']}")
        for flag in ('implicit_true', 'due_to_null'):
            if flag in span and type(span[flag]) is not bool:
                raise ValueError(f'Expected Boolean {flag}')


def prepare_rows(sources, responses, seed=SEED, prompt_sources=48, validation_sources=200):
    if prompt_sources < 1 or validation_sources < 1:
        raise ValueError('Both development source counts must be positive')
    indexed = {s['source_id']: s for s in sources}
    if len(indexed) != len(sources) or len({r['id'] for r in responses}) != len(responses):
        raise ValueError('Duplicate source or response IDs')
    qa_sources = {sid: s for sid, s in indexed.items() if s['task_type'] == 'QA'}
    # Fail explicitly on duplicate question/context identities, rather than silently
    # putting related sources in separate subsets.
    for field in ('question', 'passages'):
        owners = {}
        for sid, source in qa_sources.items():
            text = source['source_info'][field]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'Missing QA {field}: {sid}')
            key = normalized(text)
            if key in owners:
                raise ValueError(f'Duplicate QA {field}: {owners[key]}, {sid}; regroup sources before splitting')
            owners[key] = sid
    official, grouped, excluded = defaultdict(set), defaultdict(list), []
    qa = []
    for row in responses:
        if row['source_id'] not in indexed:
            raise ValueError(f"Missing source: {row['source_id']}")
        if row['source_id'] not in qa_sources:
            continue
        validate_spans(row)
        if row['split'] not in ('train', 'test') or row['quality'] not in ('good', 'incorrect_refusal', 'truncated'):
            raise ValueError('Unknown official split or quality value')
        official[row['source_id']].add(row['split'])
        qa.append(row)
        source = qa_sources[row['source_id']]['source_info']
        sample = dict(dataset_version=VERSION, sample_id=row['id'], source_id=row['source_id'],
                      group_id=row['source_id'], official_split=row['split'], generator_model=row['model'],
                      generator_temperature=row['temperature'], quality=row['quality'], question=source['question'],
                      evidence=source['passages'], response=row['response'], annotations=row['labels'],
                      gold='H' if row['labels'] else 'N',
                      gold_without_implicit_true='H' if any(not s.get('implicit_true', False) for s in row['labels']) else 'N')
        if row['quality'] != 'good':
            excluded.append(dict(sample, exclusion_reason='quality != good (matches upstream baseline eligibility)'))
        else:
            grouped[row['source_id']].append(sample)
    if any(len(values) != 1 for values in official.values()):
        raise ValueError('An official source occurs in both train and test')
    train = sorted(sid for sid in grouped if official[sid] == {'train'})
    if len(train) < prompt_sources + validation_sources:
        raise ValueError('Not enough independent training sources; constraints were not relaxed')
    rng = random.Random(seed)
    rng.shuffle(train)
    assignments = {sid: 'test' for sid in grouped if official[sid] == {'test'}}
    for index, sid in enumerate(train):
        assignments[sid] = ('prompt_dev' if index < prompt_sources else 'validation'
                            if index < prompt_sources + validation_sources else 'train_reserved')
    splits = {name: [] for name in ('prompt_dev', 'validation', 'test', 'train_reserved')}
    for sid in sorted(grouped):
        for sample in sorted(grouped[sid], key=lambda s: s['sample_id']):
            splits[assignments[sid]].append(dict(sample, experiment_split=assignments[sid]))
    # One response per reserved pilot source; choose underrepresented generator
    # models without looking at labels, confidence, or detector predictions.
    pilot, generator_counts = [], Counter()
    for sid in train[:prompt_sources]:
        candidates = sorted(grouped[sid], key=lambda s: s['sample_id'])
        rng.shuffle(candidates)
        selected = min(candidates, key=lambda s: generator_counts[s['generator_model']])
        generator_counts[selected['generator_model']] += 1
        pilot.append(dict(selected, experiment_split='pilot'))
    splits['pilot'] = pilot
    all_ids = [s['sample_id'] for name, rows in splits.items() if name != 'pilot' for s in rows]
    if len(all_ids) != len(set(all_ids)) or len(all_ids) + len(excluded) != len(qa):
        raise ValueError('Response accounting failed')
    overlap = {}
    for a, b in itertools.combinations(('prompt_dev', 'validation', 'test', 'train_reserved'), 2):
        overlap[f'{a}:{b}'] = len({r['source_id'] for r in splits[a]} & {r['source_id'] for r in splits[b]})
    if any(overlap.values()):
        raise ValueError('Source leakage')
    audit = dict(raw_sources=len(sources), raw_responses=len(responses), qa_sources=len(qa_sources), qa_responses=len(qa),
                 quality_counts=dict(Counter(r['quality'] for r in qa)), excluded_responses=len(excluded),
                 raw_qa_labels=dict(Counter('H' if r['labels'] else 'N' for r in qa)),
                 span_types=dict(Counter(s['label_type'] for r in qa for s in r['labels'])),
                 implicit_true_spans=sum(bool(s.get('implicit_true', False)) for r in qa for s in r['labels']),
                 span_offsets_valid=True, source_overlap=overlap, duplicate_questions=0, duplicate_contexts=0,
                 pilot_is_subset_of_prompt_dev=True)
    return splits, excluded, audit


def subset_stats(rows):
    words = sorted(len((r['question'] + ' ' + r['evidence'] + ' ' + r['response']).split()) for r in rows)
    return dict(responses=len(rows), sources=len({r['source_id'] for r in rows}),
                labels=dict(Counter(r['gold'] for r in rows)),
                labels_without_implicit_true=dict(Counter(r['gold_without_implicit_true'] for r in rows)),
                generator_models=dict(Counter(r['generator_model'] for r in rows)),
                input_words_median=words[len(words) // 2] if words else None,
                input_words_max=max(words) if words else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/ragtruth_raw')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/ragtruth_qa')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--prompt-sources', type=int, default=48)
    parser.add_argument('--validation-sources', type=int, default=200)
    args = parser.parse_args()
    saved_report = None
    if args.out.exists() and any(args.out.iterdir()):
        if {p.name for p in args.out.iterdir()} != {'split_report.json'}:
            parser.error('Expanded output already exists; use a new directory to reproduce')
        # A fresh clone contains the tracked manifest but not ignored JSONL data.
        # Rehydrate only if it exactly matches the newly computed split report.
        saved_report = json.loads((args.out / 'split_report.json').read_text(encoding='utf-8'))
    download(args.raw)
    splits, excluded, audit = prepare_rows(read_rows(args.raw / 'dataset/source_info.jsonl'),
                                          read_rows(args.raw / 'dataset/response.jsonl'),
                                          args.seed, args.prompt_sources, args.validation_sources)
    outputs = {name: ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')
               for name, rows in {**splits, 'excluded_quality': excluded}.items()}
    report = dict(dataset_version=VERSION, upstream_commit=COMMIT, seed=args.seed,
                  parameters=dict(prompt_sources=args.prompt_sources, validation_sources=args.validation_sources),
                  input_sha256=FILES, output_sha256={name: hashlib.sha256(content).hexdigest() for name, content in outputs.items()},
                  label_policy='H iff at least one original span exists, including implicit_true; N otherwise. Binary response-level detection.',
                  quality_policy='Only quality=good; other responses retained in excluded_quality.jsonl, never silently relabeled.',
                  sampling_policy='Source-level train partition; official test reserved. Pilot: one response/source, generator-balanced without label selection.',
                  subsets={name: subset_stats(rows) for name, rows in splits.items()}, audit=audit,
                  pilot_initial_calls=4 * len(splits['pilot']), formal_validation_test_initial_calls=4 * (len(splits['validation']) + len(splits['test'])),
                  status='Data prepared; model collection has not been inferred from these files.')
    if saved_report is not None and saved_report != report:
        parser.error('Tracked split report does not match these parameters; use a new output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (args.out / (name + '.jsonl')).write_bytes(content)
    write_json(args.out / 'split_report.json', report)
    print(json.dumps({'subsets': report['subsets'], 'pilot_initial_calls': report['pilot_initial_calls']}, indent=2))


if __name__ == '__main__':
    main()
