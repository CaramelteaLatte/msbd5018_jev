#!/usr/bin/env python3
"""Reproducible, source-group-isolated VitaminC-real experiment subsets.

Python 3.9+, standard library only. No model/API calls. See README.md.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import itertools
import json
from pathlib import Path
import random

LABELS = ('SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO')
MAP = dict(zip(LABELS, ('S', 'C', 'I')))


def norm(value):
    return ' '.join(value.split())


def pair(row):
    return (norm(row['claim']), norm(row['evidence']))


def load(path, origin):
    rows, counts = [], Counter()
    ids = set()
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts['total'] += 1
            counts['revision_' + str(row.get('revision_type'))] += 1
            if row.get('revision_type') != 'real':
                continue
            for key in ('claim', 'evidence', 'page', 'case_id', 'unique_id'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'{path}:{line_no}: missing/empty {key}')
            if row.get('label') not in LABELS:
                raise ValueError(f'{path}:{line_no}: unknown label')
            if row['unique_id'] in ids:
                raise ValueError(f'{path}:{line_no}: duplicate unique_id')
            ids.add(row['unique_id'])
            rows.append(dict(row, source_split=origin))
    return rows, dict(counts)


def clean_and_group(rows):
    # Remove pairs with conflicting labels entirely; never choose a preferred label.
    labels = defaultdict(set)
    for row in rows:
        labels[pair(row)].add(row['label'])
    bad = {key for key, values in labels.items() if len(values) > 1}
    usable = [r for r in rows if pair(r) not in bad]

    # Connected components link page, case, or identical text pair. Components
    # touching both official splits are excluded from BOTH sides before dedup.
    parent = list(range(len(usable)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(usable):
        keys = [('page', norm(row['page']).replace('_', ' ')),
                ('case', row['case_id']), ('pair', pair(row)),
                ('id', row['unique_id'])]
        for key in keys:
            if key in owners:
                parent[root(i)] = root(owners[key])
            else:
                owners[key] = i
    components = defaultdict(list)
    for i, row in enumerate(usable):
        components[root(i)].append(row)
    pools = {'dev': [], 'test': []}
    excluded, duplicate_rows = 0, 0
    for group in components.values():
        origins = {r['source_split'] for r in group}
        if len(origins) > 1:
            excluded += len(group)
            continue
        seen, dedup = set(), []
        for row in group:
            key = pair(row)
            if key in seen:
                duplicate_rows += 1
                continue
            seen.add(key)
            dedup.append(row)
        pools[next(iter(origins))].append(dedup)
    return pools, {
        'conflicting_pair_count': len(bad),
        'conflicting_pair_rows_removed': len(rows) - len(usable),
        'cross_official_split_component_rows_removed': excluded,
        'same_label_duplicate_rows_removed': duplicate_rows,
    }


def quotas(n, counts):
    total = sum(counts.values())
    if not total:
        raise ValueError('No eligible rows available')
    exact = {k: n * counts[k] / total for k in LABELS}
    result = {k: int(exact[k]) for k in LABELS}
    order = sorted(LABELS, key=lambda k: -(exact[k] - result[k]))
    for k in order[:n - sum(result.values())]:
        result[k] += 1
    return result


def select(groups, n, target_counts, rng):
    """Reserve complete randomly ordered components, then sample exact quotas.

    Unselected rows in reserved components stay unused, never enter another
    subset. Label proportions match the cleaned SOURCE pool, up to rounding.
    """
    need = quotas(n, target_counts)
    buckets = {k: [] for k in LABELS}
    consumed = 0
    while any(len(buckets[k]) < need[k] for k in LABELS):
        if consumed >= len(groups):
            raise ValueError(f'Insufficient independent groups for {n} rows; '
                             'reduce requested sizes or supply more data.')
        for row in groups[consumed]:
            buckets[row['label']].append(row)
        consumed += 1
    chosen = []
    for k in LABELS:
        chosen.extend(rng.sample(buckets[k], need[k]))
    rng.shuffle(chosen)
    return chosen, groups[consumed:], consumed


def verify(subsets, sizes):
    keys = {}
    for name, rows in subsets.items():
        if len(rows) != sizes[name]:
            raise ValueError(f'Wrong size: {name}')
        if any(r['revision_type'] != 'real' for r in rows):
            raise ValueError('Non-real row in output')
        keys[name] = {
            'page': {norm(r['page']).replace('_', ' ') for r in rows},
            'case': {r['case_id'] for r in rows},
            'pair': {pair(r) for r in rows},
            'id': {r['unique_id'] for r in rows},
        }
        if len(keys[name]['id']) != len(rows) or len(keys[name]['pair']) != len(rows):
            raise ValueError(f'Duplicates in {name}')
    for a, b in itertools.combinations(subsets, 2):
        for kind in keys[a]:
            if keys[a][kind] & keys[b][kind]:
                raise ValueError(f'Leakage: {a}/{b}, {kind}')


def identity_keys(row):
    return {('page', norm(row['page']).replace('_', ' ')),
            ('case', row['case_id']), ('pair', pair(row)), ('id', row['unique_id'])}


def capped_select(groups, n, target_counts, rng, page_cap):
    """Integral max-flow allocates exact label quotas under per-page capacities.

    Consider every eligible page, not a prefix of whole groups. Seeded page and
    bucket orders make the otherwise non-unique flow allocation reproducible.
    """
    need = quotas(n, target_counts)
    buckets = defaultdict(lambda: defaultdict(list))
    group_ids = {}
    for group in groups:
        group_id = min(r['unique_id'] for r in group)
        for row in group:
            page = norm(row['page']).replace('_', ' ')
            buckets[page][row['label']].append(row)
            group_ids[row['unique_id']] = group_id
    pages = sorted(buckets)
    rng.shuffle(pages)
    graph = defaultdict(dict)

    def edge(a, b, capacity):
        graph[a][b] = capacity
        graph[b].setdefault(a, 0)

    for label, amount in need.items():
        edge('source', ('label', label), amount)
    for page in pages:
        edge(('page', page), 'sink', page_cap)
        for label in LABELS:
            if buckets[page][label]:
                edge(('label', label), ('page', page), len(buckets[page][label]))
    flow = 0
    while flow < n:
        parent = {'source': None}
        queue = deque(['source'])
        while queue and 'sink' not in parent:
            node = queue.popleft()
            for other, capacity in graph[node].items():
                if capacity > 0 and other not in parent:
                    parent[other] = node
                    queue.append(other)
        if 'sink' not in parent:
            raise ValueError(f'Page cap {page_cap} and label quotas permit only '
                             f'{flow}/{n} rows. No constraints were relaxed.')
        path, node = [], 'sink'
        while parent[node] is not None:
            path.append((parent[node], node))
            node = parent[node]
        amount = min(graph[a][b] for a, b in path)
        for a, b in path:
            graph[a][b] -= amount
            graph[b][a] += amount
        flow += amount
    selected = []
    for page in pages:
        for label in LABELS:
            amount = graph[('page', page)].get(('label', label), 0)
            candidates = sorted(buckets[page][label], key=lambda r: r['unique_id'])
            for row in rng.sample(candidates, amount):
                selected.append(dict(row, group_id=group_ids[row['unique_id']]))
    rng.shuffle(selected)
    return selected


def make_v2(args, pools, source_counts):
    old = {}
    for name in ('prompt_dev', 'calibration', 'test'):
        path = args.previous / (name + '.jsonl')
        old[name] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    raw = {r['unique_id']: r for groups in pools.values() for group in groups for r in group}
    for name, rows in old.items():
        for row in rows:
            original = raw.get(row['unique_id'])
            if original is None or any(row.get(k) != v for k, v in original.items()):
                raise ValueError(f'Previous {name} is not an unchanged eligible source row: {row["unique_id"]}')
            if row.get('gold') != MAP[row['label']]:
                raise ValueError(f'Invalid previous gold label in {name}')
    retained = {name: old[name] for name in ('prompt_dev', 'calibration')}
    if (len(retained['prompt_dev']), len(retained['calibration'])) != (args.prompt, args.calibration):
        raise ValueError('v2 retains prompt/calibration exactly; requested sizes must match previous files')
    blocked = {
        'dev': set().union(*(identity_keys(r) for rows in retained.values() for r in rows)),
        'test': set().union(*(identity_keys(r) for r in old['test'])),
    }
    eligible = {origin: [group for group in groups
                         if not any(identity_keys(row) & blocked[origin] for row in group)]
                for origin, groups in pools.items()}
    rng = random.Random(args.seed)
    subsets = dict(retained)
    for name, origin, size in [('threshold_validation', 'dev', args.validation),
                              ('test', 'test', args.test_size)]:
        rows = capped_select(eligible[origin], size, source_counts[origin], rng, args.page_cap)
        subsets[name] = [dict(row, experiment_split=name, gold=MAP[row['label']], dataset_version='v2')
                         for row in rows]
        assert max(Counter(norm(r['page']).replace('_', ' ') for r in rows).values()) <= args.page_cap
    verify(subsets, {name: len(rows) for name, rows in subsets.items()})
    if any(identity_keys(r) & blocked['test'] for r in subsets['test']):
        raise ValueError('v2 test overlaps previously evaluated test')
    details = {
        'page_cap': args.page_cap,
        'previous_sha256': {name: hashlib.sha256((args.previous / (name + '.jsonl')).read_bytes()).hexdigest()
                            for name in old},
        'remaining_pools': {origin: {'rows': sum(map(len, groups)), 'groups': len(groups),
                                    'pages': len({norm(r['page']).replace('_', ' ') for g in groups for r in g})}
                            for origin, groups in eligible.items()},
        'retained_splits': ['prompt_dev', 'calibration'],
        'calibration_status': 'Reserved only; no fitted calibrator. Routing uses raw confidence.',
        'checks': 'PASS: real only; exact label quotas; page caps; no page/case/pair/id overlap; test excludes old test components',
        'policy': 'Seeded integral label-to-page max-flow, then random rows within page/label; '
                  'full components excluded for retained development sets and old test; no prediction-based selection.',
    }
    return subsets, details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev', type=Path, required=True)
    p.add_argument('--test', type=Path, required=True)
    p.add_argument('--out', type=Path, help='default: vitaminc_splits for v1; data/vitaminc_splits_v2 for v2')
    p.add_argument('--mode', choices=('v1', 'v2'), default='v1')
    p.add_argument('--previous', type=Path, default=Path('data/vitaminc_splits'))
    p.add_argument('--page-cap', type=int, default=20)
    p.add_argument('--seed', type=int)
    for name, default in [('prompt', 200), ('calibration', 2000),
                          ('validation', 2000), ('test-size', None)]:
        p.add_argument('--' + name, type=int, default=default)
    args = p.parse_args()
    if args.out is None:
        args.out = Path('data/vitaminc_splits_v2' if args.mode == 'v2' else 'vitaminc_splits')
    if args.seed is None:
        args.seed = 20261109 if args.mode == 'v2' else 20261015
    if args.test_size is None:
        args.test_size = 2000 if args.mode == 'v2' else 5000
    if args.page_cap < 1:
        p.error('--page-cap must be positive')
    sizes = {'prompt_dev': args.prompt, 'calibration': args.calibration,
             'threshold_validation': args.validation, 'test': args.test_size}
    if min(sizes.values()) <= 0:
        p.error('All sizes must be positive')
    if args.out.exists() and any(args.out.iterdir()):
        p.error('Output directory must be empty (use a new --out path)')
    dev, dev_stats = load(args.dev, 'dev')
    test, test_stats = load(args.test, 'test')
    pools, cleaning = clean_and_group(dev + test)
    rng = random.Random(args.seed)
    for groups in pools.values():
        rng.shuffle(groups)
    source_counts = {
        origin: Counter(r['label'] for g in groups for r in g)
        for origin, groups in pools.items()
    }
    subsets, groups_consumed = {}, {}
    details = {}
    if args.mode == 'v2':
        subsets, details = make_v2(args, pools, source_counts)
    else:
        for name, size in sizes.items():
            origin = 'test' if name == 'test' else 'dev'
            selected, pools[origin], consumed = select(
                pools[origin], size, source_counts[origin], rng)
            subsets[name] = [dict(r, experiment_split=name, gold=MAP[r['label']])
                             for r in selected]
            groups_consumed[name] = consumed
    verify(subsets, sizes)
    report = {
        'seed': args.seed, 'input': {'dev': dev_stats, 'test': test_stats},
        'input_sha256': {k: hashlib.sha256(path.read_bytes()).hexdigest()
                         for k, path in [('dev', args.dev), ('test', args.test)]},
        'cleaning': cleaning, 'eligible_label_counts': source_counts,
        'checks': 'PASS: exact sizes; real only; no page/case/pair/id overlap',
        'subsets': {name: {'rows': len(rows),
                          'labels': dict(Counter(r['gold'] for r in rows)),
                          'pages': len({r['page'] for r in rows}),
                          'components_reserved': groups_consumed.get(name),
                          **({'cases': len({r['case_id'] for r in rows}),
                              'largest_page_rows': max(Counter(norm(r['page']).replace('_', ' ') for r in rows).values()),
                              'top_pages': Counter(norm(r['page']).replace('_', ' ') for r in rows).most_common(10)}
                             if args.mode == 'v2' else {})}
                    for name, rows in subsets.items()},
        'policy': 'Whole connected groups reserved; exact label-stratified row '
                  'sampling within them; surplus rows are unused. No fuzzy '
                  'semantic deduplication or model-based sample selection.',
    }
    if args.mode == 'v2':
        report.update(dataset_version='v2', **details)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in subsets.items():
        if args.mode == 'v2' and name in ('prompt_dev', 'calibration'):
            (args.out / (name + '.jsonl')).write_bytes((args.previous / (name + '.jsonl')).read_bytes())
            continue
        with (args.out / (name + '.jsonl')).open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    if args.mode == 'v2':
        report['output_sha256'] = {name: hashlib.sha256((args.out / (name + '.jsonl')).read_bytes()).hexdigest()
                                  for name in subsets}
    (args.out / 'split_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
