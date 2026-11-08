"""Shared v2 collection, scoring and routing. API payloads come from run_pilot."""

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time
from urllib.request import ProxyHandler, build_opener

import run_pilot as pilot

JEV = pilot.JEV
LLMS = {"qwen": "qwen3.8-max", "deepseek": "deepseek-v4.1-flash-1017", "kimi": "kimi-k3.1-pro"}
MODELS = (JEV, *LLMS.values())
LABELS = tuple(pilot.CRITERIA)
SEED = 20261109


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    pilot.atomic_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    temp = Path(str(path) + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def load_samples(path, split):
    samples, seen = [], set()
    mapping = {"SUPPORTS": "S", "REFUTES": "C", "NOT ENOUGH INFO": "I"}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset_version") != "v2" or row.get("experiment_split") != split:
            raise ValueError(f"{path}:{number}: expected a v2 {split} row")
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("unique_id", "claim", "evidence", "page", "case_id", "group_id")):
            raise ValueError(f"{path}:{number}: missing required text or group metadata")
        if (row["unique_id"] in seen or row.get("gold") != mapping.get(row.get("label"))
                or row.get("gold") not in LABELS or row.get("revision_type") != "real"):
            raise ValueError(f"{path}:{number}: duplicate ID or invalid label/revision")
        seen.add(row["unique_id"])
        samples.append(dict(sample_id=row["unique_id"], **{k: row[k] for k in
                            ("claim", "evidence", "gold", "page", "case_id", "group_id")}))
    if not samples:
        raise ValueError("Empty dataset")
    # A case/page must not be assigned to more than one CV grouping unit.
    owners = {}
    for sample in samples:
        for key in (("page", " ".join(sample["page"].split()).replace("_", " ")),
                    ("case", sample["case_id"])):
            if key in owners and owners[key] != sample["group_id"]:
                raise ValueError(f"Inconsistent group_id for {key}")
            owners[key] = sample["group_id"]
    return samples


def request_configs():
    return {model: pilot.make_payload(model, {"claim": "", "evidence": ""}) for model in MODELS}


def make_config(path, samples, stage, **extra):
    return dict(version="v2", stage=stage, sample_count=len(samples), data_sha256=digest(path),
                request_configs=request_configs(), jev_url=pilot.JEV_URL, bailian_url=pilot.BAILIAN_URL,
                **extra)


def provider_refusal(model, row):
    """The provider's explicit sample-level content rejection, not other HTTP 400s."""
    return (model != JEV and row.get('http_status') == 400
            and row.get('provider_error_code') == 'data_inspection_failed')


def terminal_response(model, row):
    if provider_refusal(model, row):
        row['evaluation_disposition'] = 'provider_refusal: standalone wrong; cascade falls back to JEV'
        return True
    return row['status'] == 'ok' or (model != JEV and row.get('http_status') == 200
                                    and row.get('error_type') == 'invalid_response')


def open_checkpoint(db, config):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "config" in tables:
        saved = db.execute("SELECT manifest FROM config").fetchone()
        if saved and json.loads(saved[0]) != config:
            raise ValueError("Data, threshold artifact or request settings changed; use a new output directory")
    with db:
        db.execute("CREATE TABLE IF NOT EXISTS config (manifest TEXT NOT NULL)")
        if not db.execute("SELECT 1 FROM config").fetchone():
            db.execute("INSERT INTO config VALUES (?)", (json.dumps(config, sort_keys=True),))
        db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, model TEXT NOT NULL, sample_id TEXT NOT NULL,
            terminal INTEGER NOT NULL, body TEXT NOT NULL)""")
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS terminal_once ON attempts(model, sample_id) WHERE terminal=1")
        # A saved explicit refusal is already a definitive provider response.
        # Retain its original error fields and attempt ID; do not resubmit it.
        completed_pairs = set(db.execute('SELECT model,sample_id FROM attempts WHERE terminal=1'))
        refused = []
        for attempt_id, model, sid, body in db.execute('SELECT id,model,sample_id,body FROM attempts WHERE terminal=0 ORDER BY id DESC'):
            row = json.loads(body)
            if (model, sid) not in completed_pairs and provider_refusal(model, row):
                terminal_response(model, row)
                refused.append((json.dumps(row, ensure_ascii=False), attempt_id))
                completed_pairs.add((model, sid))
        db.executemany('UPDATE attempts SET terminal=1,body=? WHERE id=?', refused)
    return {(model, sample_id): json.loads(body) for model, sample_id, body in db.execute(
        "SELECT model, sample_id, body FROM attempts WHERE terminal=1 ORDER BY id")}


def progress(db, out, config, completed):
    counts = Counter(model for model, _ in completed)
    failures = db.execute("SELECT COUNT(*) FROM attempts WHERE terminal=0").fetchone()[0]
    write_json(out / "progress.json", dict(
        stage=config["stage"], version="v2", expected_per_model=config["sample_count"],
        terminal_by_model={m: counts[m] for m in MODELS}, nonterminal_attempts=failures,
        complete=all(counts[m] == config["sample_count"] for m in MODELS),
        updated_at_utc=datetime.now(timezone.utc).isoformat()))


def collect(db, completed, model, samples, key, args, config):
    pending_samples = iter(s for s in samples if (model, s["sample_id"]) not in completed)
    already = sum((model, s["sample_id"]) in completed for s in samples)
    print(f"{model}: {already}/{len(samples)} cached; workers={args.workers}", flush=True)
    if already == len(samples):
        return
    stopped = None
    futures = {}
    last_print = time.monotonic()

    def request(sample):
        opener = build_opener(ProxyHandler({})) if args.no_proxy else build_opener()
        return pilot.run_one(opener, model, key, sample, args.timeout)

    def submit_next(pool):
        sample = next(pending_samples, None)
        if sample is not None:
            futures[pool.submit(request, sample)] = (sample, 0)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in range(args.workers):
            submit_next(pool)
        try:
            while futures:
                done, _ = wait(futures, timeout=.5, return_when=FIRST_COMPLETED)
                retry = []
                for future in done:
                    sample, failures = futures.pop(future)
                    row = future.result()
                    terminal = terminal_response(model, row)
                    with db:
                        db.execute("INSERT INTO attempts(model,sample_id,terminal,body) VALUES(?,?,?,?)",
                                   (model, sample["sample_id"], int(terminal), json.dumps(row, ensure_ascii=False)))
                    if terminal:
                        completed[(model, sample["sample_id"])] = row
                    else:
                        transient = row.get("error_type") == "connection_error" or row.get("http_status") in (408, 429, 500, 502, 503, 504)
                        if transient and failures < args.retries and stopped is None:
                            retry.append((sample, failures + 1))
                        else:
                            stopped = f"{model} {sample['sample_id']}: {row.get('http_status')} {row.get('provider_error_code') or row.get('error_type')}: {row.get('error_message', '')}"
                if stopped is None:
                    for sample, failures in retry:
                        time.sleep(min(2 ** failures, 8))
                        futures[pool.submit(request, sample)] = (sample, failures)
                    while len(futures) < args.workers:
                        before = len(futures)
                        submit_next(pool)
                        if len(futures) == before:
                            break
                current = sum((model, s["sample_id"]) in completed for s in samples)
                if time.monotonic() - last_print >= 15 or not futures:
                    print(f"{model}: {current}/{len(samples)} terminal", flush=True)
                    progress(db, args.out, config, completed)
                    last_print = time.monotonic()
        except KeyboardInterrupt:
            # Drain already submitted requests so successful billable work is cached.
            for future, (sample, _) in futures.items():
                if future.cancel():
                    continue
                row = future.result()
                terminal = terminal_response(model, row)
                with db:
                    db.execute("INSERT INTO attempts(model,sample_id,terminal,body) VALUES(?,?,?,?)",
                               (model, sample["sample_id"], int(terminal), json.dumps(row)))
                if terminal:
                    completed[(model, sample["sample_id"])] = row
            raise
    if stopped:
        raise RuntimeError(stopped + "; successful in-flight requests saved. Resume with the same output directory.")


def collect_stage(args, samples, config, checkpoint):
    """Collect selected models, or regenerate reports with --report-only."""
    args.out.mkdir(parents=True, exist_ok=True)
    if args.report_only and not checkpoint.exists():
        raise ValueError("No checkpoint available for --report-only")
    with sqlite3.connect(checkpoint) as db:
        completed = open_checkpoint(db, config)
        expected = {(model, s["sample_id"]) for model in MODELS for s in samples}
        if not set(completed) <= expected:
            raise ValueError("Checkpoint contains unexpected model/sample IDs")
        if not args.report_only:
            history_path = args.out / 'collection_runs.json'
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            history.append(dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                                workers=args.workers, timeout_seconds=args.timeout, retries=args.retries,
                                selected_models=args.model or list(MODELS),
                                cached_terminal_responses=len(completed)))
            write_json(history_path, history)
        try:
            if not args.report_only:
                aliases = {"jev": JEV, **LLMS}
                selected = [aliases[m] for m in dict.fromkeys(args.model)] if args.model else MODELS
                for model in selected:
                    if all((model, s["sample_id"]) in completed for s in samples):
                        continue
                    name = "TYPESAFE_API_KEY" if model == JEV else "DASHSCOPE_API_KEY"
                    key = os.getenv(name, "").strip()
                    if not key:
                        raise ValueError(f"Set {name} in the environment; see README")
                    collect(db, completed, model, samples, key, args, config)
        finally:
            progress(db, args.out, config, completed)
            pilot.atomic_text(args.out / "api_attempts.jsonl", "".join(body + "\n" for (body,) in db.execute("SELECT body FROM attempts ORDER BY id")))
            write_json(args.out / "run_config.json", config)
    if set(completed) != expected:
        print("Partial checkpoint saved. All four models are required for final reports.", flush=True)
        return None
    for (model, sid), row in completed.items():
        sample = next(s for s in samples if s["sample_id"] == sid)
        if row["gold"] != sample["gold"] or (row["status"] == "ok" and row["correct"] != (row["prediction"] == sample["gold"])):
            raise ValueError("Checkpoint label/correctness mismatch")
    return {model: {s["sample_id"]: completed[(model, s["sample_id"])] for s in samples} for model in MODELS}


def prediction(row):
    return row.get("prediction") if row["status"] == "ok" else None


def is_routed(row, cutoff):
    return cutoff is not None and row["status"] == "ok" and row["confidence"] <= cutoff


def cascade_prediction(jev, llm, cutoff):
    return prediction(llm) if is_routed(jev, cutoff) and llm["status"] == "ok" else prediction(jev)


def score(samples, predictions):
    if len(samples) != len(predictions):
        raise ValueError("Prediction count does not match sample count")
    n = len(samples)
    gold = [s["gold"] for s in samples]
    per_label = {}
    for label in LABELS:
        tp = sum(g == p == label for g, p in zip(gold, predictions))
        support, predicted = gold.count(label), predictions.count(label)
        per_label[label] = dict(support=support, true_positive=tp, predicted=predicted,
                                precision=tp / predicted if predicted else 0,
                                recall=tp / support if support else 0,
                                f1=2 * tp / (support + predicted) if support + predicted else 0)
    correct = sum(g == p for g, p in zip(gold, predictions))
    confusion = {label: {p: sum(g == label and out == (None if p == "INVALID" else p)
                               for g, out in zip(gold, predictions)) for p in (*LABELS, "INVALID")}
                 for label in LABELS}
    return dict(samples=n, correct=correct, accuracy_all=correct / n,
                macro_f1_all=sum(v["f1"] for v in per_label.values()) / len(LABELS),
                failed_predictions=predictions.count(None), failure_rate=predictions.count(None) / n,
                per_label=per_label, confusion=confusion)


def routing_metrics(samples, jev, llm, cutoff):
    ids = [s["sample_id"] for s in samples]
    routed = [s for s in ids if is_routed(jev[s], cutoff)]
    corrected = sum(not jev[s]["correct"] and llm[s]["status"] == "ok" and bool(llm[s]["correct"]) for s in routed)
    harmed = sum(jev[s]["correct"] and llm[s]["status"] == "ok" and not llm[s]["correct"] for s in routed)
    return dict(confidence_threshold=cutoff, routed=len(routed), routing_rate=len(routed) / len(ids),
                corrected=corrected, harmed=harmed, net=corrected - harmed,
                llm_fallbacks=sum(llm[s]["status"] != "ok" for s in routed),
                cascade_correct=sum(jev[s]["correct"] for s in ids) + corrected - harmed,
                cascade_accuracy=(sum(jev[s]["correct"] for s in ids) + corrected - harmed) / len(ids))


def threshold_sweep(samples, jev, llm, budget_percent):
    ids = [s["sample_id"] for s in samples]
    groups = defaultdict(list)
    for sid in ids:
        if jev[sid]["status"] != "ok":
            raise ValueError("Threshold selection requires valid Jev responses")
        groups[jev[sid]["confidence"]].append(sid)
    thresholds = [None]
    routed = 0
    for confidence, members in sorted(groups.items()):
        if routed + len(members) > len(ids) * budget_percent // 100:
            break
        routed += len(members)
        thresholds.append(confidence)
    points = [routing_metrics(samples, jev, llm, t) for t in thresholds]
    best = max(points, key=lambda p: (p["cascade_correct"], -p["routed"]))
    return points, best


def grouped_folds(samples, count=5, seed=SEED):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["group_id"]].append(sample)
    if len(groups) < count:
        raise ValueError("Not enough independent groups for folds")
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))
    totals = Counter(s["gold"] for s in samples)
    labels = [Counter() for _ in range(count)]
    sizes = [0] * count
    assignment = {}
    for group in order:
        histogram = Counter(s["gold"] for s in groups[group])
        n = len(groups[group])
        # Greedily minimize change in normalized squared load for size and labels.
        def loss(fold):
            return ((2 * sizes[fold] * n + n * n) / len(samples) ** 2 +
                    sum((2 * labels[fold][g] * histogram[g] + histogram[g] ** 2) /
                        max(1, totals[g]) ** 2 for g in LABELS), sizes[fold], fold)
        fold = min(range(count), key=loss)
        assignment[group] = fold
        labels[fold].update(histogram)
        sizes[fold] += n
    return assignment


def cross_validate(samples, results, budget_percent, seed=SEED):
    assignment = grouped_folds(samples, seed=seed)
    jev = results[JEV]
    rows, oof = [], {model: {} for model in LLMS.values()}
    for fold in range(5):
        train = [s for s in samples if assignment[s["group_id"]] != fold]
        held = [s for s in samples if assignment[s["group_id"]] == fold]
        for model in LLMS.values():
            _, best = threshold_sweep(train, jev, results[model], budget_percent)
            metrics = routing_metrics(held, jev, results[model], best["confidence_threshold"])
            rows.append(dict(model=model, fold=fold + 1, train_samples=len(train), held_samples=len(held),
                             held_groups=len({s["group_id"] for s in held}),
                             train_selected_accuracy=best["cascade_accuracy"],
                             held_jev_accuracy=sum(jev[s["sample_id"]]["correct"] for s in held) / len(held),
                             **metrics))
            for s in held:
                sid = s["sample_id"]
                oof[model][sid] = cascade_prediction(jev[sid], results[model][sid], best["confidence_threshold"])
    summary = {model: score(samples, [oof[model][s["sample_id"]] for s in samples]) for model in LLMS.values()}
    return assignment, rows, summary


def bootstrap(samples, baseline, candidate, repetitions=10000, seed=SEED):
    import numpy as np
    groups = defaultdict(lambda: [0, 0])
    for sample, base, other in zip(samples, baseline, candidate):
        key = " ".join(sample["page"].split()).replace("_", " ")
        groups[key][0] += int(other == sample["gold"]) - int(base == sample["gold"])
        groups[key][1] += 1
    values = np.array(list(groups.values()), dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repetitions, 256):
        indices = rng.integers(0, len(values), size=(min(256, repetitions - start), len(values)))
        selected = values[indices].sum(axis=1)
        estimates.extend((selected[:, 0] / selected[:, 1]).tolist())
    low, high = np.quantile(estimates, [.025, .975])
    return dict(difference_accuracy=float(values[:, 0].sum() / values[:, 1].sum()),
                ci95_low=float(low), ci95_high=float(high), page_groups=len(values),
                repetitions=repetitions, seed=seed, method="paired page-cluster percentile bootstrap")


def usage_and_cost(rows, rates=None, per_1000=None):
    observed = {field: [r[field] for r in rows if type(r.get(field)) is int and r[field] >= 0]
                for field in ("input_tokens", "output_tokens")}
    result = dict(calls=len(rows), invalid_responses=sum(r["status"] != "ok" for r in rows),
                  provider_refusals=sum(r.get('provider_error_code') == 'data_inspection_failed' and r.get('http_status') == 400 for r in rows))
    for field, values in observed.items():
        result[field + "_observed"] = sum(values)
        result["calls_with_" + field] = len(values)
    if not rows:
        costs = (0., 0.)
    elif per_1000 is not None:
        costs = (len(rows) * per_1000 / 1000,) * 2
    elif rates and all(len(v) == len(rows) for v in observed.values()):
        costs = tuple((sum(observed["input_tokens"]) * a + sum(observed["output_tokens"]) * b) / 1_000_000 for a, b in rates)
    else:
        costs = (None, None)
    result.update(estimated_cost_usd_low=costs[0], estimated_cost_usd_high=costs[1])
    return result


def add_collection_args(parser, default_out):
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(default_out))
    parser.add_argument("--model", choices=("jev", *LLMS), action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2, help="retries per interrupted/rate-limited/5xx sample")
    parser.add_argument("--no-proxy", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="read-only request budget; no files or API calls")
    mode.add_argument("--report-only", action="store_true", help="regenerate reports from the checkpoint without API calls")


def validate_args(args):
    if not 1 <= args.workers <= 32 or args.timeout < 1 or not 0 <= args.retries <= 5:
        raise ValueError("workers must be 1..32, timeout positive, retries 0..5")


def plan(samples, args):
    models = len(set(args.model)) if args.model else len(MODELS)
    print(f"Samples: {len(samples)}; selected models: {models}; first-pass requests: {len(samples) * models}")
    print("No API requests sent; no files written.")
