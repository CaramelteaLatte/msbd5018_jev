#!/usr/bin/env python3
"""Evaluate fixed Jev confidence routing and standalone models on VitaminC test."""

import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from urllib.request import ProxyHandler, build_opener

import run_pilot as pilot

CUTOFF = 0.48
LLMS = {
    "qwen": "qwen3.8-max",
    "deepseek": "deepseek-v4.1-flash-1017",
    "kimi": "kimi-k3.1-pro",
}


def load_test(path):
    data = path.read_bytes()
    samples = []
    ids = set()
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("experiment_split") != "test":
            raise ValueError(f"{path}:{line_number}: expected the prepared test split.")
        sample_id = row["unique_id"]
        if (not isinstance(sample_id, str) or not sample_id or sample_id in ids
                or row["gold"] not in pilot.CRITERIA
                or any(not isinstance(row[k], str) or not row[k].strip()
                       for k in ("claim", "evidence"))):
            raise ValueError(f"{path}:{line_number}: invalid or duplicate test sample.")
        ids.add(sample_id)
        samples.append(dict(sample_id=sample_id, **{k: row[k] for k in ("claim", "evidence", "gold")}))
    if not samples:
        raise ValueError("The test split is empty.")
    return samples, hashlib.sha256(data).hexdigest()


def open_checkpoint(db, config):
    with db:
        db.execute("CREATE TABLE IF NOT EXISTS config (manifest TEXT NOT NULL)")
        db.execute("""CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY, model TEXT NOT NULL, sample_id TEXT NOT NULL,
            terminal INTEGER NOT NULL, body TEXT NOT NULL
        )""")
        manifest = json.dumps(config, sort_keys=True)
        saved = db.execute("SELECT manifest FROM config").fetchone()
        if saved is None:
            db.execute("INSERT INTO config VALUES (?)", (manifest,))
        elif saved[0] != manifest:
            raise ValueError("Test data or request settings changed. Use a new --out directory.")
    return {
        (model, sample_id): json.loads(body)
        for model, sample_id, body in db.execute(
            "SELECT model, sample_id, body FROM attempts WHERE terminal = 1 ORDER BY id"
        )
    }


def collect(db, completed, opener, model, samples, key, timeout):
    cached = sum((model, sample["sample_id"]) in completed for sample in samples)
    print(f"{model}: {cached}/{len(samples)} already saved", flush=True)
    for index, sample in enumerate(samples, 1):
        pair = (model, sample["sample_id"])
        if pair in completed:
            continue
        row = pilot.run_one(opener, model, key, sample, timeout)
        terminal = row["status"] == "ok" or (
            row.get("http_status") == 200 and row.get("error_type") == "invalid_response"
        )
        with db:
            db.execute("INSERT INTO attempts (model, sample_id, terminal, body) VALUES (?, ?, ?, ?)",
                       (*pair, terminal, json.dumps(row, ensure_ascii=False)))
        if not terminal:
            error = row.get("provider_error_code") or row.get("error_type") or "request_error"
            raise RuntimeError(
                f"{model} {pair[1]}: {error}, request_id={row.get('request_id') or '-'}. "
                "Check api_attempts.jsonl and rerun with the same --out to resume."
            )
        completed[pair] = row
        if index % 100 == 0 or index == len(samples):
            print(f"{model}: {index}/{len(samples)}", flush=True)


def score(samples, predictions):
    count = len(samples)
    labels = tuple(pilot.CRITERIA)
    pairs = [(sample["gold"], prediction) for sample, prediction in zip(samples, predictions)]
    correct = sum(gold == prediction for gold, prediction in pairs)
    failures = predictions.count(None)
    f1 = []
    for label in labels:
        tp = sum(gold == prediction == label for gold, prediction in pairs)
        denominator = sum(gold == label for gold, _ in pairs) + predictions.count(label)
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "samples": count,
        "correct": correct,
        "accuracy_all": correct / count,
        "macro_f1_all": sum(f1) / len(f1),
        "failed_predictions": failures,
        "failure_rate": failures / count,
    }


def observed_tokens(rows, field):
    counts = [row[field] for row in rows if type(row.get(field)) is int and row[field] >= 0]
    return sum(counts) if counts or not rows else None, len(counts)


def usage_and_cost(rows, rates, per_1000_calls=None):
    inputs, input_coverage = observed_tokens(rows, "input_tokens")
    outputs, output_coverage = observed_tokens(rows, "output_tokens")
    result = {
        "calls": len(rows),
        "invalid_responses": sum(row["status"] != "ok" for row in rows),
        "input_tokens_observed": inputs,
        "output_tokens_observed": outputs,
        "calls_with_input_tokens": input_coverage,
        "calls_with_output_tokens": output_coverage,
    }
    if not rows:
        low = high = 0.0
    elif per_1000_calls is not None:
        low = high = len(rows) * per_1000_calls / 1000
    elif rates and input_coverage == output_coverage == len(rows):
        (input_low, output_low), (input_high, output_high) = rates
        low = (inputs * input_low + outputs * output_low) / 1_000_000
        high = (inputs * input_high + outputs * output_high) / 1_000_000
    else:
        low = high = None
    result.update(estimated_cost_usd_low=low, estimated_cost_usd_high=high)
    return result


def summarize(method, samples, jev_rows, llm_rows, llm, jev_rates, jev_per_1000):
    is_cascade = method == "cascade"
    predictions, latencies, llm_calls = [], [], []
    fallbacks = corrected = harmed = 0
    for sample in samples:
        sample_id = sample["sample_id"]
        jev = jev_rows.get(sample_id)
        candidate = llm_rows.get(sample_id)
        final = jev
        elapsed = jev["latency_ms"] if jev else 0.0
        routed = (is_cascade and jev["status"] == "ok" and jev["confidence"] <= CUTOFF)
        if method == "llm_only" or routed:
            llm_calls.append(candidate)
            elapsed += candidate["latency_ms"]
            if routed and candidate["status"] != "ok":
                fallbacks += 1
            else:
                final = candidate
                if routed:
                    corrected += int(not jev["correct"] and candidate["correct"])
                    harmed += int(jev["correct"] and not candidate["correct"])
        predictions.append(final["prediction"])
        latencies.append(elapsed)

    row = dict(method=method, jev_model=pilot.JEV if jev_rows else None, llm_model=llm,
               confidence_cutoff=CUTOFF if is_cascade else None)
    row.update(score(samples, predictions))
    row.update(
        routed_samples=len(llm_calls) if is_cascade else None,
        routing_rate=len(llm_calls) / len(samples) if is_cascade else None,
        llm_fallbacks=fallbacks if is_cascade else None,
        corrected_vs_jev=corrected if is_cascade else None,
        harmed_vs_jev=harmed if is_cascade else None,
        api_calls=len(jev_rows) + len(llm_calls),
        mean_e2e_latency_ms=sum(latencies) / len(samples),
        p95_e2e_latency_ms=sorted(latencies)[math.ceil(0.95 * len(samples)) - 1],
        total_sample_latency_seconds=sum(latencies) / 1000,
    )
    jev_usage = usage_and_cost(list(jev_rows.values()), jev_rates, jev_per_1000)
    llm_usage = usage_and_cost(llm_calls, pilot.RATES.get(llm))
    for provider, usage in (("jev", jev_usage), ("llm", llm_usage)):
        row.update({f"{provider}_{key}": value for key, value in usage.items()})
    for bound in ("low", "high"):
        key = f"estimated_cost_usd_{bound}"
        parts = jev_usage[key], llm_usage[key]
        total = sum(parts) if all(part is not None for part in parts) else None
        row[key] = total
        row[f"estimated_cost_per_sample_usd_{bound}"] = total / len(samples) if total is not None else None
    return row


def export_reports(db, out, config, samples, completed, jev_rates, jev_per_1000):
    attempts = [body for (body,) in db.execute("SELECT body FROM attempts ORDER BY id")]
    pilot.atomic_text(out / "api_attempts.jsonl", "".join(body + "\n" for body in attempts))
    by_model = defaultdict(dict)
    for (model, sample_id), row in completed.items():
        by_model[model][sample_id] = row
    n = len(samples)
    jev = by_model[pilot.JEV]
    summary = []
    if len(jev) == n:
        summary.append(summarize("jev_only", samples, jev, {}, None, jev_rates, jev_per_1000))
    for model in LLMS.values():
        llm_rows = by_model[model]
        if len(llm_rows) == n:
            summary.append(summarize("llm_only", samples, {}, llm_rows, model, jev_rates, jev_per_1000))
            if len(jev) == n:
                summary.append(summarize("cascade", samples, jev, llm_rows, model, jev_rates, jev_per_1000))
    report = dict(
        **config,
        terminal_responses_by_model={model: len(rows) for model, rows in by_model.items()},
        recorded_attempts=len(attempts),
        llm_rates_usd_per_million_tokens={model: pilot.RATES.get(model) for model in LLMS.values()},
        jev_rates_usd_per_million_tokens=jev_rates,
        jev_usd_per_1000_calls=jev_per_1000,
        cost_note="Estimates use pilot rates and reported tokens, without separate cache discounts. "
                  "Blank costs indicate unknown pricing or incomplete usage. "
                  "Scenario totals count terminal responses only. api_attempts.jsonl also records interruptions.",
        latency_note="Reconstructed from terminal request durations. Routed latency is Jev plus LLM. "
                     "Excludes retry delays, queueing, and orchestration. P95 uses nearest rank.",
        scoring_note="Every test row counts. Missing predictions are wrong and contribute false negatives "
                     "for macro-F1. Invalid routed LLM responses fall back to Jev.",
    )
    pilot.atomic_text(out / "run_config.json", json.dumps(report, indent=2) + "\n")
    if not summary:
        return
    temp = out / "summary.csv.tmp"
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    temp.replace(out / "summary.csv")


def run(args):
    samples, digest = load_test(args.data)
    selected = tuple(LLMS[alias] for alias in dict.fromkeys(args.model or LLMS))
    models = (pilot.JEV, *selected)
    if args.plan:
        print(f"Test samples: {len(samples):,} | Jev cutoff: {CUTOFF}")
        print(f"First-pass calls from an empty checkpoint: {len(samples) * len(models):,}")
        for model in models:
            print(f"  {model}: {len(samples):,}")
        print("No API requests sent.")
        return

    keys = {
        pilot.JEV: (pilot.TYPESAFE_API_KEY or os.getenv("TYPESAFE_API_KEY", "")).strip(),
        "bailian": (pilot.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")).strip(),
    }
    if not all(keys.values()):
        raise ValueError("Set TYPESAFE_API_KEY and DASHSCOPE_API_KEY in your environment. "
                         "No requests sent.")
    config = dict(
        split="test", sample_count=len(samples), data_sha256=digest, confidence_cutoff=CUTOFF,
        jev_url=pilot.JEV_URL, bailian_url=pilot.BAILIAN_URL,
        request_configs={model: pilot.make_payload(model, {"claim": "", "evidence": ""})
                         for model in (pilot.JEV, *LLMS.values())},
    )
    jev_rates = (tuple(args.jev_rates),) * 2 if args.jev_rates else None
    opener = build_opener(ProxyHandler({})) if args.no_proxy else build_opener()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This output directory is already in use.") from None
        with sqlite3.connect(args.out / "final_eval_checkpoint.sqlite3") as db:
            completed = open_checkpoint(db, config)
            try:
                for model in models:
                    key = keys[pilot.JEV if model == pilot.JEV else "bailian"]
                    collect(db, completed, opener, model, samples, key, args.timeout)
            finally:
                export_reports(db, args.out, config, samples, completed, jev_rates, args.jev_usd_per_1000_calls)
    print(f"Reports: {args.out.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Keep this script next to run_pilot.py.")
    parser.add_argument("--data", type=Path, required=True, help="prepared test.jsonl")
    parser.add_argument("--out", type=Path, default=Path("final_eval_results"))
    parser.add_argument("--model", choices=tuple(LLMS), action="append",
                        help="select an LLM, repeat to select several (default: all three)")
    parser.add_argument("--plan", action="store_true", help="show the call budget without sending requests")
    parser.add_argument("--timeout", type=int, default=120, help="request timeout in seconds")
    parser.add_argument("--no-proxy", action="store_true", help="disable Python's automatic proxy")
    pricing = parser.add_mutually_exclusive_group()
    pricing.add_argument("--jev-rates", type=float, nargs=2, metavar=("INPUT", "OUTPUT"),
                         help="Jev USD per million input/output tokens")
    pricing.add_argument("--jev-usd-per-1000-calls", type=float, help="Jev USD per 1,000 calls")
    args = parser.parse_args()
    rates = args.jev_rates or [args.jev_usd_per_1000_calls]
    if args.timeout < 1 or any(rate is not None and (not math.isfinite(rate) or rate < 0) for rate in rates):
        parser.error("Timeout must be positive and price rates must be finite and nonnegative.")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped. Rerun with the same --out to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
