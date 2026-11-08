#!/usr/bin/env python3
"""Select one Jev confidence cutoff on the held-out VitaminC validation split."""

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from urllib.request import ProxyHandler, build_opener

import run_pilot as pilot

LLM_MODELS = ("qwen3.8-max", "deepseek-v4.1-flash-1017", "kimi-k3.1-pro")


def load_validation(path):
    samples = []
    ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("experiment_split") != "threshold_validation":
            raise ValueError(f"Expected threshold_validation at line {line_number}.")
        sample_id = row["unique_id"]
        if (not isinstance(sample_id, str) or not sample_id or sample_id in ids
                or row["gold"] not in pilot.CRITERIA
                or any(not isinstance(row[k], str) or not row[k].strip()
                       for k in ("claim", "evidence"))):
            raise ValueError(f"Invalid or duplicate validation sample at line {line_number}.")
        ids.add(sample_id)
        samples.append({"sample_id": sample_id, "gold": row["gold"],
                        "claim": row["claim"], "evidence": row["evidence"]})
    if not samples:
        raise ValueError("The validation file is empty.")
    return samples


def register_config(db, samples):
    stub = {"claim": "", "evidence": ""}
    config = {
        "sample_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest(),
        "sample_count": len(samples),
        "jev_url": pilot.JEV_URL,
        "bailian_url": pilot.BAILIAN_URL,
        "request_configs": {model: pilot.make_payload(model, stub)
                            for model in (pilot.JEV, *LLM_MODELS)},
    }
    manifest = json.dumps(config, sort_keys=True)
    db.execute("CREATE TABLE IF NOT EXISTS config (manifest TEXT NOT NULL)")
    db.execute("""CREATE TABLE IF NOT EXISTS results (
        model TEXT NOT NULL, sample_id TEXT NOT NULL, body TEXT NOT NULL,
        PRIMARY KEY(model, sample_id)
    )""")
    saved = db.execute("SELECT manifest FROM config").fetchone()
    if saved is None:
        with db:
            db.execute("INSERT INTO config VALUES (?)", (manifest,))
    elif saved[0] != manifest:
        previous = json.loads(saved[0])
        # Accept the original four-model checkpoint if only GLM was removed.
        previous["request_configs"].pop("glm-5.3-prime", None)
        if previous != config:
            raise ValueError("Validation data or model settings changed. Use a new --out directory.")
        with db:
            db.execute("UPDATE config SET manifest = ?", (manifest,))
    return config


def collect(db, opener, model, samples, key, timeout):
    records = {
        sample_id: json.loads(body)
        for sample_id, body in db.execute("SELECT sample_id, body FROM results WHERE model = ?", (model,))
    }
    for index, sample in enumerate(samples, 1):
        sample_id = sample["sample_id"]
        if sample_id in records:
            continue
        row = pilot.run_one(opener, model, key, sample, timeout)
        # An HTTP 200 with an invalid LLM label uses the already available Jev result.
        if (model != pilot.JEV and row["status"] != "ok"
                and row.get("http_status") == 200 and row.get("error_type") == "invalid_response"):
            row["status"] = "fallback_to_jev"
        if row["status"] not in {"ok", "fallback_to_jev"}:
            if row.get("error_type") == "connection_error":
                retry_note = "The provider may have received the interrupted request."
            else:
                retry_note = ""
            message = row.get("error_message", "")
            request_id = row.get("request_id") or "-"
            raise RuntimeError(
                f"{model} {sample_id}: {row.get('provider_error_code') or row.get('error_type')} "
                f"({message}; request_id={request_id}). Rerun the same command to resume. {retry_note}"
            )
        with db:
            db.execute("INSERT INTO results VALUES (?, ?, ?)", (model, sample_id, json.dumps(row)))
        records[sample_id] = row
        if index % 25 == 0 or index == len(samples):
            print(f"{model}: {index}/{len(samples)}", flush=True)
    return records


def eligible_groups(jev, max_routes):
    ordered = sorted(jev, key=lambda sample_id: (jev[sample_id]["confidence"], sample_id))
    groups = []
    routed = 0
    for confidence, members in itertools.groupby(ordered, key=lambda sample_id: jev[sample_id]["confidence"]):
        members = list(members)
        if routed + len(members) > max_routes:
            break
        groups.append((confidence, members))
        routed += len(members)
    return groups


def evaluate(jev, llms, groups):
    n = len(jev)
    baseline = sum(row["correct"] for row in jev.values())
    correct = dict.fromkeys(llms, baseline)
    failures = dict.fromkeys(llms, 0)
    routed = 0
    points = []

    def append_point(threshold):
        points.append({
            "confidence_threshold": threshold,
            "routed": routed,
            "routing_rate": routed / n,
            "mean_cascade_accuracy": sum(correct.values()) / (n * len(llms)),
            "cascade_correct": correct.copy(),
            "llm_invalid_fallbacks": failures.copy(),
        })

    append_point(None)
    for threshold, members in groups:
        routed += len(members)
        for sample_id in members:
            jev_correct = jev[sample_id]["correct"]
            for model, predictions in llms.items():
                result = predictions[sample_id]
                if result["status"] == "ok":
                    correct[model] += int(result["correct"]) - int(jev_correct)
                else:
                    failures[model] += 1
        append_point(threshold)

    # Integer counts avoid floating-point tie-breaking. Prefer fewer calls on ties.
    best = max(points, key=lambda row: (sum(row["cascade_correct"].values()), -row["routed"]))
    return baseline, points, best


def write_plot(plt, out, n, baseline, points, best, budget_pct):
    fig, (accuracy, routing) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1], layout="constrained",
    )
    fig.suptitle("Jev confidence threshold sweep | validation set")
    numeric = points[1:]
    accuracy.axhline(100 * baseline / n, color="#52525b", linestyle="--",
                     label=f"Jev only ({baseline / n:.1%})")
    if numeric:
        x = [row["confidence_threshold"] for row in numeric]
        for model in best["cascade_correct"]:
            accuracy.step(x, [100 * row["cascade_correct"][model] / n for row in numeric],
                          where="post", marker="." if len(x) < 5 else None, label=model)
        accuracy.step(x, [100 * row["mean_cascade_accuracy"] for row in numeric],
                      where="post", color="#18181b", linewidth=2.2,
                      marker="." if len(x) < 5 else None, label="Mean across three LLMs")
        routing.step(x, [100 * row["routing_rate"] for row in numeric], where="post",
                     color="#0f766e", marker="." if len(x) < 5 else None)
        if len(x) == 1:
            pad = min(0.04, max(0.005, x[0] * 0.05))
            accuracy.set_xlim(max(0, x[0] - pad), min(1, x[0] + pad))
    else:
        accuracy.set_xlim(0, 1)
        accuracy.text(0.5, 0.5, "No confidence group fits the routing budget",
                      ha="center", transform=accuracy.transAxes)
    cutoff = best["confidence_threshold"]
    if cutoff is not None:
        for axis in (accuracy, routing):
            axis.axvline(cutoff, color="#be123c", linewidth=1, linestyle=":")
        accuracy.scatter([cutoff], [100 * best["mean_cascade_accuracy"]],
                         color="#be123c", zorder=5)
        accuracy.set_title(f"Selected: confidence <= {cutoff:.6g} | routed {best['routed']}/{n}",
                           loc="left", fontsize=10)
    else:
        accuracy.set_title("Selected: Jev only (zero LLM calls)", loc="left", fontsize=10)
    accuracy.set_ylabel("Accuracy over all validation samples (%)")
    accuracy.legend(loc="best", fontsize=8, ncols=2)
    routing.axhline(budget_pct, color="#52525b", linestyle="--", linewidth=1)
    routing.set_ylim(0, max(1, budget_pct * 1.12))
    routing.set_ylabel("Routed (%)")
    routing.set_xlabel("Raw Jev confidence cutoff (route if confidence <= cutoff)")
    for axis in (accuracy, routing):
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    path = out / "threshold_curve.png"
    temp = out / "threshold_curve.tmp.png"
    fig.savefig(temp, dpi=180)
    plt.close(fig)
    temp.replace(path)


def write_reports(out, n, baseline, points, best, budget_pct, config, plt):
    models = list(best["cascade_correct"])
    temp = out / "threshold_sweep.csv.tmp"
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["confidence_threshold", "routed", "routing_rate", "jev_accuracy",
                         "mean_cascade_accuracy", *[f"{model}_accuracy" for model in models],
                         *[f"{model}_invalid_fallbacks" for model in models]])
        for row in points:
            writer.writerow([
                row["confidence_threshold"], row["routed"], row["routing_rate"], baseline / n,
                row["mean_cascade_accuracy"], *[row["cascade_correct"][m] / n for m in models],
                *[row["llm_invalid_fallbacks"][m] for m in models],
            ])
    temp.replace(out / "threshold_sweep.csv")
    report = {
        "split": "threshold_validation",
        "validation_size": n,
        "score": "raw Jev confidence",
        "rule": "Route when confidence <= threshold. Null threshold means never route.",
        "max_routing_rate": budget_pct / 100,
        "objective": "Maximize mean accuracy across the three cascades; ties prefer fewer calls.",
        "baseline_jev_correct": baseline,
        "baseline_jev_accuracy": baseline / n,
        "selected": best,
        "selected_cascade_accuracy": {m: best["cascade_correct"][m] / n for m in models},
        "validation_sample_sha256": config["sample_sha256"],
        "request_configs": config["request_configs"],
    }
    write_plot(plt, out, n, baseline, points, best, budget_pct)
    pilot.atomic_text(out / "selected_threshold.json", json.dumps(report, indent=2) + "\n")
    print(f"Jev accuracy: {baseline / n:.2%}; selected mean cascade accuracy: "
          f"{best['mean_cascade_accuracy']:.2%}")
    print(f"Selected threshold: {best['confidence_threshold']}; "
          f"routing: {best['routed']}/{n} ({best['routing_rate']:.2%})")
    print(f"Reports: {out.resolve()}")


def run(args):
    samples = load_validation(args.data)
    max_routes = len(samples) * args.max_routing_percent // 100
    if args.plan:
        print(f"Validation samples: {len(samples)}")
        print(f"Planned calls: {len(samples)} Jev + up to {max_routes * len(LLM_MODELS)} LLM")
        print("Plan only. No API requests sent.")
        return
    jev_key = (pilot.TYPESAFE_API_KEY or os.getenv("TYPESAFE_API_KEY", "")).strip()
    llm_key = (pilot.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not jev_key or not llm_key:
        raise ValueError("Set TYPESAFE_API_KEY and DASHSCOPE_API_KEY in your environment. No requests sent.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ValueError("Run python3 -m pip install matplotlib first. No requests sent.") from None
    opener = build_opener(ProxyHandler({})) if args.no_proxy else build_opener()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This output directory is in use by another run.") from None
        with sqlite3.connect(args.out / "threshold_checkpoint.sqlite3") as db:
            config = register_config(db, samples)
            jev = collect(db, opener, pilot.JEV, samples, jev_key, args.timeout)
            groups = eligible_groups(jev, max_routes)
            eligible = {sample_id for _, members in groups for sample_id in members}
            routed_samples = [sample for sample in samples if sample["sample_id"] in eligible]
            print(f"Eligible for the sweep: {len(routed_samples)}/{len(samples)} "
                  f"(equal confidence values stay together).", flush=True)
            llms = {model: collect(db, opener, model, routed_samples, llm_key, args.timeout)
                    for model in LLM_MODELS}
            baseline, points, best = evaluate(jev, llms, groups)
            write_reports(args.out, len(samples), baseline, points, best,
                          args.max_routing_percent, config, plt)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Keep this script next to run_pilot.py. Rerun the same command to resume.",
    )
    parser.add_argument("--data", type=Path, required=True, help="path to threshold_validation.jsonl")
    parser.add_argument("--out", type=Path, default=Path("threshold_results"))
    parser.add_argument("--max-routing-percent", type=int, default=20, metavar="0..100")
    parser.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")
    parser.add_argument("--plan", action="store_true", help="show the request count without calling APIs")
    parser.add_argument("--no-proxy", action="store_true", help="disable Python's automatic proxy")
    args = parser.parse_args()
    if not 0 <= args.max_routing_percent <= 100 or args.timeout < 1:
        parser.error("Routing percent must be 0..100 and timeout must be positive.")
    try:
        run(args)
        return 0
    except KeyboardInterrupt:
        print("Stopped. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
