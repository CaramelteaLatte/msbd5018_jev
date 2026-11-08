#!/usr/bin/env python3
"""Offline audit of existing VitaminC data and checkpoints. Never calls APIs.

Run from any directory: python code/audit_experiments.py
Only writes new audit artifacts under --out; source datasets/results are read-only.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sqlite3

JEV = "jev-1.14.0"
LLMS = ("qwen3.8-max", "deepseek-v4.1-flash-1017", "kimi-k3.1-pro")
LABELS = "SCI"
LABEL_MAP = dict(zip(("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"), LABELS))
CUTOFF = 0.48


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    return " ".join(text.split())


def row_keys(row):
    return {"id": row["unique_id"], "case": row["case_id"],
            "page": norm(row["page"]).replace("_", " "),
            "pair": (norm(row["claim"]), norm(row["evidence"]))}


def dataset_stats(rows):
    pages = Counter(row_keys(row)["page"] for row in rows)
    return {
        "rows": len(rows), "labels": dict(Counter(row["gold"] for row in rows)),
        "pages": len(pages), "cases": len({row["case_id"] for row in rows}),
        "revision_types": dict(Counter(row["revision_type"] for row in rows)),
        "top_pages": pages.most_common(12),
        "largest_page_share": max(pages.values()) / len(rows),
        "unique_claims": len({norm(row["claim"]) for row in rows}),
        "unique_evidence": len({norm(row["evidence"]) for row in rows}),
    }


def read_checkpoint(path, table, where=""):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        records = [json.loads(body) for (body,) in db.execute(
            f"SELECT body FROM {table} {where}")]
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        config = json.loads(db.execute("SELECT manifest FROM config").fetchone()[0]) if "config" in tables else None
    by_model = defaultdict(dict)
    for row in records:
        by_model[row["model"]][row["sample_id"]] = row
    return records, by_model, config


def accuracy(rows, ids):
    return sum(bool(rows[s].get("correct")) for s in ids) / len(ids) if ids else None


def paired(jev, llm, ids):
    corrected = sum(not jev[s]["correct"] and bool(llm[s].get("correct")) for s in ids)
    harmed = sum(jev[s]["correct"] and not llm[s].get("correct") for s in ids)
    return {"n": len(ids), "jev_correct": sum(bool(jev[s]["correct"]) for s in ids),
            "llm_correct": sum(bool(llm[s].get("correct")) for s in ids),
            "jev_accuracy": accuracy(jev, ids), "llm_accuracy": accuracy(llm, ids),
            "corrected": corrected, "harmed": harmed, "net": corrected - harmed}


def score(gold, prediction):
    f1 = []
    for label in LABELS:
        tp = sum(g == p == label for g, p in zip(gold, prediction))
        denom = gold.count(label) + prediction.count(label)
        f1.append(2 * tp / denom if denom else 0)
    return {"correct": sum(g == p for g, p in zip(gold, prediction)),
            "accuracy_all": sum(g == p for g, p in zip(gold, prediction)) / len(gold),
            "macro_f1_all": sum(f1) / len(f1)}


def cluster_interval(data, jev, llm, cascade, unit, repetitions):
    """Paired percentile cluster bootstrap; exploratory, conditional on this test."""
    groups = defaultdict(lambda: [0, 0])
    for sample_id, row in jev.items():
        other = llm[sample_id] if not cascade or row["confidence"] <= CUTOFF else row
        group = groups[data[sample_id][unit]]
        group[0] += int(bool(other.get("correct"))) - int(bool(row["correct"]))
        group[1] += 1
    values = list(groups.values())
    rng = random.Random(20261109)
    estimates = []
    for _ in range(repetitions):
        sampled = rng.choices(values, k=len(values))
        estimates.append(sum(v[0] for v in sampled) / sum(v[1] for v in sampled))
    estimates.sort()
    return {"unit": unit, "groups": len(groups), "repetitions": repetitions,
            "difference_llm_or_cascade_minus_jev": sum(v[0] for v in values) / sum(v[1] for v in values),
            "percentile_95_interval": [estimates[int(.025 * repetitions)], estimates[int(.975 * repetitions)]]}


def write_plot(report, subgroup_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    colors = ["#c75b39", "#c88e6a", "#dfa98c", "#ebc3ae", "#f1dacb", "#e8e8e8"]
    for y, name in enumerate(("threshold_validation", "test")):
        dataset = report["datasets"][name]
        counts = [count for _, count in dataset["top_pages"][:5]]
        values = [100 * n / dataset["rows"] for n in counts + [dataset["rows"] - sum(counts)]]
        start = 0
        for color, value in zip(colors, values):
            left.barh(y, value, left=start, color=color, edgecolor="white", height=.5)
            if value > 7:
                left.text(start + value / 2, y, f"{value:.1f}%", ha="center", va="center", fontsize=10)
            start += value
    left.set_yticks([0, 1], [f"Validation\n{report['datasets']['threshold_validation']['pages']} pages",
                            f"Test\n{report['datasets']['test']['pages']} pages"])
    left.invert_yaxis()
    left.set_xlim(0, 100)
    left.set_xlabel("Share of samples (%)")
    left.set_title("Page concentration\nTop 5 pages, then all other pages", loc="left", fontsize=12)
    italy = next(row for row in subgroup_rows if row["stage"] == "validation" and row["model"] == LLMS[0]
                 and row["group"] == "routed_page" and row["value"] == "2020 coronavirus pandemic in Italy")
    validation = report["runs"]["validation"]["models"][LLMS[0]]["routed"]
    test = report["runs"]["test"]["models"][LLMS[0]]["routed"]
    net = [italy["net"], validation["net"] - italy["net"], test["net"]]
    right.barh(range(3), net, color=["#1c776e", "#75aaa4", "#c75b39"], height=.5)
    for y, value in enumerate(net):
        right.text(value + (.5 if value >= 0 else -.5), y, f"{value:+d}", va="center",
                   ha="left" if value >= 0 else "right", fontweight="bold")
    right.set_yticks(range(3), [f"Validation: Italy\n{italy['n']} routed",
                               f"Validation: other pages\n{validation['n'] - italy['n']} routed",
                               f"Test: all pages\n{test['n']} routed"])
    right.invert_yaxis()
    right.axvline(0, color="#888888", linewidth=.8)
    right.set_xlim(min(net) - 3, max(net) + 5)
    right.set_xlabel("Corrected minus harmed samples")
    right.set_title("Qwen routing gain at confidence <= 0.48\n18 of 19 validation gains come from one page", loc="left", fontsize=12)
    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "audit_overview.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--plot", action="store_true", help="also render an overview; requires matplotlib")
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        parser.error("Use at least 100 bootstrap repetitions")
    root = args.root.resolve()
    out = args.out or root / "code/audit_results"
    subsets = {p.stem: read_jsonl(p) for p in sorted((root / "data/vitaminc_splits").glob("*.jsonl"))}
    split_report = json.loads((root / "data/vitaminc_splits/split_report.json").read_text())
    report = {"cutoff": CUTOFF, "datasets": {}, "checks": {}, "runs": {}}
    checks = report["checks"]
    keys = {}
    for name, rows in subsets.items():
        report["datasets"][name] = dataset_stats(rows)
        keys[name] = {k: {row_keys(row)[k] for row in rows} for k in ("id", "case", "page", "pair")}
        checks[name] = {
            "duplicate_ids": len(rows) - len(keys[name]["id"]),
            "duplicate_pairs": len(rows) - len(keys[name]["pair"]),
            "gold_mapping_errors": sum(row["gold"] != LABEL_MAP[row["label"]] for row in rows),
            "reported_rows": split_report["subsets"][name]["rows"],
            "current_rows": len(rows),
        }
    checks["prepared_split_overlaps"] = {
        f"{a}/{b}": {k: len(keys[a][k] & keys[b][k]) for k in keys[a]}
        for a, b in itertools.combinations(subsets, 2)
    }

    # Replay the original splitter in memory, including its exact RNG sequence.
    spec = importlib.util.spec_from_file_location("audit_splitter", root / "data/split_vitaminc.py")
    splitter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(splitter)
    official = {}
    for origin in ("dev", "test"):
        path = root / f"data/vitaminc/{origin}.jsonl"
        rows, _ = splitter.load(path, origin)
        official[origin] = rows
        checks[f"official_{origin}_sha256_matches"] = digest(path) == split_report["input_sha256"][origin]
    raw = {row["unique_id"]: row for rows in official.values() for row in rows}
    for name, rows in subsets.items():
        checks[name]["source_content_mismatches"] = sum(
            row["unique_id"] not in raw or any(row.get(k) != v for k, v in raw[row["unique_id"]].items())
            for row in rows)
    pools, cleaning = splitter.clean_and_group(official["dev"] + official["test"])
    checks["cleaning_matches_report"] = cleaning == split_report["cleaning"]
    rng = random.Random(split_report["seed"])
    for groups in pools.values():
        rng.shuffle(groups)
    counts = {origin: Counter(row["label"] for group in groups for row in group) for origin, groups in pools.items()}
    for name in ("prompt_dev", "calibration", "threshold_validation", "test"):
        origin = "test" if name == "test" else "dev"
        selected, pools[origin], _ = splitter.select(pools[origin], split_report["subsets"][name]["rows"], counts[origin], rng)
        rebuilt = [dict(row, experiment_split=name, gold=LABEL_MAP[row["label"]]) for row in selected]
        positions = {row["unique_id"]: i for i, row in enumerate(rebuilt)}
        current = subsets[name]
        checks[name].update(
            original_split_reproduced_exactly=rebuilt == current,
            current_is_subset_of_original=all(row["unique_id"] in positions for row in current),
            current_is_original_prefix=rebuilt[:len(current)] == current,
        )
    # Inspect official train overlap without assuming it trained any API model.
    train_overlaps = {name: Counter() for name in subsets}
    with (root / "data/vitaminc/train.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rk = row_keys(row)
            for name, ks in keys.items():
                for k in ks:
                    if rk[k] in ks[k]:
                        train_overlaps[name][k] += 1
    checks["official_train_overlapping_rows"] = {name: {k: count[k] for k in keys[name]} for name, count in train_overlaps.items()}

    configs, all_predictions, subgroup_rows, disagreements = {}, {}, [], []
    definitions = {
        "pilot": ("pilot_results/checkpoint.sqlite3", "attempts", "ORDER BY id", "prompt_dev"),
        "validation": ("threshold_results/threshold_checkpoint.sqlite3", "results", "", "threshold_validation"),
        "test": ("final_eval_results/final_eval_checkpoint.sqlite3", "attempts", "WHERE terminal=1 ORDER BY id", "test"),
    }
    for stage, (path, table, where, subset) in definitions.items():
        records, by_model, config = read_checkpoint(root / "code" / path, table, where)
        configs[stage], all_predictions[stage] = config, by_model
        data = {row["unique_id"]: row for row in subsets[subset]}
        jev = by_model[JEV]
        ids = list(jev)
        low = [s for s in ids if jev[s]["confidence"] <= CUTOFF]
        run = {"models": {}, "routing_count": len(low),
               "routed_labels": dict(Counter(data[s]["gold"] for s in low)),
               "routed_pages": Counter(data[s]["page"] for s in low).most_common(),
               "gold_mismatches": sum(r["gold"] != data[r["sample_id"]]["gold"] for r in records),
               "correct_flag_mismatches": sum(r["correct"] != (r["prediction"] == r["gold"]) for r in records if r["status"] == "ok"),
               "confidence_vs_rescaled_max_probability_max_residual": max(abs(r["confidence"] - (1.5 * max(r["probabilities"].values()) - .5)) for r in jev.values())}
        report["runs"][stage] = run
        for model, rows in by_model.items():
            common = [s for s in ids if s in rows]
            routed = [s for s in low if s in rows]
            run["models"][model] = {
                "saved": len(rows), "status": dict(Counter(r["status"] for r in rows.values())),
                "returned_model": dict(Counter(r.get("model_returned") for r in rows.values())),
                "finish_reason": dict(Counter(r.get("finish_reason") for r in rows.values())),
                "paired_available": paired(jev, rows, common), "routed": paired(jev, rows, routed),
                "per_label_available": {g: paired(jev, rows, [s for s in common if data[s]["gold"] == g]) for g in LABELS},
                "per_label_routed": {g: paired(jev, rows, [s for s in routed if data[s]["gold"] == g]) for g in LABELS},
                "confusion_available": {g: dict(Counter(rows[s]["prediction"] for s in common if data[s]["gold"] == g)) for g in LABELS},
            }
            if model == JEV:
                continue
            groupings = [("all_available", "all", common), ("routed", "all", routed)]
            groupings += [("routed_page", page, [s for s in routed if data[s]["page"] == page]) for page in sorted({data[s]["page"] for s in routed})]
            groupings += [("confidence_bin", f"({lo}, {hi}]", [s for s in common if lo < jev[s]["confidence"] <= hi]) for lo, hi in ((-1, .2), (.2, .48), (.48, .66), (.66, 1))]
            for kind, value, group_ids in groupings:
                subgroup_rows.append(dict(stage=stage, model=model, group=kind, value=value, **paired(jev, rows, group_ids)))
            if model == LLMS[0]:
                for s in routed:
                    if jev[s]["correct"] != rows[s]["correct"]:
                        disagreements.append(dict(stage=stage, **data[s], jev_prediction=jev[s]["prediction"],
                                                  qwen_prediction=rows[s]["prediction"], confidence=jev[s]["confidence"],
                                                  change="corrected" if rows[s]["correct"] else "harmed"))
        if stage == "validation":
            samples = [dict(sample_id=r["unique_id"], **{k: r[k] for k in ("gold", "claim", "evidence")}) for r in subsets[subset]]
            checks["validation_data_hash_matches"] = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest() == config["sample_sha256"]
        if stage == "test":
            checks["test_data_hash_matches"] = digest(root / "data/vitaminc_splits/test.jsonl") == config["data_sha256"]

    checks["validation_and_test_request_configs_equal"] = configs["validation"]["request_configs"] == configs["test"]["request_configs"]
    checks["validation_and_test_endpoints_equal"] = all(configs["validation"][k] == configs["test"][k] for k in ("jev_url", "bailian_url"))
    checks["final_summary_recalculation_errors"] = []
    test = all_predictions["test"]
    data = {r["unique_id"]: r for r in subsets["test"]}
    gold = [r["gold"] for r in subsets["test"]]
    with (root / "code/final_eval_results/summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            pred = []
            for s in data:
                jev = test[JEV][s]
                use_llm = row["method"] == "llm_only" or (row["method"] == "cascade" and jev["confidence"] <= CUTOFF)
                selected = test[row["llm_model"]][s] if use_llm else jev
                if row["method"] == "cascade" and selected["status"] != "ok":
                    selected = jev
                pred.append(selected["prediction"])
            for field, value in score(gold, pred).items():
                if abs(float(row[field]) - value) > 1e-12:
                    checks["final_summary_recalculation_errors"].append([row["method"], row["llm_model"], field])

    validation = all_predictions["validation"]
    jev = validation[JEV]
    checks["threshold_sweep_recalculation_errors"] = []
    with (root / "code/threshold_results/threshold_sweep.csv").open() as handle:
        sweep = list(csv.DictReader(handle))
    for index, point in enumerate(sweep):
        threshold = float(point["confidence_threshold"]) if point["confidence_threshold"] else -1
        routed = [s for s in jev if jev[s]["confidence"] <= threshold]
        if len(routed) != int(point["routed"]):
            checks["threshold_sweep_recalculation_errors"].append([index, "routed"])
        for model in LLMS:
            value = (sum(r["correct"] for r in jev.values()) + paired(jev, validation[model], routed)["net"]) / len(jev)
            if abs(value - float(point[f"{model}_accuracy"])) > 1e-12:
                checks["threshold_sweep_recalculation_errors"].append([index, model])
    report["threshold_search_points_including_no_route"] = len(sweep)
    report["qwen_best_validation_points"] = [p for p in sweep if float(p[f"{LLMS[0]}_accuracy"]) == max(float(x[f"{LLMS[0]}_accuracy"]) for x in sweep)]
    report["qwen_test_uncertainty"] = {
        method: {unit: cluster_interval(data, test[JEV], test[LLMS[0]], method == "cascade", unit, args.bootstrap_repetitions)
                 for unit in ("case_id", "page")}
        for method in ("llm_only", "cascade")
    }
    report["uncertainty_note"] = "Paired cluster bootstrap, seed 20261109; conditional on this test sample. Does not correct validation threshold selection, label noise, or domain shift."
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "subgroups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subgroup_rows[0]))
        writer.writeheader()
        writer.writerows(subgroup_rows)
    (out / "qwen_routed_disagreements.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in disagreements), encoding="utf-8")
    if args.plot:
        write_plot(report, subgroup_rows, out)
    print(json.dumps({"output": str(out), "checks": checks, "qwen_test_uncertainty": report["qwen_test_uncertainty"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
