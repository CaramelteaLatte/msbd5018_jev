#!/usr/bin/env python3
"""Independently recompute formal H/N scores and routing from local API checkpoints."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import sqlite3

import experiment_v2 as shared
import ragtruth_experiment as ex

ROOT = Path(__file__).resolve().parents[1]


def close(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for k in a: close(a[k], b[k])
    elif isinstance(a, float):
        assert math.isclose(a, b, abs_tol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def independent_score(samples, outputs, gold='gold'):
    confusion = {g: {p: 0 for p in ('H', 'N', 'INVALID')} for g in ('H', 'N')}
    for s, p in zip(samples, outputs):
        confusion[s[gold]][p if p is not None else 'INVALID'] += 1
    per_label = {}
    for label in ('H', 'N'):
        support = sum(confusion[label].values())
        predicted = sum(confusion[g][label] for g in ('H', 'N'))
        tp = confusion[label][label]
        per_label[label] = dict(support=support, true_positive=tp, predicted=predicted,
            precision=tp/predicted if predicted else 0., recall=tp/support if support else 0.,
            f1=2*tp/(support+predicted) if support+predicted else 0.)
    correct = sum(confusion[g][g] for g in ('H', 'N'))
    return dict(samples=len(samples), correct=correct, accuracy=correct/len(samples),
                macro_f1=sum(r['f1'] for r in per_label.values())/2,
                hallucination_precision=per_label['H']['precision'], hallucination_recall=per_label['H']['recall'],
                hallucination_f1=per_label['H']['f1'], invalid_predictions=outputs.count(None),
                false_positive=confusion['N']['H'], false_negative=confusion['H']['N']+confusion['H']['INVALID'],
                per_label=per_label, confusion=confusion)


def read_cache(out, stage, samples):
    with sqlite3.connect(f'file:{out / (stage+"_checkpoint.sqlite3")}?mode=ro', uri=True) as db:
        config = json.loads(db.execute('SELECT manifest FROM config').fetchone()[0])
        records = {(m,sid):json.loads(body) for m,sid,body in db.execute('SELECT model,sample_id,body FROM attempts WHERE terminal=1')}
        attempts = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    expected = {(m,s['sample_id']) for m in shared.MODELS for s in samples}
    assert set(records) == expected, 'Incomplete/unexpected checkpoint outcomes'
    close(config, json.loads((out/'run_config.json').read_text()))
    assert config['protocol_sha256'] == shared.digest(ex.PROTOCOL_PATH)
    for s in samples:
        for m in shared.MODELS:
            r = records[m,s['sample_id']]
            assert r['gold'] == s['gold']
            if r['status'] == 'ok':
                assert r['prediction'] in ('H','N')
                assert r['correct'] == (r['prediction'] == s['gold'])
                assert r['model_returned'] == m, (m,r['model_returned'])
            elif m == shared.JEV:
                raise AssertionError('Invalid JEV result')
    progress = json.loads((out/'progress.json').read_text())
    assert progress['complete'] and all(n == len(samples) for n in progress['terminal_by_model'].values())
    assert progress['nonterminal_attempts'] == attempts-len(expected)
    accounting=json.loads((out/'attempt_summary.json').read_text())
    assert accounting['total_attempts']==attempts and accounting['terminal_outcomes']==len(records)
    assert accounting['nonterminal_attempts']==attempts-len(records)
    assert sum(r['attempts'] for r in accounting['nonterminal_errors'])==attempts-len(records)
    invalid={(m,sid) for (m,sid),r in records.items() if r['status']!='ok'}
    assert {(r['model'],r['sample_id']) for r in accounting['invalid_terminal']}==invalid
    return records, config, attempts


def predict(records, s, model):
    r = records[model,s['sample_id']]
    return r['prediction'] if r['status'] == 'ok' else None


def routed(records, s, cutoff):
    return cutoff is not None and records[shared.JEV,s['sample_id']]['confidence'] <= cutoff


def cascade(records, samples, model, cutoff):
    return [(predict(records,s,model) if routed(records,s,cutoff) and predict(records,s,model) is not None
             else predict(records,s,shared.JEV)) for s in samples]


def best_threshold(records, samples, model):
    candidates = [None]+sorted({records[shared.JEV,s['sample_id']]['confidence'] for s in samples})
    feasible = []
    for t in candidates:
        count = sum(routed(records,s,t) for s in samples)
        if count > len(samples)*20//100:
            continue
        stats = independent_score(samples,cascade(records,samples,model,t))
        h = stats['per_label']['H']
        f1 = Fraction(2*h['true_positive'], h['support']+h['predicted']) if h['support']+h['predicted'] else Fraction(0)
        feasible.append(((f1,stats['correct'],-count), t, count, stats))
    return max(feasible,key=lambda r:r[0]), feasible


def verify_bootstrap(samples, baseline, candidate, saved):
    import numpy as np
    groups = sorted({s['source_id'] for s in samples})
    columns = []
    for g in groups:
        idx = [i for i,s in enumerate(samples) if s['source_id'] == g]
        counts = [len(idx),sum(samples[i]['gold'] == 'H' for i in idx)]
        for pred in (baseline,candidate):
            counts.extend([sum(pred[i] == samples[i]['gold'] for i in idx),
                           sum(pred[i] == samples[i]['gold'] == 'H' for i in idx),sum(pred[i] == 'H' for i in idx)])
        columns.append(counts)
    columns = np.array(columns)
    draws = np.random.default_rng(ex.SEED).integers(0,len(groups),size=(10000,len(groups)))
    # Independent group multiplicity aggregation; same fixed random draws as the protocol.
    weights = np.array([np.bincount(d,minlength=len(groups)) for d in draws])
    total = weights@columns
    def f1(tp,den):
        return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den != 0)
    differences = ((total[:,5]-total[:,2])/total[:,0],
                   f1(total[:,6],total[:,1]+total[:,7])-f1(total[:,3],total[:,1]+total[:,4]))
    b,p = independent_score(samples,baseline),independent_score(samples,candidate)
    for i,metric in enumerate(('accuracy','hallucination_f1')):
        lo,hi=np.quantile(differences[i],[.025,.975])
        close(saved[metric],dict(difference=p[metric]-b[metric],ci95_low=float(lo),ci95_high=float(hi)))
    assert saved['source_groups']==len(groups) and saved['repetitions']==10000 and saved['seed']==ex.SEED


def verify_stage(stage, data, out, thresholds):
    samples,report,protocol = ex.load_data(data,stage)
    records,config,attempts = read_cache(out,stage,samples)
    assert config['data_sha256'] == shared.digest(data)
    assert config['split_report_sha256'] == shared.digest(data.parent/'split_report.json')
    assert config['request_configs'] == protocol['request_configs']
    artifact = ex.load_thresholds(thresholds,protocol)
    assert config['thresholds_sha256'] == (shared.digest(thresholds) if stage=='test' else None)
    metrics = json.loads((out/'metrics.json').read_text())
    summaries = {r['scenario']:r for r in csv.DictReader((out/'summary.csv').open())}
    sensitivity = {r['scenario']:r for r in csv.DictReader((out/'sensitivity_without_implicit_true.csv').open())}
    subgroups = {(r['scenario'],r['generator_model']):r for r in csv.DictReader((out/'by_generator.csv').open())}
    exported = {r['sample_id']:r for r in map(json.loads,(out/'predictions.jsonl').read_text().splitlines())}
    assert set(exported)=={s['sample_id'] for s in samples}
    intervals = json.loads((out/'bootstrap.json').read_text()) if stage=='test' else None
    baseline = [predict(records,s,shared.JEV) for s in samples]
    scenarios=[('jev_only',shared.JEV,False)]+[(f'llm_only/{a}',m,False) for a,m in shared.LLMS.items()]+[(f'cascade/{a}',m,True) for a,m in shared.LLMS.items()]
    for name,model,isc in scenarios:
        t = artifact['models'][model]['confidence_threshold'] if isc else None
        outputs = cascade(records,samples,model,t) if isc else [predict(records,s,model) for s in samples]
        stats=independent_score(samples,outputs)
        close(metrics[name],stats)
        for k,v in stats.items():
            if not isinstance(v,dict): close(float(summaries[name][k]),float(v))
        alternate=independent_score(samples,outputs,'gold_without_implicit_true')
        for k,v in alternate.items():
            if not isinstance(v,dict): close(float(sensitivity[name][k]),float(v))
        for generator in {s['generator_model'] for s in samples}:
            ix=[i for i,s in enumerate(samples) if s['generator_model']==generator]
            grouped=independent_score([samples[i] for i in ix],[outputs[i] for i in ix])
            for k,v in grouped.items():
                if not isinstance(v,dict): close(float(subgroups[name,generator][k]),float(v))
        for s,p in zip(samples,outputs):
            e=exported[s['sample_id']]
            assert e['predictions'][name]==p and e['gold']==s['gold'] and e['source_id']==s['source_id']
        ju,lu,delays=[],[],[]
        for s in samples:
            jr,lr=records[shared.JEV,s['sample_id']],records[model,s['sample_id']]
            usesj=isc or model==shared.JEV
            usesl=model!=shared.JEV and (not isc or routed(records,s,t))
            if usesj: ju.append(jr)
            if usesl: lu.append(lr)
            delays.append((jr['latency_ms'] if usesj else 0)+(lr['latency_ms'] if usesl else 0))
        row=summaries[name]
        assert int(row['jev_calls'])==len(ju) and int(row['llm_calls'])==len(lu)
        import numpy as np
        close(float(row['mean_latency_ms']),float(np.mean(delays)))
        close(float(row['p95_latency_ms']),float(np.percentile(delays,95)))
        for prefix,used in (('jev',ju),('llm',lu)):
            for field in ('input_tokens','output_tokens'):
                observed=[r[field] for r in used if type(r.get(field)) is int and r[field]>=0]
                assert int(row[prefix+'_'+field+'_observed'])==sum(observed)
                assert int(row[prefix+'_calls_with_'+field])==len(observed)
        if not lu:
            costs=(0.,0.)
        elif all(type(r.get(f)) is int and r[f]>=0 for r in lu for f in ('input_tokens','output_tokens')):
            costs=tuple((sum(r['input_tokens'] for r in lu)*a+sum(r['output_tokens'] for r in lu)*b)/1_000_000
                        for a,b in shared.pilot.RATES[model])
        else:
            costs=(None,None)
        for bound,cost in zip(('low','high'),costs):
            saved=row['llm_estimated_cost_usd_'+bound]
            if cost is None: assert saved==''
            else: close(float(saved),cost)
            if not ju: assert row['estimated_total_cost_usd_'+bound]==saved
        if ju: assert row['estimated_total_cost_usd_low']==row['estimated_total_cost_usd_high']==''
        if isc:
            corrected=sum(b!=s['gold'] and p==s['gold'] for s,b,p in zip(samples,baseline,outputs))
            harmed=sum(b==s['gold'] and p!=s['gold'] for s,b,p in zip(samples,baseline,outputs))
            assert int(row['corrected'])==corrected and int(row['harmed'])==harmed
            assert int(row['routed'])==sum(routed(records,s,t) for s in samples)
            assert int(row['fallbacks'])==sum(routed(records,s,t) and predict(records,s,model) is None for s in samples)
        if intervals is not None:
            verify_bootstrap(samples,baseline,outputs,intervals[name])
    if stage=='validation':
        sweep_rows=list(csv.DictReader((out/'threshold_sweep.csv').open()))
        for model in shared.LLMS.values():
            best,feasible=best_threshold(records,samples,model)
            choice=artifact['models'][model]
            assert choice['confidence_threshold']==best[1] and choice['routed']==best[2]
            saved=[r for r in sweep_rows if r['model']==model]
            assert len(saved)==len(feasible)
            for row,(_,t,n,stats) in zip(saved,feasible):
                assert (None if row['confidence_threshold']=='' else float(row['confidence_threshold']))==t
                assert int(row['routed'])==n
                close(float(row['accuracy']),stats['accuracy'])
                close(float(row['hallucination_f1']),stats['hallucination_f1'])
        folds=json.loads((out/'folds.json').read_text())['source_id_to_fold']
        assert set(folds)=={s['source_id'] for s in samples} and set(folds.values())==set(range(5))
        assert folds==ex.grouped_folds(samples)
        cv=list(csv.DictReader((out/'cross_validation.csv').open()))
        oof_saved=json.loads((out/'oof_metrics.json').read_text())
        for model in shared.LLMS.values():
            oof={}
            for fold in range(5):
                train=[s for s in samples if folds[s['source_id']]!=fold]
                held=[s for s in samples if folds[s['source_id']]==fold]
                best,_=best_threshold(records,train,model)
                row=next(r for r in cv if r['model']==model and int(r['fold'])==fold+1)
                assert (None if row['confidence_threshold']=='' else float(row['confidence_threshold']))==best[1]
                pred=cascade(records,held,model,best[1])
                stats=independent_score(held,pred)
                close(float(row['hallucination_f1']),stats['hallucination_f1'])
                close(float(row['accuracy']),stats['accuracy'])
                oof.update((s['sample_id'],p) for s,p in zip(held,pred))
            close(oof_saved[model],independent_score(samples,[oof[s['sample_id']] for s in samples]))
    print(f'PASS {stage}: {len(samples)} responses, {len(records)} terminal outcomes, {attempts} attempts; independent scores, routing, sensitivity, tokens/latency'+('; threshold choices and all held-out folds' if stage=='validation' else '; 10,000 paired source bootstrap draws'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('validation','test','all'),default='all')
    args=p.parse_args()
    thresholds=ROOT/'code/ragtruth_validation_results/selected_threshold.json'
    for stage in (('validation','test') if args.stage=='all' else (args.stage,)):
        out=ROOT/('code/ragtruth_validation_results' if stage=='validation' else 'code/ragtruth_final_results')
        verify_stage(stage,ROOT/'data/ragtruth_qa'/f'{stage}.jsonl',out,thresholds)


if __name__=='__main__': main()
