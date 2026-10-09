"""Annual cold refits, monthly decisions, immutable decision/payoff ledgers and resume."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .accounting import turnover, net_excess
from .calendar import ages, next_month
from .config import PACKAGE, REPO, canonical_hash, digest, provenance, write_json
from .data_adapter import PanelStore
from .guard import mature_records, select_guard, soft_weights
from .memory import grid, diagnostics, matched_horizon, taper
from .networks import Policy
from .portfolio import holdings, payoff, ensemble
from .training import fit


def architecture_key(a):
    return f'L{a["depth"]}W{a["width"] if a["depth"] else 0}'


def previous_positions(path):
    """Decompress each previous-month array once, not once per method."""
    with np.load(path,allow_pickle=False) as saved:
        return {key:saved[key] for key in ['ids','method_names','weights','total_returns']}


def fit_bank(store, origin, c, directory, source, monitor=None):
    history = store.history(origin, c['history_start'])
    age, span = ages(origin, [next_month(d) for d in history.dates], next_month(c['history_start']))
    bank, metadata, excluded, weight_log = {}, {}, [], []
    canonical = []
    for memory in grid(c['memories']):
        try:
            omega = memory.weights(age, span)
            stats = diagnostics(omega, age)
            if stats['n_eff'] < c['minimum_n_eff'] - 1e-9:
                raise ValueError('N_EFF_BELOW_MINIMUM')
        except ValueError as error:
            excluded.append(dict(memory=memory.key,reason=str(error)))
            continue
        match = None
        if memory.family == 'theory':
            same, horizon = matched_horizon(age, span, memory.value, memory.gamma)
            error = float(np.max(abs(same-taper(age,horizon,memory.gamma))))
            if error > 1e-10:
                raise AssertionError('QP/taper mismatch.')
            match = dict(horizon=horizon,maximum_weight_error=error,exceeds_history=horizon>span)
        alias = next((key for key, other in canonical if np.allclose(omega,other,rtol=0,atol=1e-14)), None)
        if alias is None:
            canonical.append((memory.key,omega))
        weight_log.append(dict(memory=memory.key, **stats, match=match, alias=alias,
                              ages=age.tolist(), omega=omega.tolist()))
        for a in c['architectures']:
            akey = architecture_key(a)
            key = akey+'|'+memory.key
            metadata[key] = dict(architecture=akey, family=memory.family, memory=memory.key, **stats)
            if alias is not None:
                bank[key] = bank[akey+'|'+alias]
                continue
            models = []
            for seed in c['seeds']:
                spec = dict(origin=str(origin.date()), training_cutoff=str(origin.date()),
                    history_start=c['history_start'],source=store.fingerprint,features=store.feature_hash,
                    architecture=a, memory=memory.__dict__, omega=canonical_hash(omega.tolist()),
                    seed=seed,optimizer=c['optimizer'],code_hash=source['code_hash'],sample_mode=store.mode)
                token = canonical_hash(spec)
                checkpoint = directory/'fits'/f'{token}.pt'
                log = directory/'fits'/f'{token}.json'
                if log.exists() and checkpoint.exists():
                    info = json.loads(log.read_text())
                    if info['checkpoint_hash'] != digest(checkpoint):
                        raise ValueError('Checkpoint checksum mismatch.')
                    model = Policy(len(store.names),a['depth'],a['width'],seed)
                    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
                else:
                    model, info = fit(history,omega,a,seed,c['optimizer'],origin,monitor)
                    checkpoint.parent.mkdir(parents=True,exist_ok=True)
                    temp = checkpoint.with_suffix('.tmp')
                    torch.save(model.state_dict(),temp)
                    temp.replace(checkpoint)
                    write_json(log,dict(**info,spec=spec,checkpoint_hash=digest(checkpoint)))
                    print(f'fit {origin:%Y-%m} {key} seed={seed} seconds={info["seconds"]:.2f}',flush=True)
                models.append(model)
            bank[key] = models
    write_json(directory/'origins'/f'{origin:%Y-%m}.json',dict(origin=str(origin.date()),
        training_cutoff=str(origin.date()),training_months=len(history),calendar_span=span,
        candidate_count=len(bank),excluded=excluded,weights=weight_log))
    return bank, metadata


def choose_methods(keys, metadata, past, c):
    """Choose using only mature records. Store actual baseline history, never rewrite it."""
    selections, gates = {}, {}
    akeys = list(dict.fromkeys(architecture_key(a) for a in c['architectures']))
    window = c['guard']['window']
    families = [x for x in c['memories'] if x != 'uniform']
    for scope in akeys+['joint']:
        available = [k for k in keys if scope == 'joint' or metadata[k]['architecture'] == scope]
        uniforms = [k for k in available if metadata[k]['family'] == 'uniform']
        if not uniforms:
            raise ValueError('Every architecture needs a full-history baseline.')
        baseline_method = scope+'/full'
        fixed = c['baseline_architecture']+'|uniform:0:1'
        baseline = fixed if scope == 'joint' and fixed in uniforms else uniforms[0]
        baseline_recent = [r for r in past if all(k in r['experts'] for k in uniforms)][-window:]
        if len(baseline_recent) >= window:
            baseline = min(uniforms, key=lambda k: (np.mean([(1-r['experts'][k])**2 for r in baseline_recent]),k))
        selections[baseline_method] = {baseline:1.}
        for family in families:
            candidates = [k for k in available if metadata[k]['family'] == family]
            if not candidates:
                continue
            recent = [r for r in past if baseline_method in r['methods']
                      and all(k in r['experts'] for k in candidates)][-window:]
            label = scope+'/'+family
            # Unguarded uses the same causal recent response-one ranking, fixed first tie rule.
            selected = min(candidates, key=lambda k:(np.mean([(1-r['experts'][k])**2 for r in recent]),k)) if recent else candidates[0]
            selections[label+'/unguarded'] = {selected:1.}
            baselines = np.array([r['methods'][baseline_method]['raw_excess'] for r in recent])
            candidate_returns = np.array([[r['experts'][k] for k in candidates] for r in recent]).reshape(len(recent),len(candidates))
            gate = select_guard(baselines,candidate_returns,c['guard'])
            winner = candidates[gate['index']] if gate['active'] else baseline
            selections[label+'/guarded'] = {winner:1.}
            gates[label] = {**gate, 'selected':winner,'baseline':baseline,'candidate_keys':candidates}
            losses = (1-candidate_returns)**2
            probabilities = soft_weights(losses,c['soft_temperature'])
            selections[label+'/soft'] = dict(zip(candidates,probabilities.tolist()))
    return selections, gates


def run(c, directory, resume=False, monitor=None, stop_after=None, store=None):
    directory = Path(directory)
    source = provenance(c)
    store = store or PanelStore(c['sample_mode'])
    source.update(data_fingerprint=store.fingerprint,feature_manifest_hash=store.feature_hash,
                  features=store.names,sample_mode=c['sample_mode'])
    signature = canonical_hash({k:source[k] for k in ['config_hash','code_hash','data_fingerprint']})
    source['signature'] = signature
    existing = directory/'source.json'
    if existing.exists():
        if not resume:
            raise FileExistsError('Use a new private run directory or --resume.')
        if json.loads(existing.read_text())['signature'] != signature:
            raise ValueError('Resume configuration/code/data signature mismatch.')
    else:
        write_json(existing,source)
    write_json(directory/'calendar_manifest.json',dict(
        formation_start=c['formation_start'],formation_end=c['formation_end'],
        annual_refit='January formation close; previous December forward payoff available',
        target='next calendar month', common_payoffs=['1994-01-31','2024-12-31'],
        terminal_payoffs=['2020-01-31','2024-12-31'], terminal_status='retrospective, previously studied',
        historical_vintage_certified=False))
    rf_meta = json.loads((REPO/'data/risk_free.json').read_text())
    if digest(REPO/'data/risk_free.csv') != rf_meta['sha256']:
        raise ValueError('Frozen cash file mismatch.')
    if rf_meta['raw_manifest_sha256'] != digest(REPO/'data/raw/manifest.json'):
        raise ValueError('Cash source belongs to a different snapshot.')
    cash = pd.read_csv(REPO/'data/risk_free.csv').set_index('return_date').rf.to_dict()
    events = []
    for file in sorted((directory/'events').glob('*.json')):
        event = json.loads(file.read_text())
        previous = events[-1]['event_hash'] if events else None
        if event['previous_hash'] != previous:
            raise ValueError('Broken append-only event chain.')
        check = dict(event)
        token = check.pop('event_hash')
        if canonical_hash(check) != token:
            raise ValueError('Modified historical event.')
        holdings_path = directory/'holdings'/f'{pd.Timestamp(event["decision_date"]):%Y-%m}.npz'
        if digest(holdings_path) != event['holdings_hash']:
            raise ValueError('Modified saved holdings.')
        events.append(event)
    dates = pd.date_range(c['formation_start'],c['formation_end'],freq='ME')
    if events and [e['decision_date'] for e in events] != [str(d.date()) for d in dates[:len(events)]]:
        raise ValueError('Resume ledger is not a contiguous calendar prefix.')
    previous_holdings = None
    if events:
        previous_holdings = previous_positions(directory/'holdings'/f'{pd.Timestamp(events[-1]["decision_date"]):%Y-%m}.npz')
    bank, meta, origin = None, None, None
    completed_this_call = 0
    for date in dates[len(events):]:
        if monitor:
            monitor.check()
        fit_origin = pd.Timestamp(year=date.year,month=1,day=31)
        if bank is None or fit_origin != origin:
            bank,meta = fit_bank(store,fit_origin,c,directory,source,monitor)
            origin = fit_origin
        panel = store.panel(date)
        key_order = sorted(bank)
        raw_weights, policy_scores, seed_weights = {}, {}, {}
        for key in key_order:
            parts = [holdings(model,panel.Z,c['optimizer']['stock_block']) for model in bank[key]]
            policy_scores[key] = ensemble([p[0] for p in parts])
            seed_weights[key] = [p[1] for p in parts]
            raw_weights[key] = ensemble(seed_weights[key])
        past = mature_records(events,date)
        selections,gates = choose_methods(key_order,meta,past,c)
        financial = {name:sum(value*raw_weights[key] for key,value in allocation.items())
                     for name,allocation in selections.items()}
        decision = dict(decision_date=str(date.date()), return_date=str(next_month(date).date()),
                        fit_origin=str(origin.date()),selections=selections,gates=gates,
                        sample_mode=c['sample_mode'],signature=signature, scale=1.)
        decision_path = directory/'decisions'/f'{date:%Y-%m}.json'
        if decision_path.exists():
            if json.loads(decision_path.read_text()) != decision:
                raise ValueError('Attempt to rewrite an existing decision.')
        else:
            write_json(decision_path,decision)
        # Future payoffs are consumed only after saving the decision.
        r = panel.forward_excess_returns
        expert_payoffs = {k:payoff(w,r) for k,w in raw_weights.items()}
        separate = {k:[payoff(w,r) for w in seed_weights[k]] for k in key_order}
        return_date = str(next_month(date).date())
        rf = cash[return_date]
        method_rows = {}
        names = sorted(financial)
        for name in names:
            w = financial[name]
            old_ids = old_w = old_returns = old_port = None
            if previous_holdings is not None and name in previous_holdings['method_names']:
                index = list(previous_holdings['method_names']).index(name)
                old_ids = previous_holdings['ids']
                old_w = previous_holdings['weights'][index]
                old_returns = previous_holdings['total_returns']
                old_port = events[-1]['methods'][name]['raw_excess'] + events[-1]['risk_free']
            target_turn,_ = turnover(panel.asset_ids,w,old_ids,old_w)
            actual_turn,kind = turnover(panel.asset_ids,w,old_ids,old_w,old_returns,old_port)
            p = payoff(w,r)
            short = float(-w[w<0].sum())
            gross = float(abs(w).sum())
            method_rows[name] = dict(raw_excess=p,response_one=(1-p)**2,gross=gross,net=float(w.sum()),
                cash=1-float(w.sum()),short=short,concentration=float((w*w).sum()/gross**2) if gross else 0,
                maximum_absolute_weight=float(abs(w).max()),weight_quantiles=np.quantile(w,[0,.01,.5,.99,1]).tolist(),
                turnover=actual_turn,turnover_kind=kind,target_weight_turnover=target_turn,
                cost_sensitivities={f'{cost}bps':net_excess(p,actual_turn,short,cost,0) for cost in c['cost_bps']},
                borrowing_25bps_plus_30bps_year=net_excess(p,actual_turn,short,25,30))
        arrays = dict(ids=panel.asset_ids,method_names=np.array(names),weights=np.stack([financial[n] for n in names]),
            expert_names=np.array(key_order),raw_weights=np.stack([raw_weights[k] for k in key_order]),
            policy_scores=np.stack([policy_scores[k] for k in key_order]),
            seed_weights=np.stack([seed_weights[k] for k in key_order]),reporting_scale=np.array(1.),
            total_returns=np.asarray(r)+rf)
        hfile = directory/'holdings'/f'{date:%Y-%m}.npz'
        hfile.parent.mkdir(parents=True,exist_ok=True)
        with hfile.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f,**arrays)
        hfile.with_suffix('.tmp').replace(hfile)
        event = dict(**decision,available_at=return_date,experts=expert_payoffs,seed_payoffs=separate,
            methods=method_rows,risk_free=float(rf),holdings_hash=digest(hfile),
            previous_hash=events[-1]['event_hash'] if events else None)
        event['event_hash'] = canonical_hash(event)
        event_file = directory/'events'/f'{date:%Y-%m}.json'
        if event_file.exists():
            raise ValueError('Append-only ledger collision.')
        write_json(event_file,event)
        events.append(event)
        previous_holdings = previous_positions(hfile)
        completed_this_call += 1
        print(f'decided {date:%Y-%m}; realized {return_date}; experts={len(bank)} months={len(events)}',flush=True)
        if stop_after is not None and completed_this_call >= stop_after:
            return dict(status='INTERRUPTED_FOR_RESUME_TEST',months=len(events))
    manifest = dict(status='REAL_RUN_COMPLETED',interpretation='RETROSPECTIVE_ONLY' if store.mode == 'retrospective_complete_payoff' else 'FORMATION_AUDIT',
        signature=signature,months=len(events),fits=len(list((directory/'fits').glob('*.json'))),
        origins=len(list((directory/'origins').glob('*.json'))),
        candidates_per_month=sorted(set(len(e['experts']) for e in events)),
        first_payoff=events[0]['return_date'],last_payoff=events[-1]['return_date'],
        final_event_hash=events[-1]['event_hash'], completed_at=time.time())
    write_json(directory/'manifest.json',manifest)
    return manifest
