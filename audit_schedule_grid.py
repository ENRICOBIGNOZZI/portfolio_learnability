"""Audit the frozen C0 choice and extend diagnostic curves without OOS selection."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from data_pipeline import digest,write_json
from complexity_schedule import calibrate,window,plot_results,cache_managed
from portfolio import annual_splits,sharpe


def audit(clean='data/clean',cache='results/schedule_cache',out='paper/complexity_schedule'):
    out=Path(out)
    old=json.loads((out/'manifest.json').read_text())
    for name in ['paths.csv','initial_validation.csv','annual_schedule.csv','selected_monthly.csv']:
        assert digest(out/name)==old['outputs'][name],name
    g,dates,source=cache_managed(clean,cache)
    assert source['managed_matrix_sha256']==old['source']['managed_matrix_sha256']
    cal=calibrate(g,dates)
    chosen=float(cal['c0'][cal['choice']])
    np.testing.assert_allclose(chosen,old['C0'],rtol=1e-14)
    # A much wider, dense validation-only check of the frozen 200-point choice.
    h=g[:120];values,vectors=np.linalg.eigh(h@h.T)
    projection=vectors.T@np.ones(120);cross=(g[120:180]@h.T)@vectors
    exponent=cal['spectrum']['b']/(cal['spectrum']['b']+1)
    def loss(log_c0):
        penalty=np.exp(log_c0)*120**(-exponent)
        payoff=cross@(projection/(np.maximum(values,0)+120*penalty))
        return float(np.mean((1-payoff)**2))
    low,high=float(cal['c0'].min()/1000),float(cal['c0'].max()*1000)
    x=np.linspace(np.log(low),np.log(high),3001)
    l=np.array([loss(v) for v in x]);best=int(l.argmin())
    assert 0<best<len(x)-1,'Validation optimum at extended audit boundary'
    fit=minimize_scalar(loss,bounds=(x[best-1],x[best+1]),method='bounded',options={'xatol':1e-12})
    # Diagnostic grid only: include every original candidate and frozen selection.
    candidates=np.unique(np.r_[cal['c0'],np.geomspace(low,high,1001)])
    diagnostic=dict(cal,c0=candidates,choice=int(np.flatnonzero(candidates==chosen)[0]))
    original=pd.read_csv(out/'paths.csv');selected=pd.read_csv(out/'annual_schedule.csv')
    records,checks=[],[]
    for split in annual_splits(dates):
        r=window(g,dates,split,diagnostic)
        year=r['year'];d=original[original.year==year].sort_values('complexity')
        baseline=selected[selected.year==year].iloc[0];j=diagnostic['choice']
        for key,new in [('lambda',r['lambda'][j]),('complexity',r['complexity'][j]),
                        ('oos_loss',r['loss'][j]),('oos_sharpe',r['sharpe'][j])]:
            np.testing.assert_allclose(new,baseline[key],rtol=1e-10,atol=1e-10)
        inside=(r['complexity']>=d.complexity.min())&(r['complexity']<=d.complexity.max())
        errors={}
        for label,key in [('loss','oos_loss'),('sharpe','oos_sharpe')]:
            linear=np.interp(np.log(r['complexity'][inside]),np.log(d.complexity),d[key])
            errors[label+'_max_plot_interpolation_error']=float(np.max(np.abs(linear-r[label][inside])))
        mid_c0=np.sqrt(candidates[:-1]*candidates[1:])
        mid=window(g,dates,split,dict(cal,c0=mid_c0,choice=0))
        for label in ['loss','sharpe']:
            interpolated=np.interp(np.log(mid['complexity']),np.log(r['complexity'][::-1]),r[label][::-1])
            errors['dense_'+label+'_max_midpoint_error']=float(np.max(np.abs(interpolated-mid[label])))
        hist=g[np.r_[split[1],split[2]]];test=g[split[3]]
        val,vec=np.linalg.eigh(hist@hist.T)
        assert val.min()>val.max()*1e-12,'Unregularized limit numerically unresolved'
        testcross=test@hist.T
        zero=testcross@(vec@((vec.T@np.ones(len(hist)))/val))
        strong=testcross@np.ones(len(hist))
        qzero=float(np.mean((1-zero)**2));szero=float(sharpe(zero));sstrong=float(sharpe(strong))
        qmin=int(np.argmin(r['loss']));smax=int(np.argmax(r['sharpe']))
        checks.append({'year':year,'T':r['T'],'loss_min_at_endpoint':qmin in (0,len(candidates)-1),
            'sharpe_max_at_endpoint':smax in (0,len(candidates)-1),
            'loss_argmin_C0':float(candidates[qmin]),'sharpe_argmax_C0':float(candidates[smax]),
            'min_complexity_over_T':float(r['complexity'].min()/r['T']),
            'max_complexity_over_T':float(r['complexity'].max()/r['T']),
            'zero_lambda_loss':qzero,'zero_lambda_sharpe':szero,'infinite_lambda_sharpe_limit':sstrong,
            'weak_endpoint_loss_gap':float(abs(r['loss'][0]-qzero)),
            'weak_endpoint_sharpe_gap':float(abs(r['sharpe'][0]-szero)),
            'strong_endpoint_loss_gap_to_one':float(abs(r['loss'][-1]-1)),
            'strong_endpoint_sharpe_gap':float(abs(r['sharpe'][-1]-sstrong)),**errors})
        for i,c0 in enumerate(candidates):
            records.append({'year':year,'candidate':i,'C0':c0,'lambda':r['lambda'][i],
                'complexity':r['complexity'][i],'oos_loss':r['loss'][i],
                'oos_sharpe':r['sharpe'][i],'selected':i==j})
        print('Diagnostic year',year,flush=True)
    paths=pd.DataFrame(records);check=pd.DataFrame(checks)
    paths.to_csv(out/'diagnostic_paths.csv',index=False)
    check.to_csv(out/'grid_endpoint_checks.csv',index=False)
    initial_loss=float(cal['loss'][cal['choice']])
    info={'status':'complete','initial_selection_unchanged':True,'selected_C0':chosen,
        'initial_candidates':200,'validation_audit_candidates':3001,
        'validation_refined_C0':float(np.exp(fit.x)),'original_validation_loss':initial_loss,
        'refined_validation_loss':float(fit.fun),'relative_validation_loss_gap':float((initial_loss-fit.fun)/fit.fun),
        'diagnostic_candidates':len(candidates),'original_C0_bounds':[float(cal['c0'].min()),float(cal['c0'].max())],
        'diagnostic_C0_bounds':[low,high],'range_extension_factor_each_end':1000,
        'selected_policy_check':'All 47 selected lambda, complexity, OOS loss and Sharpe match the frozen 200-candidate policy.',
        'maximum_old_plot_interpolation_loss_error':float(check.loss_max_plot_interpolation_error.max()),
        'maximum_old_plot_interpolation_sharpe_error':float(check.sharpe_max_plot_interpolation_error.max()),
        'midpoint_checks':47*(len(candidates)-1),
        'maximum_dense_loss_midpoint_error':float(check.dense_loss_max_midpoint_error.max()),
        'maximum_dense_sharpe_midpoint_error':float(check.dense_sharpe_max_midpoint_error.max()),
        'interpretation':'Endpoint extrema may represent genuine weak/strong regularization limits, not finite interior optima. OOS extrema never select C0. Both analytical limiting policies are evaluated for every year.',
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'code_sha256':digest(Path(__file__))}
    write_json(out/'grid_audit.json',info)
    plot_results(paths,selected,out)
    old['figure_path_input']='diagnostic_paths.csv'
    old['figure_grid_audit']='grid_audit.json'
    old['render_git_sha']=info['git_sha']
    old['render_code_checksum']=digest(Path('complexity_schedule.py'))
    old['outputs']={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    old.pop('visual_qa',None)
    write_json(out/'manifest.json',old)
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':
    audit()
