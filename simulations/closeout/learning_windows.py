"""Complete contiguous-window sensitivity, chosen without exponent targeting.

At least five distinct horizons per fit; no observation is removed from the
archived experiment. Population-complexity MCSE under the fixed theoretical
constant is zero conditional on the numerical operator, not an uncertainty
statement about that operator. Whole-history covariance and jackknife retained.
"""
import numpy as np
import pandas as pd
from simulations.closeout.results import DEST,SPECS
from simulations.extended.design import ENVIRONMENTS
from simulations.provenance import json_write,file_hash

LOWERS=(60,90,120,180,240,360,540,720)
UPPERS=(720,1080,1440,2160,3240,4860)
METRICS=('annual_gap','regret','empirical_complexity','population_complexity')


def slope(times,values):
    times=np.asarray(times,float);v=np.asarray(values,float);mean=v.mean(axis=0)
    if np.any(mean<=0):raise ValueError('Log-mean slopes require positive means.')
    x=np.log(times);weights=(x-x.mean())/np.sum((x-x.mean())**2)
    estimate=float(weights@np.log(mean));gradient=weights/mean
    covariance=np.cov(v,rowvar=False,ddof=1)/len(v)
    se=float(np.sqrt(max(gradient@covariance@gradient,0)))
    leave=(v.sum(axis=0)-v)/(len(v)-1);jack=np.log(leave)@weights
    js=float(np.sqrt((len(v)-1)*np.mean((jack-jack.mean())**2)))
    return dict(slope=estimate,MCSE_full_covariance=se,MCSE_jackknife=js,CI_low=estimate-1.96*se,CI_high=estimate+1.96*se)


def run():
    warnings=pd.read_csv(DEST/'numerical_warnings.csv');rows=[];local=[];sources={}
    for name in ENVIRONMENTS:
        path=DEST/'distributions'/f'{name}.npz';sources[str(path)]=file_hash(path)
        with np.load(path) as z:data={k:z[k] for k in z.files}
        times=data['T']
        for j,spec in enumerate(SPECS):
            for low in LOWERS:
                for high in UPPERS:
                    if high>times[-1]:continue
                    keep=(times>=low)&(times<=high)
                    if keep.sum()<5:continue
                    numerical=warnings[(warnings.environment==name)&(warnings.specification==spec)&warnings['T'].isin(times[keep])]
                    for metric in METRICS:
                        result=slope(times[keep],data[metric][:,keep,j])
                        if spec.startswith('oracle'):
                            result={k:v for k,v in result.items() if k=='slope'}
                        rows.append(dict(environment=name,specification=spec,quantity=metric,T_min=int(times[keep][0]),T_max=int(times[keep][-1]),
                            horizons=int(keep.sum()),all_projection_floor_pass=bool(numerical.floor_pass.all()),
                            failed_floor_horizons=';'.join(str(int(t)) for t in numerical.loc[~numerical.floor_pass,'T']),
                            maximum_floor_fraction=float(numerical.floor_fraction.max()),
                            maximum_rank_certified=False,
                            uncertainty='Whole-history Monte Carlo; numerical/operator and oracle-selection uncertainty separate.',**result))
            for metric in METRICS:
                for k in range(len(times)-4):
                    keep=np.arange(k,k+5)
                    result=slope(times[keep],data[metric][:,keep,j])
                    if spec.startswith('oracle'):result={key:value for key,value in result.items() if key=='slope'}
                    local.append(dict(environment=name,specification=spec,quantity=metric,T_min=int(times[k]),T_max=int(times[k+4]),
                                      horizons=5,T_center=float(np.exp(np.log(times[keep]).mean())),**result))
    frame=pd.DataFrame(rows);frame.to_csv(DEST/'contiguous_learning_windows.csv',index=False)
    pd.DataFrame(local).to_csv(DEST/'local_learning_slopes.csv',index=False)
    emphasized=frame[(frame.environment=='baseline')&(frame.specification=='old_fixed')&(frame.T_min==720)&(frame.T_max==4860)]
    emphasized.to_csv(DEST/'secondary_learning_window.csv',index=False)
    json_write(DEST/'learning_window_policy.json',dict(primary='Complete thirteen-horizon grid, theoretical predetermined constant.',
        secondary=[720,4860],selection_reason='An existing, economically interpretable long-history sensitivity window from the earlier audit; illustrates late recovery after the initial 5-to-60-year transient. Complete-grid and neighboring-window results remain next to it. Never selected by minimizing distance to -0.6.',
        minimum_horizons=5,candidate_lower_boundaries=list(LOWERS),candidate_upper_boundaries=list(UPPERS),
        late_window_includes_failed_projection_floor_conditions=True,all_archived_observations_retained=True,
        window_rows=len(frame),sources=sources))

if __name__=='__main__':run()
