"""Paired rank sensitivity using common economic paths, not a fitted spectrum."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.config.design import design_for


def rank_audit(output):
    output = Path(output)
    design = design_for('paper')
    ranks = {'rank128': 128, 'baseline': 256, 'rank512': 512}
    R = 200
    summary = {}
    for name in ranks:
        arrays = []
        choices = []
        for r in range(R):
            with np.load(output/'data'/name/'replications'/f'{r:04d}.npz') as saved:
                arrays.append(saved['regret'])
                choices.append(np.argmin(saved['validation_loss'], axis=1))
        regret = np.stack(arrays)
        choices = np.stack(choices)
        oracle = np.argmin(regret.mean(axis=0), axis=1)
        summary[name] = {'oracle': regret[:, np.arange(len(design.T)), oracle],
                         'selected': np.take_along_axis(regret, choices[..., None], axis=2)[..., 0]}
    rows = []
    for name in ('rank128', 'rank512'):
        for metric in ('oracle', 'selected'):
            for t, T in enumerate(design.T):
                reference = summary['baseline'][metric][:, t]
                difference = summary[name][metric][:, t]-reference
                relative = float(difference.mean()/reference.mean())
                rows.append({'rank': ranks[name], 'T': T, 'metric': metric,
                             'paired_mean_difference': difference.mean(),
                             'paired_difference_MCSE': difference.std(ddof=1)/np.sqrt(R),
                             'relative_mean_difference': relative,
                             'tolerance': design.approximation_relative_tolerance,
                             'within_tolerance': abs(relative)<=design.approximation_relative_tolerance})
    frame = pd.DataFrame(rows)
    destination = output/'audit'
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination/'approximation_rank.csv', index=False)
    # Increasing rank is the acceptance comparison; the smaller rank is a
    # diagnostic illustrating the direction/size of approximation error.
    passed = bool(frame.loc[frame['rank']==512, 'within_tolerance'].all())
    report = {'passed': passed, 'paired_replications': R, 'headline_rank': 256,
              'increased_rank': 512, 'relative_tolerance': design.approximation_relative_tolerance,
              'criterion': 'All T: oracle and validation mean regret change <=10% when increasing rank 256 to 512.',
              'lower_rank_diagnostic': 128}
    (destination/'approximation_rank.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
