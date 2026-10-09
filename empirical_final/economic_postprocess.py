"""Publish only defined cell densities from immutable reconstruction checkpoints.

An empty stock cell has zero payoff but no observed weight density. The raw
checkpoint uses a zero accumulator; this publication step restores missingness
before time averages, without changing any stock holding or attribution.
"""
import json
import numpy as np
import pandas as pd
from data_pipeline import digest
from empirical_final.economic_holdings import OUT, PRIVATE, save


def main():
    paths=[PRIVATE/f'year_{y}/joint_profiles.parquet' for y in range(1978,2025)]
    p=pd.concat([pd.read_parquet(path) for path in paths],ignore_index=True)
    empty=p['count'].eq(0)
    p.loc[empty,['relative_weight_density','nonadditive_density']]=np.nan
    p.to_parquet(OUT/'tables/monthly_joint_profiles.parquet',index=False,compression='zstd')
    save('monthly_joint_profiles',p)
    (OUT/'audit/empty_cell_convention.json').write_text(json.dumps(dict(
        empty_group_cell_months=int(empty.sum()),
        partitions=p.loc[empty,'partition'].value_counts().to_dict(),
        convention='Empty cells: payoff zero; observed density and nonadditive residual missing; additive fitted prediction retained. Time means omit missing observed densities.',
        checkpoint_hashes={str(path.relative_to(PRIVATE)):digest(path) for path in paths}),indent=2))


if __name__=='__main__':main()
