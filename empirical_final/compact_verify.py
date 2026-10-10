"""Independent public-aggregate, source-package and rendered-PDF verification."""
import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from pypdf import PdfReader
from data_pipeline import digest
from portfolio import sharpe
from empirical_final.compact_common import ROOT,OUT,PROTOCOL,audit


def main():
    pub=OUT/'publication'
    manifest=json.loads((OUT/'audit/run_manifest.json').read_text())
    for name,value in manifest['artifact_sha256'].items():
        if digest(OUT/name)!=value:raise ValueError('Artifact changed: '+name)
    curve=pd.read_csv(OUT/'tables/nine_panel_curves.csv')
    ids=pd.read_csv(OUT/'tables/nine_panel_month_identifiers.csv')
    assert len(ids)==564 and not ids.return_date.duplicated().any()
    assert len(curve)==9*120
    m=pd.read_csv(OUT/'tables/table1_performance.csv')
    assert set(m.policy)=={'linear','gaussian','matern32','neural'}
    assert m.months.eq(564).all()
    for seed in PROTOCOL['neural']['seeds']:
        a=pd.read_csv(OUT/'tables'/f'neural_annual_seed_{seed}.csv')
        assert len(a)==47 and a.parameter_change.gt(0).all() and a.head_residual.max()<1e-8
    pages={}
    for name in ['empirical_main','empirical_appendix_standalone','empirical_combined','full_manuscript']:
        reader=PdfReader(pub/f'{name}.pdf');pages[name]=len(reader.pages)
        for page in reader.pages:
            text=page.extract_text() or ''
            assert not re.search(r'\?\?|PLACEHOLDER|TODO|nan\b',text,re.I),f'Unresolved content: {name}'
    if not 10<=pages['empirical_main']<=12:raise ValueError('Main section page limit: '+str(pages))
    assert pages['empirical_combined']==pages['empirical_main']+pages['empirical_appendix_standalone']
    source=(pub/'empirics.tex').read_text()
    assert source.count(r'\begin{figure}')==5
    assert source.count(r'\input{tables/')==2
    # Word target applies to main prose, excluding display environments/captions.
    prose=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}','',source,flags=re.S)
    prose=re.sub(r'\\[A-Za-z]+(?:\[[^]]*\])?\{[^}]*\}','',prose)
    words=len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",prose))
    assert 2000<=words<=2600,f'Main prose words: {words}'
    full=pub/'full_source';archive=json.loads((full/'source_archive_manifest.json').read_text())
    changed={'main.tex','empirics.tex','tables/performance.tex'}
    for name,value in archive['files'].items():
        if name in changed or name.startswith('figures/'):continue
        if digest(full/name)!=value:raise ValueError('Nonempirical archive source changed: '+name)
    with zipfile.ZipFile(pub/'Portfolio_Empirics_Compact_LaTeX.zip') as z:
        names=set(z.namelist())
        for name in ['fig01_kernel_wealth','fig02_managed_spectrum','fig03_gaussian_complexity','fig04_gaussian_nine_panels','fig05_economic_holdings']:
            assert 'Portfolio/figures/'+name+'.pdf' in names
        assert 'Portfolio/tables/performance.tex' in names and 'Portfolio/tables/spectral_value.tex' in names
    # Require source-map numerical macros to match independently evaluated CSV values.
    mapping=json.loads((OUT/'audit/source_map.json').read_text())
    for claim in mapping['main_text_claims']:
        if claim['input'].startswith('tables/') and 'column' in claim and 'filters' in claim:
            f=pd.read_csv(OUT/claim['input'])
            for k,v in claim['filters'].items():f=f[f[k].eq(v)]
            assert len(f)==1
            np.testing.assert_allclose(float(f.iloc[0][claim['column']])*claim.get('scale',1),claim['raw_value'],atol=1e-10)
    audit('final_verification',dict(passed=True,pages=pages,main_prose_words=words,main_figures=5,main_tables=2,
        observed_months=564,curve_points=len(curve),neural_seeds_completed=PROTOCOL['neural']['seeds'],
        archive_theory_bytes_preserved=True,latex_zip_complete=True,source_map_checks=True,
        limitations=['Conditional complete-payoff sample; 7489 unresolved formation payoffs',
                    'Archived 3.350/1.76 result lacks run-level provenance; historical cause unverified',
                    'Conditional path resampling is not full-pipeline inference',
                    'Finite neural budgets do not certify global nonconvex convergence']))
    print(json.dumps(dict(pages=pages,main_prose_words=words,passed=True),indent=2))


if __name__=='__main__':main()
