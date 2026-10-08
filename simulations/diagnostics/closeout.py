"""Write a final evidence report only after numerical and full-PDF acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from simulations.diagnostics.verify import verify
from simulations.provenance import require, file_hash
from simulations.manuscript import ROOT, verify_preservation


def closeout():
    output = ROOT/'simulations/outputs/paper'
    numerical = verify(output, 'paper')
    preservation = verify_preservation()
    compilation = json.loads((output/'paper/manuscript_compilation.json').read_text())
    pdf = ROOT/compilation['pdf']
    require(compilation['profile']=='paper' and compilation['compiled'], 'Closeout verification failed at line 19')
    require(hashlib.sha256(pdf.read_bytes()).hexdigest()==compilation['pdf_sha256'], 'Closeout verification failed at line 20')
    require(hashlib.sha256((ROOT/'paper/main.tex').read_bytes()).hexdigest()==compilation['source_main_sha256'], 'Closeout verification failed at line 21')
    for name, expected in compilation['generated_source_hashes'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name)
    require(bool(compilation.get('compiler_input_hashes')), 'Missing compiler dependency hashes')
    for name, expected in compilation['compiler_input_hashes'].items():
        require(file_hash(ROOT/name) == expected, 'Stale PDF dependency: '+name)
    visual = json.loads((output/'paper/visual_review.json').read_text())
    require(visual['profile']=='paper' and visual['passed'], 'Closeout verification failed at line 25')
    require(visual['pdf_sha256']==compilation['pdf_sha256'], 'Closeout verification failed at line 26')
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    pages = int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:')))
    require(visual['pages_reviewed']==pages, 'Closeout verification failed at line 29')
    tests = json.loads((ROOT/'simulations/outputs/audit/test_verification.json').read_text())
    for name, expected in tests.get('source_hashes', {}).items():
        require(file_hash(ROOT/name) == expected, 'Stale test evidence: '+name)
    require(bool(tests.get('source_hashes')) and tests.get('return_code') == 0, 'Missing executed test evidence')
    require(tests['passed'], 'Closeout verification failed at line 31')
    cleanup = json.loads((ROOT/'simulations/outputs/audit/cleanup_inventory.json').read_text())
    main = pd.read_csv(output/'data/baseline/main_results.csv')
    rates = pd.read_csv(output/'data/baseline/rates.csv')
    runtime = sum(pd.read_csv(path).runtime_seconds.sum() for path in (output/'data').glob('*/seeds.csv'))
    source = json.loads((ROOT/'simulations/outputs/audit/manuscript_preservation.json').read_text())
    result = {'passed':True, 'numerical':numerical, 'preservation':preservation,
              'manuscript_pages':pages, 'manuscript_source':source['source_tree'],
              'manuscript_source_selection':source['selection_evidence'],
              'manuscript_source_user_confirmation':source['user_confirmation'],
              'tests':tests, 'legacy_paths_deleted':len(cleanup),
              'sum_replication_elapsed_seconds':float(runtime),
              'runtime_note':'Sum of individual replication elapsed times; parallel work means this is not wall-clock duration.',
              'headline_largest_T':main.iloc[-1].to_dict(),
              'headline_rate_regressions':rates.to_dict(orient='records'),
              'figure_pdf_count':len(list((output/'figures').glob('*.pdf'))),
              'table_csv_count':len(list((output/'tables').glob('*.csv'))),
              'compiled_pdf':compilation['pdf'], 'pdf_sha256':compilation['pdf_sha256'],
              'limitations':['Finite Nyström representation with separately passed rank sensitivity.',
                            'Finite population quadrature with separately passed sensitivity.',
                            'Fixed smooth target need not attain a minimax envelope with equality.',
                            'Empirical-rank and heteroskedastic variants deliberately violate specified baseline assumptions.'],
              'reproduction_commands':[
                  'PYTHONPYCACHEPREFIX=/tmp/codex_dgp_pycache VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.run --profile smoke',
                  'PYTHONPYCACHEPREFIX=/tmp/codex_dgp_pycache VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m simulations.run --profile paper --workers 4',
                  'python3 -m simulations.manuscript',
                  'python3 -m simulations.diagnostics.closeout']}
    (output/'audit/final_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    report=closeout()
    print(json.dumps({key:report[key] for key in ('passed','legacy_paths_deleted','manuscript_pages','compiled_pdf')},indent=2))
