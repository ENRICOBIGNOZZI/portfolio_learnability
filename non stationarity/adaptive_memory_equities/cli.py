"""Local-only audit, benchmark, smoke, run and reporting entry points."""
import argparse
import json
from pathlib import Path
from .config import PACKAGE, load_config, ResourceMonitor, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','smoke','run','report','benchmark','synthetic'])
    p.add_argument('--config',default=str(PACKAGE/'configs/smoke.yaml'))
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if args.command == 'report':
        from .reporting import report
        if args.run_dir is None:
            p.error('--run-dir required')
        print(json.dumps(report(args.run_dir),indent=2))
        return
    c = load_config(args.config)
    directory = args.run_dir or PACKAGE/'private_runs'/args.command
    directory = directory.resolve()
    if not directory.is_relative_to(PACKAGE/'private_runs'):
        p.error('Run outputs must stay under ignored private_runs/.')
    with ResourceMonitor(directory/'memory.jsonl',**c['resources']) as monitor:
        if args.command == 'audit':
            from .data_adapter import audit
            result = audit(directory,monitor)
        elif args.command == 'benchmark':
            from .benchmark import benchmark
            result = benchmark(c,directory,monitor)
        elif args.command == 'synthetic':
            from .synthetic import run_synthetic
            result = run_synthetic(directory,monitor)
        else:
            from .prequential import run
            from .reporting import report
            try:
                result = run(c,directory,args.resume,monitor)
                report(directory)
            except (ValueError,MemoryError,ArithmeticError) as error:
                write_json(directory/'failure.json',dict(status='PAYOFF_COMPLETENESS_FAILED' if 'PAYOFF_COMPLETENESS_FAILED' in str(error)
                    else 'BLOCKED',reason=str(error)))
                raise
        print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
