"""Frozen configuration, provenance and local resource limits."""
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
import psutil
import yaml

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str))
    os.replace(tmp, path)


def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    if c['sample_mode'] not in ['formation_audit', 'retrospective_complete_payoff']:
        raise ValueError('Explicit sample mode required.')
    if c['seeds'] != [0, 1] or c['optimizer']['weight_decay'] != 0:
        raise ValueError('Require both fixed seeds and no weight decay.')
    if c['optimizer']['steps'] < 1 or c['optimizer']['checkpoint_rule'] != 'final_fixed_step':
        raise ValueError('Invalid frozen optimizer budget.')
    if c['minimum_n_eff'] < 36:
        raise ValueError('Minimum effective history is 36 months.')
    return c


def provenance(c):
    import importlib.metadata
    code = {str(p.relative_to(REPO)): digest(p) for p in PACKAGE.glob('*.py')}
    code.update({p: digest(REPO / p) for p in ['data_pipeline.py', 'characteristic_selection.json']})
    return dict(config=c, config_hash=canonical_hash(c), code_hash=canonical_hash(code), code=code,
                git_sha=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
                working_tree_diff_hash=hashlib.sha256(subprocess.check_output(
                    ['git', '-C', str(REPO), 'diff', 'HEAD', '--binary'])).hexdigest(),
                platform=platform.platform(), python=platform.python_version(),
                versions={p: importlib.metadata.version(p) for p in ['torch','numpy','pandas','pyarrow','scipy','psutil']},
                total_ram_bytes=psutil.virtual_memory().total, device='cpu',
                untracked_experiment_code_included_in_code_hash=True)


class ResourceMonitor:
    """One-second sampling; cooperative abort at bounded work checkpoints."""
    def __init__(self, path, rss_limit_mb=1600, minimum_available_mb=250):
        self.path = Path(path)
        self.rss_limit = rss_limit_mb * 2**20
        self.available_limit = minimum_available_mb * 2**20
        self.stop_event = threading.Event()
        self.failure = None
        self.peak = 0
        self.minimum_available = float('inf')

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        process = psutil.Process()
        process.cpu_percent()
        psutil.cpu_percent()
        previous_disk = psutil.disk_io_counters()
        previous_time = time.monotonic()
        cpu_max = 0.
        with self.path.open('a') as stream:
            while not self.stop_event.is_set():
                rss = process.memory_info().rss
                available = psutil.virtual_memory().available
                self.peak = max(self.peak, rss)
                self.minimum_available = min(self.minimum_available, available)
                now = time.monotonic()
                disk = psutil.disk_io_counters()
                elapsed = max(now-previous_time,1e-6)
                cpu = process.cpu_percent()
                cpu_max = max(cpu_max,cpu)
                stream.write(json.dumps(dict(time=time.time(), rss_mb=rss/2**20,
                    available_mb=available/2**20, swap_used_mb=psutil.swap_memory().used/2**20,
                    process_cpu_percent=cpu,system_cpu_percent=psutil.cpu_percent(),threads=process.num_threads(),
                    system_load_average=os.getloadavg(),
                    system_read_mib_s=(disk.read_bytes-previous_disk.read_bytes)/2**20/elapsed,
                    system_write_mib_s=(disk.write_bytes-previous_disk.write_bytes)/2**20/elapsed,
                    disk_free_gib=psutil.disk_usage(self.path.parent).free/2**30))+'\n')
                previous_disk,previous_time = disk,now
                stream.flush()
                if rss > self.rss_limit:
                    self.failure = 'RESOURCE_LIMIT: process RSS exceeded configured limit'
                if available < self.available_limit:
                    self.failure = 'RESOURCE_LIMIT: system available RAM below configured floor'
                self.stop_event.wait(1)
        self.cpu_max = cpu_max

    def check(self):
        if self.failure:
            raise MemoryError(self.failure)

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        write_json(self.path.with_suffix('.summary.json'), dict(peak_rss_mb=self.peak/2**20,
            minimum_available_mb=self.minimum_available/2**20,interval_seconds=1,failure=self.failure,
            process_cpu_max=getattr(self,'cpu_max',0),process_cpu_units='100 percent = one logical core'))
