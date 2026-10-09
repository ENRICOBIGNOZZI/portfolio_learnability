"""Workspace boundary checks using the initial read-only inventory when available."""
import json
import subprocess
import pytest
from adaptive_memory_equities.config import PACKAGE, REPO, digest


def test_preexisting_tracked_artifacts_unchanged():
    initial=PACKAGE/'private_runs/initial_state.json'
    if not initial.exists():
        pytest.skip('Initial workspace inventory belongs to this local task, not distribution.')
    state=json.loads(initial.read_text())
    changed=[p for p,h in state['hashes'].items() if not (REPO/p).is_file() or digest(REPO/p)!=h]
    assert changed==[]


def test_private_artifacts_excluded_from_publishable_tree():
    public=subprocess.check_output(['git','-C',str(REPO),'ls-files','--others','--exclude-standard'],text=True)
    assert 'private_runs/' not in public
    path=PACKAGE/'private_runs/smoke/manifest.json'
    result=subprocess.run(['git','-C',str(REPO),'check-ignore',str(path)],capture_output=True,text=True)
    assert result.returncode==0
