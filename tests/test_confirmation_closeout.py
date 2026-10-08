"""A Python module launcher is not a pytest marker filter."""
import pytest

from simulations.diagnostics.closeout_v2 import full_suite_command


@pytest.mark.parametrize('command', [
    ['/usr/bin/python3', '-m', 'pytest', 'tests', '-q'],
    ['/usr/bin/pytest', 'tests', '-q'],
])
def test_accepts_unfiltered_full_suite(command):
    assert full_suite_command(command)


@pytest.mark.parametrize('arguments', [
    ['tests', '-m', 'slow'], ['tests', '-mslow'],
    ['tests', '-k', 'cache'], ['tests', '-kcache'],
    ['tests', '--ignore', 'tests/test_example.py'],
    ['tests', '--ignore=tests/test_example.py'],
    ['tests', '--ignore-glob=tests/test_example*'],
    ['tests', '--deselect=tests/test_example.py::test_one'],
    ['tests/test_example.py', '-q'],
])
def test_rejects_filtered_or_partial_suite(arguments):
    assert not full_suite_command(['/usr/bin/python3', '-m', 'pytest', *arguments])
