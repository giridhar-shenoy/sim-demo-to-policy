import sdp
from sdp.cli import main


def test_version_string():
    assert sdp.__version__.startswith("0.1")


def test_cli_runs():
    assert main([]) == 0
