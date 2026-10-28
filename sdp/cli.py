"""Command-line entry point (`sdp`)."""

import argparse

from sdp import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sdp", description=__doc__)
    parser.add_argument("--version", action="version", version=f"sdp {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
