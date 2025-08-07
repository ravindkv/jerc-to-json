# coding: utf-8
from __future__ import annotations

from rich.console import Console

from jerc2json._version import __version__

from jerc2json.cli.get import setup_get
from jerc2json.cli.create import setup_create
from jerc2json.cli.validate import setup_validate
from jerc2json.cli.plot_validation import setup_plot_validation
from jerc2json.cli.compare_payloads import setup_compare_payloads


def main() -> int:
    from argparse import ArgumentParser

    ap = ArgumentParser(prog="jerc2json", description=__doc__)

    # global parameters
    ap.add_argument(
        "--width",
        type=int,
        default=100,
        help="set the rich console output width",
    )
    ap.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # add the subcommands' CLI parameters to the parser
    subparsers = ap.add_subparsers()
    setup_get(subparsers)
    setup_create(subparsers)
    setup_validate(subparsers)
    setup_plot_validation(subparsers)
    setup_compare_payloads(subparsers)

    # parse the arguments received
    args = ap.parse_args()

    # initialize the rich console
    console = Console(
        width=args.width,
    )

    # if a command was passed, run it
    if hasattr(args, "command"):
        retcode: int = args.command(console, args)
        return retcode

    # otherwise, display usage information
    ap.parse_args(["-h"])
    return 0


if __name__ == "__main__":
    exit(main())
