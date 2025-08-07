# coding: utf-8
from __future__ import annotations

import argparse

from rich.console import Console

from jerc2json.config import ConfigDict
from jerc2json.io import get_unpack_tarball
from jerc2json.cli.util import load_check_config, setup_config_eras, get_correction_infos


def get_location(config: ConfigDict, source: str, rpath: str):
    """
    Return a URL or a path to a local file, given a `config`, a source
    name (`source`) and a relative path (`rpath`). The source should be
    defined inside the `config` object under the key `sources`.
    """
    # validate config
    if 'sources' not in config:
        raise ValueError(
            "cannot get file location; invalid config: does not contain a `sources` dict",
        )

    # validate source
    if source not in config.sources:
        raise ValueError(
            f"cannot get file location; no source called `{source}` defined in config",
        )

    # get source info from config
    cfg_source = config.sources[source]

    # check provided source is supported
    valid_sources = {"github", "localdir"}
    if cfg_source["type"] not in valid_sources:
        valid_srcs_str = ", ".join(sorted(valid_sources))
        raise ValueError(
            f"cannot get file location; invalid source type '{cfg_source['type']}'; "
            f"valid choices: {valid_srcs_str}",
        )

    # handle sources

    # github
    if cfg_source['type'] == "github":
        return f"{cfg_source['url']}/raw/{cfg_source['branch']}/{rpath}"

    # localdir
    elif cfg_source['type'] == "localdir":
        return f"{cfg_source['dir']}/{rpath}"

    else:
        assert False, f"internal error: source type '{cfg_source['type']}' not supported"


def get(console: Console, args: argparse.Namespace):
    """
    Download and extract the tarballs containing JERC text files into the work directory.
    """
    # load config, check file exists and eras are valid
    config = load_check_config(console, args)

    # iterate through the requested eras
    for era in args.eras:
        # list of dictionaries with information about
        # what should be done for each correction
        correction_infos = get_correction_infos(config, era)

        # loop through corrections
        for corr_info in correction_infos:
            type_ = corr_info["type"]
            name = corr_info["name"]
            inputs = corr_info["inputs"]

            # handle old-style config (no 'inputs')
            # -> build 'inputs' using 'name' as a single input tag
            if not inputs:
                inputs = [{
                    # JERC tag used as an input for the correction
                    "tag": name,
                }]

            # loop through input files
            for inp in inputs:
                # obtain location (URL/path) of tarball
                tarball_location = get_location(
                    config=config,
                    source=type_,
                    rpath=f"tarballs/{inp['tag']}.tar.gz",
                )

                # obtain and unpack the tarball containing the JERC text files
                get_unpack_tarball(
                    src=tarball_location,
                    work_dir=config.work_dir,
                    local_filename=f"{inp['tag']}.tar.gz",
                )


def setup_get(subparsers: argparse._SubParsersAction):
    """
    Add the CLI parameters required for subcommand 'get'.
    """
    sp = subparsers.add_parser("get", help=get.__doc__)
    sp.set_defaults(command=get)

    setup_config_eras(sp)
