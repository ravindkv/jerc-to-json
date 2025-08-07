# coding: utf-8
"""
Utility functions used by CLI commands
"""
import argparse
import os

from rich.console import Console

from jerc2json.config import ConfigDict, load_config, default_config_path

__all__ = ["load_check_config", "setup_config_eras", "get_correction_infos"]


def load_check_config(console: Console, args: argparse.Namespace) -> ConfigDict:
    """
    Check that the `config` supplied to the CLI command exists,
    and that the `eras` supplied are found in the config.

    Returns the config object.
    """
    # check config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(
            f"specified configuration file does not exist: {args.config}",
        )

    # load config
    config = load_config(args.config)

    # validate eras
    unknown_eras = set(args.eras) - set(config.eras)
    if unknown_eras:
        unknown_eras_str = ", ".join(sorted(unknown_eras))
        available_eras_str = ", ".join(sorted(config.eras))
        raise ValueError(
            "the following eras were not found "
            f"in the config: {unknown_eras_str}; "
            f"available eras are: {available_eras_str}",
        )

    return config


def get_correction_infos(config: ConfigDict, era: str) -> list[dict]:
    """
    Return a list of dictionaries with information about what should be done for
    each correction.

    This code interprets the configuration for an `era` and returns
    a list of dicts with a more regular structure understood by
    the actual JSON conversion functions.

    Example of valid era configuration:

    .. code-block:: yaml

        eras:
          ERA_NAME:
            jec:
              names:
                # specify only JEC tag as string (discouraged)
                - FIRST_CORRECTION_NAME

                # specify JEC tag as 'name' in a dict
                - name: SECOND_CORRECTION_NAME

                # use different name than the tag
                - name: THIRD_CORRECTION_NAME
                  inputs:
                    - tag: JEC_TAG_FOR_THIRD_CORRECTION

                # create run-dependent correction from
                # multiple JEC tags
                - name: FOURTH_CORRECTION_NAME
                  inputs:
                    - tag: FIRST_JEC_TAG_FOR_FOURTH_CORRECTION
                      first_run: FIRST_RUN_FOR_FIRST_JEC_TAG
                    - tag: SECOND_JEC_TAG_FOR_FOURTH_CORRECTION
                      first_run: FIRST_RUN_FOR_SECOND_JEC_TAG
                  last_run: LAST_RUN_FOR_LAST_JEC_TAG

            jer:
              names:
                - name: JER_NAME
    """
    # get era config
    era_cfg = config.eras[era]

    # list of dictionaries with information about
    # what should be done for each correction
    correction_infos = []
    for correction_type in ("jec", "jer"):
        cfg = era_cfg.get(correction_type, None)
        if cfg is None:
            continue

        for i_name, d_name in enumerate(cfg.get("names", None) or []):
            # if 'name' is a simple string, cast to dict
            if isinstance(d_name, str):
                d_name = dict(name=d_name)

            # check type
            if not isinstance(d_name, dict):
                raise ValueError(
                    "error parsing config for era "
                    f"'{era}/{correction_type}/{i_name}': "
                    f"expecting 'names' entry to be dict, got ({type(d_name)})"
                )

            # check name
            if "name" not in d_name:
                raise ValueError(
                    "error parsing config for era "
                    f"'{era}/{correction_type}/{i_name}': "
                    f"'names' entry missing mandatory key 'name'"
                )

            correction_infos.append({
                "type": correction_type,
                # name of correction in JSON file
                "name": d_name["name"],
                # input configuration (list of dicts, one per IOVs)
                "inputs": d_name.get("inputs", None),
                # last run of validity (run-dependent corrections only)
                "last_run": d_name.get("last_run", None),
                # other keys
                "version": int(cfg.get("correction_version", config.defaults.get("correction_version", 1))),
                "algos": cfg.get("algos", config.defaults.algos),
                "levels": cfg.get("levels", config.defaults.levels[correction_type]),
                "uncertainty_sets": cfg.get("uncertainty_sets", config.defaults.get("uncertainty_sets", None)),
            })

    return correction_infos


def setup_config_eras(parser: argparse.ArgumentParser):
    """
    Add `config` and `eras` CLI parameters, required by several subcommands.
    """

    parser.add_argument(
        "-c",
        "--config",
        default=default_config_path,
        help=f"YAML configuration file containing information about corrections to be "
        f"processed for each era; default: {os.path.basename(default_config_path)}",
    )

    parser.add_argument(
        "-e",
        "--eras",
        nargs="+",
        required=True,
        help="which eras to process; available eras, including all era-specific "
        "information, are configured in the configuration file",
    )
