# coding: utf-8
from __future__ import annotations

import argparse
import os

from rich.console import Console

from jerc2json.io import calc_file_hash
from jerc2json.util import jme_filename_from_keys

from jerc2json.cli.util import load_check_config, setup_config_eras, get_correction_infos


def compare_payloads(console: Console, args: argparse.Namespace):
    """
    Print a summary table containing the payload hashes for each correction within
    an era (useful for consistency checks).
    """
    # load config, check file exists and eras are valid
    config = load_check_config(console, args)

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

    try:
        import pandas as pd
    except (ModuleNotFoundError, ImportError) as e:
        e.message = "could not import `pandas` package, which is needed to run the `compare_payloads` command"
        raise e

    # iterate through the requested eras
    for era in args.eras:
        # list of dictionaries with information about
        # what should be done for each correction
        correction_infos = get_correction_infos(config, era)

        # loop through corrections
        records_for_df = []
        for corr_info in correction_infos:
            type_ = corr_info["type"]
            name = corr_info["name"]
            algos = corr_info["algos"]
            levels = corr_info["levels"].simple
            compound_levels_map = corr_info["levels"].get("compound", {})

            # check if all compound input levels are in `levels`
            compound_missing_inputs = set(
                compound_level
                for compound_levels in compound_levels_map.values()
                for compound_level in compound_levels
            ) - set(levels)
            if compound_missing_inputs:
                compound_missing_inputs_str = ", ".join(sorted(compound_missing_inputs))
                raise ValueError(
                    "the following corrections levels appear as inputs under 'compound', "
                    f"but are not declared in 'simple': {compound_missing_inputs_str}",
                )

            # process jet algorithms
            for algo in algos:
                # individual correction levels
                for level in levels:
                    jme_text_file = os.path.join(
                        config.work_dir,
                        jme_filename_from_keys(
                            name=name,
                            level="SF" if level == "ScaleFactor" else level,
                            algo=algo,
                        ),
                    )
                    if not os.path.isfile(jme_text_file):
                        continue

                    jme_text_file_hash = (
                        calc_file_hash(jme_text_file).hexdigest()
                        if os.path.isfile(jme_text_file)
                        else None
                    )
                    print(f"Processing: {jme_text_file} -> {jme_text_file_hash}")

                    records_for_df.append({
                        "type": type_,
                        "name": name,
                        "algo": algo,
                        "level": level,
                        "hash": jme_text_file_hash[:args.hash_maxlen] if jme_text_file_hash else None,
                    })

                # JEC uncertainties (MC only)
                if type_ == "jec" and name.endswith("MC"):

                    # file with collected JEC sources
                    uncertainty_file = os.path.join(
                        config.work_dir,
                        jme_filename_from_keys(
                            name=name,
                            level="UncertaintySources",
                            algo=algo,
                        )
                    )
                    if not os.path.isfile(uncertainty_file):
                        continue

                    uncertainty_file_hash = (
                        calc_file_hash(uncertainty_file).hexdigest()
                        if os.path.isfile(uncertainty_file)
                        else None
                    )
                    print(f"Processing: {uncertainty_file} -> {uncertainty_file_hash}")

                    records_for_df.append({
                        "type": type_,
                        "name": name,
                        "algo": algo,
                        "level": "UncertaintySources",
                        "hash": uncertainty_file_hash[:args.hash_maxlen] if uncertainty_file_hash else None,
                    })

        # convert to dataframe for easy data manipulation
        df = pd.DataFrame.from_records(records_for_df)

        # show the hashes as a pivot table separately for each jet collection
        # TODO: ass ways to customize this
        types = set(df["type"])
        for type_ in types:
            df_type = df[df["type"] == type_]
            df_type_pivoted = pd.pivot(
                df_type,
                index=["algo", "type", "level"],
                columns=["name"],
                values="hash",
            )
            print(
                df_type_pivoted.fillna("--"),
            )

        # # store validation results
        # for json_file, results in validation_results.items():
        #     json_file_basename, _ = os.path.splitext(json_file)
        #     results_file = f"{json_file_basename}.validation_{args.validation_type}.yml"
        #     with open(results_file, "w") as f:
        #         yaml.dump(results, f)
        #     print(f"Wrote validation results to file: {results_file}")


def setup_compare_payloads(subparsers: argparse._SubParsersAction):
    """
    Add the CLI parameters required for subcommand 'compare_payloads'.
    """

    sp = subparsers.add_parser("compare_payloads", help=compare_payloads.__doc__)
    sp.set_defaults(command=compare_payloads)

    setup_config_eras(sp)

    sp.add_argument(
        "--comparison-type",
        default="simple",
        choices=("hash",),
        help="type of comparison; hash = md5sum hash of the input file",
    )

    sp.add_argument(
        "--hash-maxlen",
        type=int,
        default=None,
        help="if given, hash values will be truncated at this string length",
    )

    # sp.add_argument(
    #     "--verbose",
    #     action="store_true",
    #     help="if given, more detailed information will be shown",
    # )
