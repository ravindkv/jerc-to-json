# coding: utf-8
from __future__ import annotations

import argparse
import os
import yaml

from collections import defaultdict
from functools import partial

from rich.console import Console

from jerc2json.proxy import JERCTextFileProxy
from jerc2json.validation import validate_json_jme
from jerc2json.io import read_sections
from jerc2json.util import jme_filename_from_keys

from jerc2json.cli.util import load_check_config, setup_config_eras, get_correction_infos


def validate(console: Console, args: argparse.Namespace):
    """
    Evaluate the closure between corrections obtained with CMSSW from the
    JERC text files and with the newly generated correctionlib JSON files.
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

    # iterate through the requested eras
    for era in args.eras:
        era_cfg = config.eras[era]

        # find JSON files in output directory
        output_dir = os.path.join(config.output_dir, era)
        if not os.path.isdir(output_dir):
            raise RuntimeError(
                f"output directory '{output_dir}' not found; "
                f"JSON file creation for era '{era}' must be "
                "run before validation"
            )

        outputs = era_cfg.get("outputs", config.defaults.outputs)
        algo_to_json_file = {}
        for output_cfg in outputs:
            for algo in output_cfg["algos"]:
                algo_to_json_file[algo] = os.path.join(
                    output_dir,
                    f"{output_cfg['name']}.json",
                )

        # list of dictionaries with information about
        # what should be done for each correction
        correction_infos = get_correction_infos(config, era)

        validation_results = defaultdict(dict)

        # loop through corrections
        for corr_info in correction_infos:
            type_ = corr_info["type"]
            version = corr_info["version"]
            name = corr_info["name"]
            inputs = corr_info["inputs"]
            last_run = corr_info["last_run"]
            algos = corr_info["algos"]
            levels = corr_info["levels"].simple
            compound_levels_map = corr_info["levels"].get("compound", {})

            # handle old-style config (no 'inputs')
            # -> build 'inputs' using 'name' as a single input tag
            if not inputs:
                inputs = [{
                    # JERC tag used as an input for the correction
                    "tag": name,
                }]

            # abort if inputs are given but no tag is provided
            if any("tag" not in inp for inp in inputs):
                raise ValueError(
                    f"required key 'tag' not found in all inputs for correction '{name}'"
                )

            # checks for run-rependent corrections
            # abort if first_run are given but no tag is provided
            first_runs = [inp.get("first_run", None) for inp in inputs]
            has_first_runs = [first_run is not None for first_run in first_runs]
            has_last_run = last_run is not None
            if has_last_run and not all(has_first_runs):
                raise ValueError(
                    "required key 'first_run' not found in all inputs "
                    f"for run-dependent correction '{name}'"
                )
            elif any(has_first_runs) and not has_last_run:
                raise ValueError(
                    "required key 'last_run' not found "
                    f"for run-dependent correction '{name}'"
                )
            elif len(inputs) >= 2 and not all(has_first_runs):
                raise ValueError(
                    "error combining multiple correction inputs: got "
                    f"{len(inputs)} (>=2) inputs for correction '{name}', "
                    "but at least one is missing a `first_run`",
                )

            # need to use patched text files for JER
            filename_suffix = "_patched" if type_ == "jer" else ""

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
                # find JSON file with algo results
                json_file = algo_to_json_file[algo]
                results = validation_results[json_file]

                # curry validation function
                validation_func = partial(
                    validate_json_jme,
                    config=config,
                    json_file=json_file,
                    values_from={
                        "simple": "config",
                        "grid": "grid",
                    }[args.validation_type],
                    quiet=not args.verbose,
                )

                # individual correction levels
                for level in levels:
                    for inp in inputs:
                        # if run-dependent corrections, override
                        # 'run' input based on IOV
                        if has_last_run:
                            validation_func = partial(
                                validation_func,
                                override_test_values={
                                    "run": [inp["first_run"]],
                                },
                            )

                        # path to JME file
                        jme_text_file = os.path.join(
                            config.work_dir,
                            jme_filename_from_keys(
                                name=inp["tag"], level=level, algo=algo, suffix=filename_suffix,
                            ),
                        )
                        print(f"Validating: {jme_text_file}")

                        json_correction_name = f"{name}_{level}_{algo}"  # TODO: use labeler

                        if level in ("ScaleFactor", "SF"):
                            for syst in ("nom", "up", "down"):
                                results[f"{json_correction_name}_{syst}"] = validation_func(
                                    json_correction_name=json_correction_name,
                                    jme_text_files=[jme_text_file],
                                    jme_mode="jer_sf",
                                    jme_args=[syst],
                                )
                        else:
                            results[json_correction_name] = validation_func(
                                json_correction_name=json_correction_name,
                                jme_text_files=[jme_text_file],
                                jme_mode=type_,
                            )

                # FIXME: issue with `grid` + compound corrections -> skip
                # # compound correction levels
                # for (output_level, input_levels) in compound_levels_map.items():
                #     json_compound_correction_name = f"{name}_{output_level}_{algo}"
                #     jme_text_files = [
                #         os.path.join(
                #             config.work_dir,
                #             jme_filename_from_keys(
                #                 name=name, level=input_level, algo=algo,
                #             ),
                #         )
                #         for input_level in input_levels
                #     ]
                #
                #     validation_results[json_compound_correction_name] = validation_func(
                #         json_correction_name=json_compound_correction_name,
                #         jme_text_files=jme_text_files,
                #         jme_mode=type_,
                #     )

                # JEC uncertainties (MC only)
                if type_ == "jec" and name.endswith("MC"):

                    # file with collected JEC sources
                    collected_uncertainty_file = os.path.join(
                        config.work_dir,
                        jme_filename_from_keys(
                            name=name,
                            level="UncertaintySources",
                            algo=algo,
                            suffix="_collected",
                        )
                    )
                    if not os.path.isfile(collected_uncertainty_file):
                        print(
                            "WARNING: could not validate uncertainties: the collected uncertainties file "
                            f"({collected_uncertainty_file}) was not found; please run the "
                            "create_jsons.py with the same config before doing any validation",
                        )
                        continue

                    # retrieve uncertainty sources
                    uncertainty_sources = read_sections(collected_uncertainty_file)
                    for uncertainty_source in uncertainty_sources:
                        print(f"Validating: {collected_uncertainty_file} [{uncertainty_source}]")

                        json_correction_name = f"{name}_{uncertainty_source}_{algo}"  # TODO: use labeler

                        results[json_correction_name] = validation_func(
                            json_correction_name=json_correction_name,
                            jme_text_files=[collected_uncertainty_file],
                            jme_mode="jec_uncertainty",
                            jme_args=[uncertainty_source],
                        )

        # store validation results
        for json_file, results in validation_results.items():
            json_file_basename, _ = os.path.splitext(json_file)
            results_file = f"{json_file_basename}.validation_{args.validation_type}.yml"
            with open(results_file, "w") as f:
                yaml.dump(results, f)
            print(f"Wrote validation results to file: {results_file}")


def setup_validate(subparsers: argparse._SubParsersAction):
    """
    Add the CLI parameters required for subcommand 'validate'.
    """

    sp = subparsers.add_parser("validate", help=validate.__doc__)
    sp.set_defaults(command=validate)

    setup_config_eras(sp)

    sp.add_argument(
        "-v",
        "--validation-type",
        default="simple",
        choices=("simple", "grid"),
        help="type of validation to perform; simple = take test points from config, "
        "grid = compute validation grid from correction binning",
    )

    sp.add_argument(
        "--verbose",
        action="store_true",
        help="if given, more detailed information will be shown",
    )
