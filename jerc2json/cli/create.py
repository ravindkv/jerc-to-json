# coding: utf-8
from __future__ import annotations

import argparse
import gzip
import os

from collections import defaultdict

from correctionlib.schemav2 import CorrectionSet
from rich.console import Console

from jerc2json.conversion import (
    JERCToCorrectionConverter,
    JERCToRunDependentCorrectionConverter,
    make_compound_correction,
)
from jerc2json.io import read_sections, compile_jme_uncertainties, patch_jer_file
from jerc2json.proxy import JERCTextFileProxy
from jerc2json.util import jme_filename_from_keys

from jerc2json.cli.get import get
from jerc2json.cli.util import load_check_config, setup_config_eras, get_correction_infos


DEFAULT_JEC_UNCERTAINTY_SETS = {
    "default": {
        "filename_prefix": "",
        "uncertainty_prefix": "",
    },
}


def create(console: Console, args: argparse.Namespace):
    """
    Create JSON files from JERC text files. If not present, the tarballs
    containing the JERC text tiles are downloaded and extracted into the
    work directory.
    """
    # load config, check file exists and eras are valid
    config = load_check_config(console, args)

    # run get command to ensure all files are present
    get(console, args)

    # iterate through the requested eras
    for era in args.eras:
        era_cfg = config.eras[era]

        # resolve output files in which to place
        # corrections for each algorithm
        outputs = era_cfg.get("outputs", config.defaults.outputs)
        algo_to_outputs = defaultdict(list)
        for output_cfg in outputs:
            for algo in output_cfg["algos"]:
                algo_to_outputs[algo].append(output_cfg["name"])

        # list of dictionaries with information about
        # what should be done for each correction
        correction_infos = get_correction_infos(config, era)

        # containers for individual correction objects
        corrections = defaultdict(list)
        compound_corrections = defaultdict(list)

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
                # find output(s) for algo results
                outputs = algo_to_outputs[algo]
                if not outputs:
                    print(
                        f"WARNING: no outputs registered for algorithm '{algo}', "
                        "any JSON files produced will not contain these corrections",
                    )
                    continue

                # individual correction levels
                level_converters = {}
                for level in levels:
                    # handle input files
                    jme_text_files = []
                    for inp in inputs:
                        jme_text_file = os.path.join(
                            config.work_dir,
                            jme_filename_from_keys(
                                name=inp["tag"], level=level, algo=algo, suffix=filename_suffix,
                            ),
                        )
                        jme_text_files.append(jme_text_file)

                        # if JER, patch file
                        if type_ == "jer":
                            jme_text_file_orig = os.path.join(
                                config.work_dir,
                                jme_filename_from_keys(
                                    name=inp["tag"],
                                    # adapt to different naming convention
                                    level="SF" if level == "ScaleFactor" else level,
                                    algo=algo,
                                ),
                            )
                            if not os.path.isfile(jme_text_file):
                                print(f"Patching: {jme_text_file_orig}")
                                patch_jer_file(
                                    jme_text_file_orig,
                                    jme_text_file,
                                )

                    # run-dependent residual corrections
                    # (currently only for L2L3Residual)
                    if level == "L2L3Residual" and has_last_run:
                        conv = JERCToRunDependentCorrectionConverter(
                            jme_text_files,
                            first_runs=first_runs,
                            last_run=last_run,
                            name=f"{name}_{level}_{algo}",  # TODO: use labeler
                        )
                        corr = conv.make_correction(config, version=version)
                        print(f"Created run-dependent correction ({corr.name}) from files:")
                        for jme_text_file in jme_text_files:
                            print(f"  - {jme_text_file}")

                    # other levels or only one input
                    else:
                        # check if all input files for other levels are identical
                        proxies = [JERCTextFileProxy(jme_text_file) for jme_text_file in jme_text_files]
                        if not all(proxy == proxies[0] for proxy in proxies[1:]):
                            print(
                                f"WARNING: inputs for level {level} should be identical, "
                                "but text file contents differ",
                            )
                            for jme_text_file, proxy in zip(jme_text_files, proxies):
                                print(f"  - {jme_text_file} (hash: {proxy._hash})")
                            # print(f"INFO: proceeding with first file: {jme_text_files[0]}")
                            raise RuntimeError("unexpected input file content detected, aborting")

                        # proceed with first input file
                        correction_name = f"{name}_{level}_{algo}"  # TODO: use labeler
                        conv = JERCToCorrectionConverter(
                            jme_text_files[0],
                            name=f"{name}_{level}_{algo}",  # TODO: use labeler
                        )
                        corr = conv.make_correction(config, version=version)
                        print(f"Created standard correction ({corr.name}) from file:")
                        print(f"  - {jme_text_files[0]}")

                    # keep track of converters for levels
                    # (used for CompoundCorrections below)
                    level_converters[level] = conv

                    # append correction to output
                    for output in outputs:
                        corrections[output].append(corr)

                # compound correction levels
                for (output_level, input_levels) in compound_levels_map.items():
                    corr = make_compound_correction(
                        config,
                        input_converters=[
                            level_converters[input_level]
                            for input_level in input_levels
                        ],
                        name=f"{name}_{output_level}_{algo}",  # TODO: use labeler
                    )
                    input_levels_str = ", ".join(input_levels)
                    print(f"Created compound correction ({corr.name}) from input levels ({input_levels_str})")
                    for output in outputs:
                        compound_corrections[output].append(corr)

                # JEC uncertainties (MC only)
                if type_ == "jec" and name.endswith("MC"):
                    # dict with information about uncertainty sets
                    # and corresponding files (e.g. default, regrouped)
                    jec_uncertainty_sets = corr_info["uncertainty_sets"]
                    if jec_uncertainty_sets is None:
                        jec_uncertainty_sets = DEFAULT_JEC_UNCERTAINTY_SETS

                    # equally long lists of filename prefixes (e.g. "RegroupedV2_")
                    # and corresponding prefix to prepend for uncertainties
                    filename_prefixes = ["RegroupedV2_"]
                    uncertainty_prefixes = ["Regrouped_"]

                    # go through uncertainty sets
                    for jus_name, jus_info in jec_uncertainty_sets.items():
                        # skip unspecified algos
                        if "algos" in jus_info and algo not in jus_info["algos"]:
                            continue

                        # add prefixes to list
                        filename_prefixes.append(jus_info["filename_prefix"])
                        uncertainty_prefixes.append(jus_info["uncertainty_prefix"])

                    # resolve input file names
                    uncertainty_files = [
                        os.path.join(
                            config.work_dir,
                            jme_filename_from_keys(
                                name=name,
                                level="UncertaintySources",
                                algo=algo,
                                suffix=filename_suffix,
                                prefix=filename_prefix,
                            )
                        )
                        for filename_prefix in filename_prefixes
                    ]

                    # skip uncertainty-related things for algos
                    # other than AK4 PUPPI
                    # FIXME: no hard-coding
                    if algo != "AK4PFPuppi":
                        continue

                    # collect all JEC sources into one file
                    tmp_file = os.path.join(
                        config.work_dir,
                        jme_filename_from_keys(
                            name=name,
                            level="UncertaintySources",
                            algo=algo,
                            suffix="_collected",
                        )
                    )
                    compile_jme_uncertainties(
                        uncertainty_files,
                        uncertainty_prefixes,
                        output_file=tmp_file,
                    )

                    # retrieve uncertainty sources
                    uncertainty_sources = read_sections(tmp_file)
                    for uncertainty_source in uncertainty_sources:
                        print(f"Processing: {tmp_file} [{uncertainty_source}]")
                        conv = JERCToCorrectionConverter(
                            tmp_file,
                            section=uncertainty_source,
                        )
                        corr = conv.make_correction(config, version=version)
                        for output in outputs:
                            corrections[output].append(corr)

        # collect corrections into top-level `CorrectionSet` objects
        correction_sets = {}
        for output in corrections:
            cset_dict = {
                "schema_version": 2,
                "corrections": corrections[output],
            }
            if compound_corrections[output]:
                cset_dict["compound_corrections"] = compound_corrections[output]

            cset = CorrectionSet.parse_obj(cset_dict)
            correction_sets[output] = cset

        # ensure output directory exists
        output_dir = os.path.join(config.output_dir, era)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # write JSON files to output directory
        for output, cset in correction_sets.items():
            output_basename = os.path.join(
                output_dir, output
            )
            # write JSON files (plain and gzipped)
            with open(f"{output_basename}.json", "w") as fout:
                fout.write(cset.json(exclude_unset=True))
            print(f"Wrote output file: {output_basename}.json")

            with gzip.open(f"{output_basename}.json.gz", "wt") as fout:
                fout.write(cset.json(exclude_unset=True))
            print(f"Wrote output file: {output_basename}.json.gz")


def setup_create(subparsers: argparse._SubParsersAction):
    """
    Add the CLI parameters required for subcommand 'create'.
    """
    sp = subparsers.add_parser("create", help=create.__doc__)
    sp.set_defaults(command=create)

    setup_config_eras(sp)
