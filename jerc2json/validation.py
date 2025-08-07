# coding: utf-8
"""
Various tools for validating correctionlib JSON files
against the source JERC text files.
"""
from __future__ import annotations

import numpy as np
import logging
import os

from correctionlib.schemav2 import Correction, Formula, Binning

from jerc2json.config import ConfigDict
from jerc2json.evaluation import CMSSWEvaluator, JSONEvaluator
from jerc2json.util import extend_edges, refine_edges

__this_dir__ = os.path.dirname(__file__)
# TODO: logger

STAT_FUNCS = {
    # usual stats
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "count": np.size,
    # stats not considering NaN values
    "nanmin": np.nanmin,
    "nanmax": np.nanmax,
    "nanmean": np.nanmean,
    "nanmedian": np.nanmedian,
    "nanstd": np.nanstd,
    "count_nonfinite": lambda arr: np.sum(~np.isfinite(arr)),
}


def make_json_correction_test_grid(corr_obj: Correction, refine_factor: int = 2):
    logging.debug("Creating test grid from correctionlib Correction object: %r", corr_obj)
    edge_map = {}

    def fill_edge_map(node):
        # for binning nodes, fill the binning directly
        if isinstance(node, Binning):
            edge_map[node.input] = node.edges
            for child_node in node.content:
                fill_edge_map(child_node)

        # for formula nodes, assume parameters
        # 2*i and 2*i+1 are used for clamping and
        # indicate the variable range, then
        # compute an equidistant binning
        elif isinstance(node, Formula):
            for i, var in enumerate(node.variables):
                edge_map[var] = np.linspace(
                    node.parameters[2 * i],
                    node.parameters[2 * i + 1],
                    num=10,
                )

    fill_edge_map(corr_obj.data)

    for var in list(edge_map):
        edge_map[var] = extend_edges(
            refine_edges(
                edge_map[var],
                factor=refine_factor,
            )
        )

    # construct a grid of input variables
    input_vars = [inp.name for inp in corr_obj.inputs]
    input_arrays = np.meshgrid(*[
        edge_map[input_var]
        for input_var in input_vars
    ])

    return input_vars, edge_map, input_arrays


def make_test_values_from_correction(corr_obj: Correction, points_per_bin: int = 1):
    """
    Create arrays of test values for input variables directly from the information in a JSON correction object.

    By default, the test values will always contain the bin edges. In addition, a
    number of equidistant points (`points_per_bin`) is sampled within each bin, including
    over/underflows (default: 1).

    Returns a dictionary mapping the variable names to the corresponding test values.
    """
    edge_map = {}

    def fill_edge_map(node):
        # for binning nodes, fill the binning directly
        if isinstance(node, Binning):
            edge_map[node.input] = node.edges
            for child_node in node.content:
                fill_edge_map(child_node)

        # for formula nodes, assume parameters
        # 2*i and 2*i+1 are used for clamping and
        # indicate the variable range, then
        # compute an equidistant binning
        elif isinstance(node, Formula):
            for i, var in enumerate(node.variables):
                edge_map[var] = np.linspace(
                    node.parameters[2 * i],
                    node.parameters[2 * i + 1],
                    num=10,  # TODO: set to 1 or `points_per_bin`?
                )

    fill_edge_map(corr_obj.data)

    for var in list(edge_map):
        edge_map[var] = extend_edges(
            refine_edges(
                edge_map[var],
                factor=points_per_bin + 1,
            )
        )

    return edge_map


def validate_json_jme(
    config: ConfigDict,
    json_file: str,
    json_correction_name: str,
    jme_text_files: list[str],
    jme_mode: str,
    jme_args: list | None = None,
    values_from="config",
    quiet=True,
    full_output=False,
    override_test_values: dict | None = None,
):
    jme_args = jme_args or []

    j = JSONEvaluator(json_file, json_correction_name)
    c = CMSSWEvaluator(jme_text_files, args=jme_args, mode=jme_mode)

    if values_from == "config":
        # retrieve test values from configuration
        # (consider only real-valued variables)
        test_values = {
            iv: config.input_variables[iv].get("test_values", None)
            for iv in j.input_vars
            if config.input_variables[iv].type == "real"
        }

    elif values_from == "grid":
        # compute test values from `Correction` object
        # (consider only real-valued variables)
        test_values = make_test_values_from_correction(
            j.corr_obj,
            points_per_bin=1,  # TODO: make configurable?
        )
    else:
        raise ValueError(
            f"invalid value '{values_from}' for 'values_from'; "
            "expected one of: config, grid",
        )

    # override test values manually if requested
    if override_test_values:
        test_values.update(override_test_values)

    # check for missing test values
    if None in test_values.values():
        missing_test_values_str = ", ".join(
            iv for iv, iv_val in test_values.items()
            if iv_val is None
        )
        raise ValueError(
            f"missing test values for variables: {missing_test_values_str}"
        )

    # compute mesh grid from cartesian product of test values
    test_values_grid = {
        iv: vals
        for iv, vals in zip(
            test_values,
            np.meshgrid(*test_values.values())
        )
    }

    c_input_vars = [v for v in j.input_vars if v in test_values_grid]
    c_input_arrays = [test_values_grid[v] for v in c_input_vars]

    # special case for scale factors: JSON can be passed
    # "systematic" as a variable, but CMSSW evaluator needs
    # to be passed the systematic variation via the `args`;
    # we append this to the JSON input args only
    if jme_mode == "jer_sf":
        assert len(jme_args) == 1, \
            "internal error: jer_sf validation, but systematic not passed in jme_args"
        j_input_vars = c_input_vars + ["systematic"]
        j_input_arrays = c_input_arrays + [jme_args[0]]
    else:
        j_input_vars = c_input_vars.copy()
        j_input_arrays = c_input_arrays.copy()

    # remove JSON-only inputs like 'run' from CMSSW evaluator
    if "run" in c_input_vars:
        idx_run = c_input_vars.index("run")
        c_input_vars.pop(idx_run)
        c_input_arrays.pop(idx_run)

    results = {}
    results["json"] = j.evaluate_vectorized(j_input_vars, j_input_arrays)
    results["cmssw"] = c.evaluate_vectorized(c_input_vars, c_input_arrays)

    jme_text_files_str = ", ".join(jme_text_files)

    j_inputs_str = "\n  - ".join(
        f"{k+':':12s} {v}"
        for k, v in sorted(zip(j_input_vars, j_input_arrays))
    )

    if not quiet:
        print(f"\nInputs:\n  - {j_inputs_str}")
        print("Outputs:")
        print(f"  CMSSW: {jme_mode} @ {jme_text_files_str}")
        print(f"  -> {results['cmssw']}")
        print(f"  JSON:  {json_correction_name} @ {json_file}")
        print(f"  -> {results['json']}")

    # evaluate the difference in the obtained corrections
    results["diff"] = (results["json"] - results["cmssw"])
    results["ratio"] = (results["json"] / results["cmssw"])

    # consider zero difference if both evaluators give a
    # nonfinite result
    mask = (~np.isfinite(results["json"])) & (~np.isfinite(results["cmssw"]))
    results["diff"][mask] = 0
    results["ratio"][mask] = 1

    data = {
        "input_vars": [
            {
                "name": input_var,
                "values": list(map(float, input_values)),
            }
            for input_var, input_values in test_values.items()
        ],
        "correction_values_stats": {
            result_name: {
                stat_name: float(stat_func(result_array))
                for stat_name, stat_func in STAT_FUNCS.items()
            }
            for result_name, result_array in results.items()
        },
    }

    # output full arrays of correction values, if requested
    # NOTE: this may drastically increases the size of the output YAML files
    if full_output:
        data["correction_values"] = {
            result_name: result_array.tolist()
            for result_name, result_array in results.items()
        }

    return data
