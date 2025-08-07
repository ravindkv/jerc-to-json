# coding: utf-8
"""
This module contains helper classes for converting JERC text files to
correctionlib JSON format.
"""
from __future__ import annotations

from typing import Union

import logging
import re

from correctionlib import schemav2 as schema
from correctionlib.schemav2 import (
    Correction,
    Binning,
    Category,
    Formula,
    FormulaRef,
    CompoundCorrection,
)
from jerc2json.labels import JERCTextFileLabeler
try:
    from pydantic_core import ValidationError
except ModuleNotFoundError:
    from pydantic import ValidationError

from jerc2json.config import ConfigDict
from jerc2json.proxy import JERCTextFileProxy
from jerc2json.util import python_float

__all__ = ["JERCToCorrectionConverter", "JERCToRunDependentCorrectionConverter"]


def make_run_bin_edges(
    iovs: list[tuple[int, int]],
    allow_discontinuous: bool = False,
):
    """
    Given a list of *iovs* (2-tuples with first and last run numbers), sort them chronologically and
    return a list of bin edges binning that can be used with correctionlib.

    Returns a tuple (*sorted_idxs*, *sorted_run_bin_edges*), where the former is a list of indices that
    would sort the *iovs* in chronological order, and the latter is a list of bin edges corresponding
    to the IOVs. The bin edges are left-inclusive, i.e. values that lie exactly on a bin edge are mapped
    to the bin having that edge as a lower bound. The final value in *sorted_run_bin_edges* is calculated
    as one plus the last run of the last IOV.

    iovs: list of tuples of (int, int)
        list of IOVs for the given tags, given as (start, end) run numbers

    allow_discontinuous: bool
        if False, an exception is raised if the IOV list
        is not continuous (i.e. the start of an IOV should be exactly 1 larger
        than the end of the previous one); if True, pseudo-IOVs will be inserted
        to cover the gaps and the list of indices returned will contain "None".
    """
    sorted_idxs = sorted(range(len(iovs)), key=iovs.__getitem__)
    sorted_iovs = [iovs[idx] for idx in sorted_idxs]

    # check that IOV sequence has no gaps
    is_discontinuous = [
        sorted_iovs[idx - 1][-1] + 1 != sorted_iovs[idx][0]
        for idx in range(1, len(iovs))
    ]

    # handle discontinuous
    if any(is_discontinuous) and not allow_discontinuous:
        raise ValueError(
            "provided IOV sequence is not continuous: "
            f"{sorted_iovs}",
        )
    else:
        for idx, is_d in list(enumerate(is_discontinuous))[::-1]:
            if is_d:
                sorted_idxs.insert(idx + 1, None)
                sorted_iovs.insert(idx + 1, (sorted_iovs[idx][1] + 1, sorted_iovs[idx + 1][0] - 1))

    # convert IOVs into binning edges (left-edge inclusive, right-edge exclusive)
    sorted_run_bin_edges = [iov[0] for iov in sorted_iovs] + [sorted_iovs[-1][-1] + 1]

    return sorted_idxs, sorted_run_bin_edges


def make_run_bin_edges_continuous(
    first_runs: list[int],
    last_run: int,
):
    """
    Like `sort_tags_make_run_bin_edges`, but create a continuous binning by construction, taking only
    the *first_runs* of each IOV and the *last_run* of the final IOV as an input.
    """
    first_runs_for_iovs = list(first_runs) + [last_run]
    iovs = [
        (first_run_1, first_run_2 - 1)
        for first_run_1, first_run_2 in zip(
            first_runs_for_iovs[:-1],
            first_runs_for_iovs[1:],
        )
    ]
    return make_run_bin_edges(iovs, allow_discontinuous=False)


class JERCToCorrectionConverter:
    """
    Class for constructing a `Correction` object for JEC/JER using a single JERC text file as an input.
    """

    # replace ROOT-specific functions in formulas
    # with generic names function names understood
    # by correctionlib
    FORMULA_REPLACEMENTS = {
        "TMath::Log": "log",
        "TMath::Max": "max",
        "TMath::Power": "pow",
    }

    def __init__(self, jme_file: str, name: str | None = None, section: str | None = None):
        self.jme_file = jme_file
        self.name = name
        self.section = section
        self.x = JERCTextFileProxy(jme_text_file=jme_file, section=section)

    def _build_lookup(self, idx_record: int):
        """
        Create special lookup nodes for JEC uncertainties and JER scale
        factors, which do not rely on a simple `Formula`.
        """
        variables = self.x.variables
        record = self.x.get_record(idx_record)
        lvl = self.x.level

        # obtain formula parameters for current record/bin
        parameters = [python_float(p) for p in record.parameters()]

        # handle special cases
        if lvl == "JECSource":
            edges = parameters[::3]  # pt
            upvar = parameters[1::3]  # upvars
            assert len(variables) == 1
            return Binning.parse_obj({
                "nodetype": "binning",
                "edges": edges,
                "input": variables[0],
                "content": [
                    # linear interpolation between bin edges
                    FormulaRef.parse_obj({
                        "nodetype": "formularef",
                        "index": "0",
                        "parameters": [
                            (
                                (upvar[idx + 1] - upvar[idx]) /
                                (edges[idx + 1] - edges[idx])
                            ),
                            (
                                (
                                    upvar[idx] * edges[idx + 1] -
                                    upvar[idx + 1] * edges[idx]
                                ) /
                                (
                                    edges[idx + 1] - edges[idx]
                                )
                            ),
                            edges[0],
                            edges[-1],
                        ],
                    })
                    for idx in range(len(edges) - 1)
                ],
                "flow": "clamp",
            })

        elif lvl in ("ScaleFactor", "SF"):
            # only for binned SFs for now (no parametrization, just nom/up/down)
            assert len(parameters) == 3
            return Category.parse_obj({
                "nodetype": "category",
                "input": "systematic",
                "content": [
                    {
                        "key": "nom",
                        "value": float(parameters[0]),
                    },
                    {
                        "key": "up",
                        "value": float(parameters[2]),
                    },
                    {
                        "key": "down",
                        "value": float(parameters[1]),
                    },
                ],
            })

    def _build_formula(self, idx_record: int):
        formula = self.x.formula

        # handle special cases where text file does not contain a formula
        # (e.g. for JER scale factors or JEC uncertainties)
        if formula == '""' or formula == "None":
            return self._build_lookup(idx_record)

        # replace ROOT function names with correctionlib ones
        for root_func, json_func in self.FORMULA_REPLACEMENTS.items():
            formula = formula.replace(root_func, json_func)

        # obtain the formula parameters for the record
        parameters = [
            python_float(p)
            for p in self.x.get_record(idx_record).parameters()
        ]

        # first 2*nParVar parameters are for clamping,
        # so we replace them with the proper indices
        n_pars = self.x.n_variables
        n_pars_clamp = 2 * n_pars
        parameters_no_clamp = parameters[n_pars_clamp:]
        for i in reversed(range(0, len(parameters_no_clamp))):
            formula = formula.replace(
                f"[{i}]",
                f"[{n_pars_clamp + i}]",
            )

        var_aliases = ["x", "y", "z", "t"]
        for i in range(n_pars):
            to_replace = f"(?<![A-z]){var_aliases[i]}(?![A-z])"
            replace_with = f"max(min({var_aliases[i]},[{2*i + 1}]),[{2*i}])"
            formula = re.sub(to_replace, replace_with, formula)
            # logging.debug(formula)

        # clamping not implemented, yet. Clamping of observables defined in first 2*nParVar parameters
        # TODO: don't understand this comment; clamping is implemented above, no?
        logging.debug(self.x.variables)

        # return formula object
        return Formula.parse_obj({
            "nodetype": "formula",
            "expression": formula,
            "parser": "TFormula",
            "parameters": parameters,
            "variables": self.x.variables,
        })

    def _get_edges(self, idx_binvar: int, idx_record: int):
        # check if binning variable index is valid
        if idx_binvar >= self.x.n_binning_variables:
            raise ValueError(
                f"invalid binning variable index {idx_binvar} for "
                f"JERC table with {self.x.n_binning_variables} binning "
                "variables"
            )

        # find neighbouring bins in the current variable.  The previous
        # implementation relied on `neighbourBin` to walk along the binning
        # axis.  Some JERC text files are not strictly continuous (e.g. there is
        # a gap at `eta=0` in the example in the issue), in which case
        # `neighbourBin` stops at the gap and the positive part of the range is
        # never visited.  To make the bin edge collection robust we inspect all
        # records and select those that correspond to the same configuration of
        # the *other* binning variables as the reference record.  The union of
        # their lower and upper edges gives the complete binning range.

        # collect the min/max values for the remaining binning variables of the
        # reference record
        ref_record = self.x.get_record(idx_record)
        other_binvars = [i for i in range(self.x.n_binning_variables) if i != idx_binvar]
        ref_other_ranges = [
            (
                python_float(ref_record.xMin(i)),
                python_float(ref_record.xMax(i)),
            )
            for i in other_binvars
        ]

        # iterate over all records to find those matching the reference binning
        edges_set: set[float] = set()
        bin_map: dict[tuple[float, float], int] = {}
        n_records = self.x.table.size()
        for idx in range(n_records):
            rec = self.x.get_record(idx)

            # keep only bins where all other binning variables match the
            # reference record
            matches = True
            for (j, (ref_min, ref_max)) in zip(other_binvars, ref_other_ranges):
                if (
                    python_float(rec.xMin(j)) != ref_min
                    or python_float(rec.xMax(j)) != ref_max
                ):
                    matches = False
                    break
            if not matches:
                continue

            xmin = python_float(rec.xMin(idx_binvar))
            xmax = python_float(rec.xMax(idx_binvar))
            edges_set.add(xmin)
            edges_set.add(xmax)
            bin_map[(xmin, xmax)] = idx

        edges = sorted(edges_set)
        bin_record_idxs = [
            bin_map[(edges[i], edges[i + 1])] for i in range(len(edges) - 1)
        ]

        return edges, bin_record_idxs

    def _recurse_binning(self, idx_binvar: int, idx_record: int):
        defs = self.x.definitions

        # if we reached the final variable, build and return a `Formula` node
        if idx_binvar >= self.x.n_binning_variables:
            return self._build_formula(idx_record)

        # otherwise, return a `Binning` object and call this function recursively
        logging.debug(
            "start recursing binvar "
            f"{self.x.get_binning_variable(idx_binvar)}"
        )

        edges, idxs_neighboring_bins = self._get_edges(idx_binvar, idx_record)

        corr_dict = {
            "nodetype": "binning",
            "edges": edges,
            "input": self.x.get_binning_variable(idx_binvar),
            "content": [
                self._recurse_binning(
                    idx_binvar + 1,
                    idx_neighboring_bin,
                )
                for idx_neighboring_bin in idxs_neighboring_bins
            ],
            # TODO: use "flow": "clamp" regardless?
            # "flow": "clamp", #JEC need a different kind of clamp (in formula)
            # "flow": "error",
            "flow": 1.0 if defs.level() != "JECSource" else -999.0,
        }

        try:
            rval = Binning.parse_obj(corr_dict)
        except ValidationError:
            obj_repr = "\n".join(f"{k}: {v!r}," for k, v in corr_dict.items())
            print(f"[ERROR] ValidationError encountered while parsing object:\n{obj_repr}")
            raise

        return rval

    def get_data(self):
        """
        Obtain raw correction data for this correction.
        """
        # recurse through binning variables to obtain correction structure
        return self._recurse_binning(
            idx_binvar=0,
            idx_record=0,
        )

    def get_inputs(self, config: ConfigDict):
        return [
            {
                "name": input_var,
                "type": config.input_variables[input_var]["type"],
                "description": config.input_variables[input_var]["description"],
            }
            for input_var in self.x.all_variables
        ]

    def make_correction(self, config: ConfigDict, version: int):
        """
        Construct a `Correction` object corresponding to an individual correction level.
        """
        # get raw correction data
        corr_data = self.get_data()

        labeler = JERCTextFileLabeler([self.x])
        generic_formulas = []
        inputs = self.get_inputs(config)

        # additional `systematic` input var for JER scale factors
        if self.x.level in ("ScaleFactor", "SF"):
            inputs.append({
                "name": "systematic",
                "type": "string",
                "description": "systematics: nom, up, down",
            })

        # generic formulas for everything except JER scale factors
        if self.x.level not in ("ScaleFactor", "SF"):
            generic_formulas.append(
                schema.Formula(
                    nodetype="formula",
                    expression="[0]*max(min(x,[3]),[2])+[1]",
                    parser="TFormula",
                    variables=["JetPt"],
                ),
            )

        # build and return the `Correction` object
        corr_obj = Correction.parse_obj({
            "version": version,
            "name": self.name or labeler.get("correction_name"),
            "description": labeler.get("description"),
            "inputs": inputs,
            "generic_formulas": generic_formulas,
            "output": {
                "name": "correction",
                "type": "real",
            },
            "data": corr_data,
        })

        return corr_obj


class JERCToRunDependentCorrectionConverter:
    """
    Class for constructing a `Correction` object for run-dependent JEC/JER. This type of correction consists of
    a series of regular corrections, each of which are valid in a specific interval of validity (IOV), as indicated
    by a range of runs. The *jme_files* for each IOV are passed to the constructor along with the *first_run* for
    which they are valid, as well as an overall *last_run*. Note that unlike the regular corrections, where the name
    can be inferred from the file, a *name* must be explicitly provided here.
    """
    def __init__(
        self,
        jme_files: list[str],
        first_runs: list[int],
        last_run: int,
        name: str,
        section: str | None = None,
    ):
        self.name = name
        self.section = section

        # validate arguments
        if not jme_files:
            raise ValueError("need at least one JERC file")
        if len(jme_files) != len(first_runs):
            raise ValueError(
                f"got {len(first_runs)} first_runs, "
                f"expected same as jme_files ({len(jme_files)})",
            )

        # compute binning for run-based lookup, sorting
        # jme_files in chronological order
        sorted_idxs, sorted_run_bin_edges = make_run_bin_edges_continuous(
            first_runs=first_runs,
            last_run=last_run,
        )
        self.jme_files = [jme_files[idx] for idx in sorted_idxs]
        self.run_bin_edges = sorted_run_bin_edges

        # create a converter object for each file
        self.subconverters = [
            JERCToCorrectionConverter(jme_file, section)
            for jme_file in self.jme_files
        ]

        # use first file proxy as a reference
        self.x = self.subconverters[0].x

        # run post-init checks
        self._check_subconverters_compatible_raise()

    def _check_subconverters_compatible_raise(self):
        """check that subconverters have the same input variables, levels"""
        assert self.subconverters, "internal error: no subconverters found"

        # check identical variables
        # TODO: loosen this check?
        all_variables = self.subconverters[0].x.all_variables
        if not all(
            subconverter.x.all_variables == all_variables
            for subconverter in self.subconverters
        ):
            jme_files_str = ", ".join(self.jme_files)
            raise ValueError(f"incompatible files (different input variables): {jme_files_str}")

        # check identical levels
        level = self.subconverters[0].x.level
        if not all(
            subconverter.x.level == level
            for subconverter in self.subconverters
        ):
            jme_files_str = ", ".join(self.jme_files)
            raise ValueError(f"incompatible files (different levels): {jme_files_str}")

    def get_data(self):
        """
        Obtain raw correction data for this run-dependent correction.
        """
        # get subcorrections data
        subcorrections_data = [
            subconverter.get_data()
            for subconverter in self.subconverters
        ]

        # wrap with run binning
        corr_data = {
            "nodetype": "binning",
            # note: add one to max to make integer binning inclusive
            "edges": self.run_bin_edges,
            "input": "run",
            "content": subcorrections_data,
            "flow": "error",
        }

        # return correction data
        return corr_data

    def get_inputs(self, config: ConfigDict):
        inputs = [
            {
                "name": input_var,
                "type": config.input_variables[input_var]["type"],
                "description": config.input_variables[input_var]["description"],
            }
            for input_var in ["run"] + self.subconverters[0].x.all_variables
        ]

        # additional `systematic` input var for JER scale factors
        if self.subconverters[0].x.level in ("ScaleFactor", "SF"):
            inputs.append({
                "name": "systematic",
                "type": "string",
                "description": "systematics: nom, up, down",
            })

        return inputs

    def make_correction(self, config: ConfigDict, version: int):
        """
        Construct a `Correction` object that combines whose top node performs a run-based
        lookup to find the correct IOV before evaluating the corresponding JEC/JER
        themselves on the .
        """
        generic_formulas = []
        inputs = self.get_inputs(config)

        # generic formulas for everything except JER scale factors
        if self.subconverters[0].x.level not in ("ScaleFactor", "SF"):
            generic_formulas.append(
                schema.Formula(
                    nodetype="formula",
                    expression="[0]*max(min(x,[3]),[2])+[1]",
                    parser="TFormula",
                    variables=["JetPt"],
                ),
            )

        # build and return the `Correction` object
        corr_data = self.get_data()

        # build and return the `Correction` object
        labeler = JERCTextFileLabeler([
            subconverter.x for subconverter in self.subconverters
        ])
        corr_obj = Correction.parse_obj({
            "version": version,
            "name": self.name,
            "description": labeler.get("description"),
            "inputs": inputs,
            "generic_formulas": generic_formulas,
            "output": {
                "name": "correction",
                "type": "real",
            },
            "data": corr_data,
        })

        return corr_obj


def make_compound_correction(
    config: ConfigDict,
    input_converters: list[Union[JERCToCorrectionConverter, JERCToRunDependentCorrectionConverter]],
    name: str,
):
    """
    Helper function for constructing a `CompoundCorrection` object for JEC/JER, which is designed to apply multiple
    correction levels one after the other. The converter objects passed to this method should correspond to the
    individual correction levels.
    """

    input_vars = sorted(list(set.union(*[
        set(input_converter.x.all_variables)
        for input_converter in input_converters
    ])))

    labelers = [
        JERCTextFileLabeler([input_converter.x])
        for input_converter in input_converters
    ]

    # extract metadata keysd and values,
    # ensuring inputs have the same "name", "data_type" and "algo"
    metadata = {}
    metadata_keys = ("name", "data_type", "algo")
    for mk in metadata_keys:
        values = {
            v for labeler in labelers
            if (v := labeler.get(mk, None)) is not None
        }
        if len(values) != 1:
            values_str = ", ".join(sorted(list(values)))
            raise ValueError(
                "compound correction got unexpected combination of "
                f"'{mk}': {values_str}"
            )
        metadata[mk] = list(values)[0]

    data_type = metadata["data_type"]
    algo = metadata["algo"]

    # merge inputs
    input_variables = {}
    for input_converter in input_converters:
        for input_var in input_converter.get_inputs(config):
            input_variables[input_var["name"]] = input_var

    # create compound correction
    compound_corr = CompoundCorrection.parse_obj({
        "name": name,
        "description": (
            f"compound correction for {algo} created from {name} ({data_type}) "
            "by using https://gitlab.cern.ch/cms-jetmet/jerc2json"
        ),
        "inputs": list(input_variables.values()),
        # update the jet pt after each step of the compound correction
        "inputs_update": ["JetPt"],
        "input_op": "*",
        "output_op": "*",
        "output": {
            "name": "correction",
            "type": "real",
        },
        "stack": [
            input_converter.name
            for input_converter in input_converters
        ],
    })

    return compound_corr
