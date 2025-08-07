# coding: utf-8
"""
This module contains helper classes for evaluating various types
of corrections from either the lookup tables contained in JERC text
files, or from correctionlib JSON files, with a unified interface
and built-in vectorization.
"""
from __future__ import annotations

import abc
import numpy as np
import logging
import os

from functools import partial

from jerc2json.proxy import JSONProxy

__all__ = ["EvaluatorBase", "CMSSWEvaluator", "JSONEvaluator"]


class EvaluatorBase(metaclass=abc.ABCMeta):
    """
    Base class for evaluators.
    """

    @abc.abstractmethod
    def evaluate_vectorized(self, input_vars, input_arrays, ignore_unknown_vars=False):
        """
        Method to be implemented by all evaluators.
        """
        pass


class CMSSWEvaluator(EvaluatorBase):
    """
    Helper class for evaluating various types of corrections
    and other lookup values contained in JERC text files.
    """

    VALID_MODES = ("jec", "jer", "jer_sf", "jec_uncertainty")
    N_ARGS = {
        "jec": 0,
        "jer": 0,
        "jer_sf": 1,
        "jec_uncertainty": (1, 2),
    }

    def __init__(self, jme_text_files: list[str], args: list | None = None, *, mode: str = "jec"):
        self.txt_files = jme_text_files
        self.args = args or []
        self.mode = mode

        # validate inputs
        if mode not in self.VALID_MODES:
            valid_modes_str = ", ".join(self.VALID_MODES)
            raise ValueError(f"invalid mode '{mode}'; available: {valid_modes_str}")

        if mode in ("jer_sf", "jer", "jec_uncertainty") and len(self.txt_files) != 1:
            raise ValueError(
                "must have exactly one JERC text file for mode "
                f"'{mode}', got {len(self.txt_files)}",
            )

        missing_files = {
            fname
            for fname in self.txt_files
            if not os.path.exists(fname)
        }
        if missing_files:
            missing_files_str = "\n  - ".join(missing_files)
            raise ValueError(
                f"the following input files were not found:\n  - {missing_files_str}"
            )

        n_args_expected = self.N_ARGS[self.mode]
        n_args_expect_range = isinstance(n_args_expected, tuple)
        if n_args_expect_range and (
            (len(self.args) < n_args_expected[0]) or
            (len(self.args) > n_args_expected[1])
        ):
            raise ValueError(
                f"expected between {n_args_expected[0]} and {n_args_expected[1]} "
                f"argument(s) for mode '{mode}', got {len(self.args)}",
            )
        elif (not n_args_expect_range) and (
            len(self.args) != n_args_expected
        ):
            raise ValueError(
                f"expected exactly {n_args_expected} argument(s) for mode "
                f"'{mode}', got {len(self.args)}",
            )

        self._evaluator = self._make_evaluator()

    def _make_evaluator(self):
        """
        Return an object with an `evaluate` method that can be called
        to obtain the correction/lookup value for a set of inputs.

        NOTE: Depends on ROOT and nanoAOD-tools.
        """
        def _jcps_check_is_flow(jcps):
            """
            Given a list of `JetCorrectorParameter` objects, check if
            the current input variables set on the evaluator object
            `self._evaluator` are inside the ranges defined in the
            input file.
            Returns True if (a) a bin is found corresponding to the
            value of the binning variables and (b) the value of the
            parametrization variables is inside the allowed range
            for that bin.
            """
            inp_var_map = getattr(self._evaluator, "_py_input_variable_values", None)
            if not inp_var_map:
                print(
                    "[WARNING] is_flow() only works after calling evaluate(), returning False",
                )
                return False

            for jcp in jcps:
                d = jcp.definitions()

                # names of binning and parametrization variables
                bin_vars = list(d.binVar())
                par_vars = list(d.parVar())

                # values of binning and parametrization variables
                bin_var_values = [inp_var_map[v] for v in bin_vars]
                par_var_values = [inp_var_map[v] for v in par_vars]

                # check if overflow of binVar (bin index is negative)
                bin_index = jcp.binIndex(bin_var_values)
                if bin_index < 0:
                    return True

                # check if overflow in parVar (outside clamp brackets passed
                # as parameters with index (2*i, 2*i + 1) for i-th parVar
                record = jcp.record(bin_index)
                parameters = list(record.parameters())
                for i, par_var_value in enumerate(par_var_values):
                    par_min = parameters[2*i]
                    par_max = parameters[2*i + 1]
                    if not (par_min <= par_var_value <= par_max):
                        return True

                # no overflow encountered
                return False

        import ROOT
        if self.mode == "jec":
            # prepare vector of JetCorrectorParameters
            jcp_vector = ROOT.vector(ROOT.JetCorrectorParameters)()
            for txt_file in self.txt_files:
                jcp_vector.push_back(
                    ROOT.JetCorrectorParameters(txt_file),
                )

            # construct evaluator
            evaluator = ROOT.FactorizedJetCorrector(jcp_vector)
            evaluator.evaluate = evaluator.getCorrection
            evaluator.is_flow = partial(_jcps_check_is_flow, jcps=jcp_vector)

        elif self.mode == "jer":
            params_wrapper = ROOT.PyJetParametersWrapper()
            jer_evaluator = ROOT.PyJetResolutionWrapper(
                self.txt_files[0],
            )
            def _eval():
                return jer_evaluator.getResolution(params_wrapper)

            # construct evaluator
            evaluator = params_wrapper
            evaluator.evaluate = _eval
            # FIXME: implement is_flow correctly for mode 'jer'
            evaluator.is_flow = (lambda: False)

        elif self.mode == "jer_sf":
            params_wrapper = ROOT.PyJetParametersWrapper()
            jersf_evaluator = ROOT.PyJetResolutionScaleFactorWrapper(
                self.txt_files[0]
            )
            syst = self.args[0]
            syst_map = {
                "up": 2,
                "down": 1,
                "nom": 0,
            }
            syst_idx = syst_map.get(syst, None)
            if syst_idx is None:
                valid_syst_str = ", ".join(syst_map)
                raise ValueError(f"invalid systematic variation for JER SF '{syst}'; valid: {valid_syst_str}")

            def _eval(syst_idx=syst_idx):
                return jersf_evaluator.getScaleFactor(params_wrapper, syst_idx)

            # construct evaluator
            evaluator = params_wrapper
            evaluator.evaluate = _eval
            # FIXME: implement is_flow correctly for mode 'jer_sf'
            evaluator.is_flow = (lambda: False)

        elif self.mode == "jec_uncertainty":
            jcp = ROOT.JetCorrectorParameters(self.txt_files[0], self.args[0])
            evaluator = ROOT.JetCorrectionUncertainty(jcp)

            # resolve uncertainty direction
            unc_bool = True
            if len(self.args) > 1:
                unc_dir = self.args[1]
                unc_dict = {
                    "up": True,
                    "down": False,
                }
                unc_bool = unc_dict.get(unc_dir, None)
                if unc_bool is None:
                    valid_unc_dir_str = ", ".join(unc_dict)
                    raise ValueError(
                        f"invalid systematic variation for JEC uncertainty '{unc_dir}'; valid: {valid_unc_dir_str}",
                    )

            def _eval(unc_bool=unc_bool):
                return evaluator.getUncertainty(unc_bool)

            # construct evaluator
            evaluator.evaluate = _eval
            evaluator.is_flow = partial(_jcps_check_is_flow, jcps=[jcp])

        else:
            raise ValueError(r"invalid mode '{mode}'")

        # sanity checks before returning
        assert hasattr(evaluator, "evaluate")
        assert hasattr(evaluator, "is_flow")
        evaluator._py_input_variable_values = {}

        # return evaluator
        return evaluator

    def evaluate_vectorized(self, input_vars: list, input_arrays: list, ignore_unknown_vars: bool = False, mask_flow: bool = False):
        """
        Evaluate the correction over arbitrary multidimensional `input_arrays` in a vectorized way.
        If `mask_flow` is true, the output array will contain `np.nan` entries in cases where
        """
        logging.info(
            "Evaluating CMSSW correction: mode=%s, args=%r, input_vars=%r",
            self.mode, self.args, input_vars,
        )
        # print("Evaluating CMSSW correction: mode=%s, args=%r, input_vars=%r" % (self._mode, self._args, input_vars))

        if len(input_vars) != len(input_arrays):
            raise ValueError(
                f"number of input vars ({len(input_vars)}) "
                f"!= number of input arrays ({len(input_arrays)})"
            )

        @np.vectorize
        def get_correction(*input_var_values):
            for inp_var, inp_var_val in zip(input_vars, input_var_values):
                try:
                    inp_var_val = float(inp_var_val)
                except:
                    raise ValueError(f"{inp_var = }, ({type(inp_var_val)}) {inp_var_val = }")
                setter_method = getattr(self._evaluator, f"set{inp_var}")
                if setter_method is not None and not ignore_unknown_vars:
                    setter_method(inp_var_val)
                    # add the value to a Python dictionary for easy retrieval
                    # (C++ API does not provide getter methods)
                    self._evaluator._py_input_variable_values[inp_var] = inp_var_val

            value = self._evaluator.evaluate()
            valid = (not mask_flow) or (not self._evaluator.is_flow())

            return value if valid else np.nan

        correction_values = get_correction(*input_arrays)
        return correction_values


class JSONEvaluator(EvaluatorBase):
    """
    Helper class for evaluating various types of corrections
    and other lookup values contained in correctionlib JSON files.

    """

    def __init__(self, json_file: str, correction_name: str, args: list | None = None, *, mode: str = "jec"):
        self._proxy = JSONProxy(json_file)  # load JSON file with caching
        self.mode = mode
        self.args = args
        self.correction_name = correction_name

        corr_set = self._proxy.correction_set
        evaluator = self._proxy.evaluator

        # check for name collisions between simple and compound corrections
        name_collisions = set(
            self._proxy.evaluator.keys()
        ).intersection(
            self._proxy.evaluator.compound.keys()
        )
        if name_collisions:
            name_collisions_str = "\n  -".join(sorted(name_collisions))
            print(
                f"WARNING: the following keys are defined both as simple "
                f"and compound corrections:\n  -{name_collisions_str}"
            )

        # dict for easy lookup of correction by name
        self._corr_dict = {
            c.name: c
            for c in corr_set.corrections + (corr_set.compound_corrections or [])
        }

        # load correction from `CorrectionSet`
        self.corr_obj = self._corr_dict.get(
            self.correction_name,
            None,
        )
        if self.corr_obj is None:
            corrections_available_str = "\n  - ".join(sorted(self._corr_dict))
            raise ValueError(
                f"correction '{self.correction_name}' not "
                f"found in JSON {json_file}; available:"
                f"\n  - {corrections_available_str}",
            )

        # input variables
        self._input_vars = [
            inp.name
            for inp in self.corr_obj.inputs
        ]

        # resolve the actual evaluation function
        try:
            self._eval_func = evaluator.compound[self.correction_name].evaluate
        except KeyError:
            # not found under compound corrections,
            self._eval_func = evaluator[self.correction_name].evaluate

        # TODO: handle jer_sf

    @property
    def input_vars(self):
        """Input variables required by the correction."""
        return self._input_vars

    def evaluate_vectorized(self, input_vars: list, input_arrays: list, ignore_unknown_vars: bool = False):
        """
        Evaluate the correction over arbitrary multidimensional `input_arrays` in a vectorized way.
        """
        logging.info(
            "Evaluating JSON correction: mode=%s, args=%r, input_vars=%r",
            self.mode, self.args, input_vars,
        )
        # print("Evaluating JSON correction: mode=%s, args=%r, input_vars=%r" % (self._mode, self._args, input_vars))

        if len(input_vars) != len(input_arrays):
            raise ValueError(
                f"number of input vars ({len(input_vars)}) "
                f"!= number of input arrays ({len(input_arrays)})"
            )

        missing_vars = set(self._input_vars) - set(input_vars)
        if missing_vars:
            missing_vars_str = ", ".join(sorted(missing_vars))
            raise ValueError(f"missing required input variables: {missing_vars_str}")

        unknown_vars = set(input_vars) - set(self._input_vars)
        if unknown_vars and not ignore_unknown_vars:
            unknown_vars_str = ", ".join(sorted(unknown_vars))
            raise ValueError(f"unknown input variables: {unknown_vars_str}")

        # reorder input variables/arrays so they correspond to the
        # order defined in the JSON file
        idxs = [input_vars.index(inp_var) for inp_var in self._input_vars]
        input_vars = [input_vars[idx] for idx in idxs]
        input_arrays = [
            np.asarray(input_arrays[idx])
            if not isinstance(input_arrays[idx], str)
            else input_arrays[idx]
            for idx in idxs
        ]

        # evaluate the correction on the grid
        correction_values = self._eval_func(*input_arrays)

        # return the correction value array
        return correction_values
