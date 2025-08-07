# coding: utf-8
"""
Various utility functions.
"""
from __future__ import annotations

import numpy as np
import logging
import os

from correctionlib.schemav2 import Correction


__this_dir__ = os.path.dirname(__file__)
# TODO: logger


def python_float(c_float):
    """Convert a C float to a Python float"""
    # TODO: better implementation?
    return float(str(np.single(c_float)))


def jme_filename_from_keys(name: str, level: str, algo: str, prefix: str | None = None, suffix: str | None = None):
    """
    Small helper to construct the JERC text file name following the
    standard conventions.
    """
    prefix = prefix or ""
    suffix = suffix or ""
    return f"{name}/{prefix}{name}_{level}_{algo}{suffix}.txt"


# TODO
def _make_generic_jer_smearing_correction():
    # TODO: this still needs to account for the |gen_pt - pt| < 3 sigma criterion
    # including JER-smearing snippet implementation by Nick to enable JER-smearing from within correctionlib:
    # https://github.com/cms-nanoAOD/correctionlib/issues/130
    jer_smear = Correction.parse_obj({
        "version": 1,
        "name": "JERSmear",
        "description": "Jet smearing tool",
        "inputs": [
            {
                "name": "JetPt",
                "type": "real",
            },
            {
                "name": "JetEta",
                "type": "real",
            },
            {
                "name": "GenPt",
                "type": "real",
                "description": "matched GenJet pt, or -1 if no match",
            },
            {
                "name": "Rho",
                "type": "real",
                "description": "entropy source",
            },
            {
                "name": "EventID",
                "type": "int",
                "description": "entropy source",
            },
            {
                "name": "JER",
                "type": "real",
                "description": "Jet energy resolution",
            },
            {
                "name": "JERSF",
                "type": "real",
                "description": "Jet energy resolution scale factor",
            },
        ],
        "output": {
            "name": "smear",
            "type": "real",
        },
        "data": {
            "nodetype": "binning",
            "input": "GenPt",
            "edges": [-1, 0, 1],
            "flow": "clamp",
            "content": [
                # stochastic (used in GenPt < 0, i.e. undefined)
                {
                    # rewrite GenPt with a random gaussian
                    "nodetype": "transform",
                    "input": "GenPt",
                    "rule": {
                        "nodetype": "hashprng",
                        "inputs": ["JetPt", "JetEta", "Rho", "EventID"],
                        "distribution": "normal",
                    },
                    "content": {
                        "nodetype": "formula",
                        # TODO min jet pt?
                        "expression": "1 + sqrt(max(x*x - 1, 0)) * y * z",
                        "parser": "TFormula",
                        # here GenPt is actually the output of hashprng,
                        # i.e. a normally distributed random value
                        "variables": ["JERSF", "JER", "GenPt"],
                    },
                },
                # deterministic (used if GenPt > 0, i.e. there is a matched gen jet)
                {
                    "nodetype": "formula",
                    # TODO min jet pt?
                    "expression": "1 + (x-1)*(y-z)/y",
                    "parser": "TFormula",
                    "variables": ["JERSF", "JetPt", "GenPt"],
                },
            ],
        },
    })

    logging.debug(jer_smear)
    return jer_smear


def extend_edges(edges: list[float] | np.ndarray):
    """
    Extend a list of floats (`edges`) by appending single values at both ends.

    The new values are chosen so that the difference between the first and second
    entries from both ends remains the same before and after the extension.
    """
    if len(edges) < 2:
        return edges
    new_low = edges[0] - (edges[1] - edges[0])
    new_high = edges[-1] + (edges[-1] - edges[-2])
    new_edges = list(edges)
    new_edges.insert(0, new_low)
    new_edges.append(new_high)
    return new_edges


def refine_edges(edges: list[float] | np.ndarray, factor: float):
    """
    Refine a binning by splitting each bin into `factor` bins.
    `factor` should be an integer larger than or equal to 2.
    """

    factor = int(factor)
    if factor < 1:
        return edges

    edges = np.asarray(edges)
    new_widths = (edges[1:] - edges[:-1]) / factor

    new_edges = []
    for i in range(factor):
        new_edges.append(
            edges[:-1] + (i / factor) * new_widths
        )

    new_edges = list(
        np.array(new_edges).T.reshape(-1)
    ) + [edges[-1]]

    return np.asarray(new_edges)
