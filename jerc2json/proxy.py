# coding: utf-8
"""
This module contains helper classes for accessing information
inside correctionlib JSON files and the JERC text files used
for storing JEC and JER information in CMSSW.
"""
from __future__ import annotations

import hashlib
import ROOT
import os
import re

from correctionlib.schemav2 import CorrectionSet

from jerc2json.util import jme_filename_from_keys

__all__ = ["JSONProxy", "JERCTextFileProxy"]


class JSONProxy:
    """
    Thin wrapper around JSON file with filename-based caching
    of loaded schema and evaluator objects.
    """
    # cache already-loaded files for quicker access
    __instance_cache = {}

    def __new__(cls, json_file):
        # load existing structures from cache
        _cache = cls.__instance_cache
        if json_file in _cache:
            return _cache[json_file]

        # initialize new instance and store in cache
        obj = object.__new__(cls)
        obj._corr_set = CorrectionSet.parse_file(json_file)
        obj._evaluator = obj._corr_set.to_evaluator()
        cls.__instance_cache[json_file] = obj

        # return the object
        return obj

    @property
    def correction_set(self):
        return self._corr_set

    @property
    def evaluator(self):
        return self._evaluator


class JERCTextFileProxy:
    """
    Wrapper around a JERC text file, with caching and convenience
    methods for accessing the information stored therein.
    """
    # cache already-loaded files for quicker access
    __instance_cache = {}

    RE_JERC_TXT_FILE_NAME = "(?P<name>.*)_(?P<data_type>DATA|MC)_(?P<level>[^_]+)_(?P<algo>[^_]+)(?P<suffix>.*).txt"

    def __new__(cls, jme_text_file: str, section: str | None = None):
        # load existing structures from cache
        _cache = cls.__instance_cache
        if jme_text_file in _cache:
            return _cache[jme_text_file]

        # initialize new instance
        obj = object.__new__(cls)

        # try to use use regex to extract name, level and algo
        # information from filename
        jme_text_file_basename = os.path.basename(jme_text_file)
        obj._fname_metadata = {
            "basename": jme_text_file_basename,
        }
        m = re.match(cls.RE_JERC_TXT_FILE_NAME, jme_text_file_basename)
        if m:
            obj._fname_metadata.update(m.groupdict())

        # check file exists
        if not os.path.isfile(jme_text_file):
            raise FileNotFoundError(
                f"JERC text file not found: {jme_text_file}",
            )

        # compute file hash
        md5 = hashlib.md5()
        with open(jme_text_file, "rb") as f:
            while (data := f.read(65536)):
                md5.update(data)
        obj._hash = md5.hexdigest()

        # store table and section
        obj._section = section or ""
        obj._table = ROOT.JetCorrectorParameters(
            jme_text_file,
            obj._section,
        )

        # store in cache
        cls.__instance_cache[(jme_text_file, obj._section)] = obj

        # return the object
        return obj

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return self._hash == other._hash

    @classmethod
    def from_keys(
        cls,
        name: str,
        lvl: str,
        algo: str,
        prefix: str | None = None,
        section: str | None = None,
        basedir: str | None = None,
    ):
        basedir = basedir or os.getcwd()
        filename = jme_filename_from_keys(name, lvl, algo, prefix=prefix)
        return cls(
            os.path.join(basedir, filename),
            section=section,
        )

    @property
    def filename_metadata(self):
        return self._fname_metadata

    @property
    def data_type(self):
        return self._fname_metadata.get("data_type", None)

    @property
    def name(self):
        return self._fname_metadata.get("name", None)

    @property
    def table(self):
        return self._table

    @property
    def section(self):
        return self._section

    @property
    def definitions(self):
        return self.table.definitions()

    @property
    def formula(self):
        return self.definitions.formula()

    @property
    def level(self):
        return self.definitions.level()

    @property
    def variables(self):
        defs = self.definitions
        return [
            defs.parVar(i)
            for i in range(defs.nParVar())
        ]

    @property
    def binning_variables(self):
        defs = self.definitions
        return [
            defs.binVar(i)
            for i in range(defs.nBinVar())
        ]

    @property
    def n_binning_variables(self):
        return self.definitions.nBinVar()

    @property
    def n_variables(self):
        return self.definitions.nParVar()

    @property
    def all_variables(self):
        """
        Sorted list of all input variables present in the table.
        """
        return list(sorted(set(self.variables).union(self.binning_variables)))

    def get_binning_variable(self, index):
        return self.definitions.binVar(index)

    def get_variable(self, index):
        return self.definitions.parVar(index)

    def get_record(self, index):
        return self.table.record(index)
