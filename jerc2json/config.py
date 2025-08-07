# coding: utf-8
"""
This module contains tools for reading in YAML configuration files.
"""
from __future__ import annotations

import os
import yaml

__all__ = ["ConfigDict", "default_config_path"]


class ConfigDict(dict):
    """
    Lightweight dict wrapper with dot-access.
    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    @classmethod
    def wrap(cls, to_wrap):
        wrap = lambda d: (
            cls({
                k: wrap(v)
                for k, v in d.items()
            })
            if isinstance(d, dict) else d
        )
        return wrap(to_wrap)


def load_config(path):
    """Load configuration from the specified YAML file."""
    with open(path, "r") as f:
        return ConfigDict.wrap(yaml.safe_load(f))


__this_dir__ = os.path.dirname(__file__)
default_config_path = os.path.join(__this_dir__, "..", "config.yml")
