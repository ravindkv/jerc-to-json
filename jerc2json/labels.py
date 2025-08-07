# coding: utf8
"""
Tools for labeling corrections
"""
from __future__ import annotations

import os

from jerc2json.proxy import JERCTextFileProxy


class JERCTextFileLabeler:
    """
    Helper class for inferring correction names and other
    labels heuristically from the JERC text file name.
    """

    tool_url_for_description = "https://gitlab.cern.ch/cms-jetmet/jerc2json"
    required_metadata_keys = ("name", "data_type", "content", "algo")

    def __init__(self, jme_file_paths_or_proxies: list[str] | list[JERCTextFileProxy]):
        self.jme_file_proxies = [
            jme_file_path_or_proxy
            if isinstance(jme_file_path_or_proxy, JERCTextFileProxy)
            else JERCTextFileProxy(jme_file_path_or_proxy)
            for jme_file_path_or_proxy in jme_file_paths_or_proxies
        ]

        # collect all existing metadata keys from proxy objects
        metadata_keys = {
            metadata_key
            for jme_file_proxy in self.jme_file_proxies
            for metadata_key in jme_file_proxy.filename_metadata
        }

        # collect unique metadata values from proxy objects
        metadata_values = {
            metadata_key: {
                jme_file_proxy.filename_metadata[metadata_key]
                for jme_file_proxy in self.jme_file_proxies
                if metadata_key in jme_file_proxy.filename_metadata
            }
            for metadata_key in metadata_keys
        }

        # compute 'content' key with either level or uncertainty name
        metadata_values["content"] = {
            jme_file_proxy.section
            if (level := jme_file_proxy.filename_metadata["level"]) == "UncertaintySources"
            else level
            for jme_file_proxy in self.jme_file_proxies
        }

        # merge file metadata to readable strings
        self.strings = {
            metadata_key: (
                list(metadata_value)[0]
                if len(metadata_value) == 1 else
                "{{{}}}".format(
                    ",".join(sorted(metadata_value))
                )
            )
            for metadata_key, metadata_value in metadata_values.items()
        }

        if all(k in self.strings for k in self.required_metadata_keys):
            self.strings["correction_name"] = "_".join(self.strings[k] for k in self.required_metadata_keys)
            self.strings["description"] = (
                f"{self.strings['content']} "
                f"for {self.strings['algo']} jets, "
                f"created from {self.strings['name']}_{self.strings['data_type']} "
                f"using {self.tool_url_for_description}"
            )
        else:
            print(
                "WARNING: unable to extract name/level/algo information from file "
                f"'{self.strings['basename']}': file name does not match JERC "
                "naming convention."
            )
            self.strings["correction_name"], _ = os.path.splitext(self.strings["basename"])
            self.strings["description"] = (
                f"created from {self.strings['basename']} "
                f"using {self.tool_url_for_description}"
            )

    def get(self, string_key, default=None):
        """
        Get a heuristically determined string by key. Returns 'None' if no
        corresponding string is found.
        """
        return self.strings.get(string_key, default)
