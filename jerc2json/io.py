# coding: utf8
"""
Tools for processing files
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import shutil


def get_unpack_tarball(src, work_dir=None, local_filename=None, overwrite=False):
    """
    Obtain a tarball from a source location (URL/local path) and extract its contents into the work directory
    `work_dir`. A `local_filename` can be specified for the local copy, otherwise it will
    be inferred from the URL. If the files already exist, nothing is done unless the
    `overwrite` flag is set.
    """
    # ensure target dir exists
    work_dir = work_dir or os.getcwd()
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # infer local filename name from src, if not given
    if not local_filename:
        local_filename = os.path.basename(src)

    # infer name of the local file and target directory
    local_file_basename = os.path.basename(local_filename).split(".")[0]
    target_file = os.path.join(work_dir, local_filename)
    target_dir = os.path.join(work_dir, local_file_basename)

    # retrieve tarball from repo
    if overwrite or not os.path.exists(target_file):
        print(f"Obtaining tarball from: {src}")
        if "://" in src:
            logging.debug("src is URL: using wget")
            subprocess.run([
                "wget",
                "-q",
                "-O",
                target_file,
                src,
            ])
        else:
            logging.debug("src is local file path: using shutil.copy2")
            shutil.copy2(src, target_file)

    else:
        logging.debug(
            f"no need to obtain tarball (already present): {target_file}",
        )

    # extract tarball
    if overwrite or not os.path.exists(target_dir):
        print(f"Extracting tarball: {target_file}")
        shutil.unpack_archive(target_file, target_dir)
    else:
        logging.debug(
            f"skipping tarball extraction (directory already exists): {target_dir}",
        )


def patch_jer_file(input_file: str, output_file: str):
    """
    Modify a JERC text file `input_file`, replacing JER-specific
    strings like "Resolution" or "ScaleFactor" with the keyword
    "Correction" so it is accepted by `JetCorrectorParameters`.
    The patched file is saved to `output_file`.
    """
    if os.path.exists(output_file) and os.path.samefile(input_file, output_file):
        raise ValueError("`input_file` and `output_file` are the same file")

    with open(input_file, "r") as fin:
        with open(output_file, "w") as fout:
            for line in fin:
                for to_replace, replace_with in patch_jer_file.REPLACEMENTS.items():
                    if to_replace not in line:
                        continue
                    line = line.replace(to_replace, replace_with)
                fout.write(line)


patch_jer_file.REPLACEMENTS = {
    "Resolution": "Correction Resolution",
    "ScaleFactor": "Correction ScaleFactor",
    "SF": "Correction ScaleFactor",
    "TotDown TotUp": "",
}


def compile_jme_uncertainties(input_files: list[str], prefixes: list[str], output_file: str):
    """
    Given a list `input_files` of JERC "UncertaintySources" files and a list `prefixes`
    of the same length, produce an `output_file` with the combined content of the input files,
    prepending the corresponding prefix to the uncertainty names.
    """
    # validate inputs
    if len(input_files) != len(prefixes):
        raise ValueError(
            f"number of input files ({len(input_files)}) != "
            f"number of prefixes ({len(prefixes)})",
        )

    # open the output file for writing
    with open(output_file, "w") as fout:

        # loop through the input files and prefixes
        for prefix, input_file in zip(prefixes, input_files):
            prefix = prefix or ""

            # write input lines to the output file,
            # prepending prefix as necessary
            with open(input_file, "r") as fin:
                for line in fin:
                    line = line.strip()

                    # ignore comment lines
                    if line.startswith("#"):
                        continue
                    line = re.sub(r"^(\s*\[)", rf"\1{prefix}", line)
                    fout.write(line + "\n")


def read_sections(jme_file: str):
    """Read a JERC text file and return a list of sections found therein."""
    with open(jme_file, "r") as f:
        uncertainty_sources = [
            line.strip().strip("[]")
            for line in f
            if line.strip().startswith("[")
        ]
    return uncertainty_sources


def calc_file_hash(fname: str, chunk_size: int = 65535, hash_algorithm: str = "md5") -> hashlib._hashlib.HASH:
    """
    Calculate hash of file contents.
    """
    # create an empty hash object to update
    hash_obj = hashlib.new(hash_algorithm)

    # read in the file in chunks of `chunk_size`
    with open(fname, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            hash_obj.update(data)

    # return hash object
    return hash_obj
