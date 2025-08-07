# coding: utf-8
from __future__ import annotations

import argparse
import numpy as np
import os
import yaml

from collections import defaultdict

from rich.console import Console

from jerc2json.cli.util import load_check_config, setup_config_eras


def do_plots(validation_results, max_per_page=30):
    """
    Perform the actual plotting of the `validation_result`.

    Multiple plots will be created if the number of entries
    exceeds `max_per_page`.
    """
    import matplotlib.markers as markers
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    stat_names = ("min", "mean", "max", "median", "std")

    labels = []
    values = defaultdict(list)
    for corr_name, corr_results in validation_results.items():
        stats_diff = corr_results["correction_values_stats"]["diff"]
        labels.append(corr_name)
        for stat_name in stat_names:
            values[stat_name].append(stats_diff[f"nan{stat_name}"])

    arrays = {k: np.asarray(v) for k, v in values.items()}

    figs_axes = []
    for i in range(0, len(labels), max_per_page):
        slc = slice(i, i + max_per_page)

        labels_slc = labels[slc]
        arrays_slc = {k: v[slc] for k, v in arrays.items()}

        fig = plt.figure(figsize=(8, 4.5))
        gs = GridSpec(
            nrows=1,
            ncols=1,
            left=0.05,
            right=0.48,
            top=0.98,
            bottom=0.1,
            figure=fig,
        )
        ax = fig.add_subplot(gs[0])

        ax.errorbar(
            arrays_slc["mean"], labels_slc,
            xerr=arrays_slc["std"],
            marker="o", color="k",
            label=r"mean $\pm$ st. dev.",
            linestyle="none",
        )

        ax.scatter(arrays_slc["mean"] + arrays_slc["std"], labels_slc, marker="|", color="k", zorder=10)
        ax.scatter(arrays_slc["mean"] - arrays_slc["std"], labels_slc, marker="|", color="k", zorder=11)
        ax.scatter(arrays_slc["min"], labels_slc, marker=markers.CARETRIGHT, color="k", label="minimum", zorder=11)
        ax.scatter(arrays_slc["max"], labels_slc, marker=markers.CARETLEFT, color="k", label="maximum", zorder=12)
        ax.scatter(arrays_slc["median"], labels_slc, marker=markers.CARETDOWN, color="r", label="median", zorder=13)

        ax.legend(frameon=False, ncol=2, loc="upper left")
        ax.grid()
        ax.yaxis.tick_right()

        ylim = ax.get_ylim()
        extend = 0.2 * (ylim[1] - ylim[0])
        ax.set_ylim((ylim[0], ylim[1] + extend))

        print(len(labels_slc))
        ax.set_yticks(list(range(len(labels_slc))))
        ax.set_yticklabels(labels_slc, fontsize=8)

        ax.set_xlabel("correctionlib $-$ CMSSW")

        figs_axes.append((fig, ax))

    return figs_axes


def plot_validation(console: Console, args: argparse.Namespace):
    """
    Plot the closure between corrections obtained with CMSSW from the
    JERC text files and with the newly generated correctionlib JSON files.
    """
    # load config, check file exists and eras are valid
    config = load_check_config(console, args)

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

        # jet paths to JSON files
        outputs = era_cfg.get("outputs", config.defaults.outputs)
        json_files = [
            f"{output_cfg['name']}.json"
            for output_cfg in outputs
        ]

        # store validation results
        for json_file in json_files:
            json_file_basename, _ = os.path.splitext(json_file)
            results_file = os.path.join(
                config.output_dir,
                era,
                f"{json_file_basename}.validation_{args.validation_type}.yml",
            )
            with open(results_file, "r") as f:
                results = yaml.safe_load(f)
            print(f"Loaded validation results from: {results_file}")

            figs_axes = do_plots(results)
            for i, (fig, _) in enumerate(figs_axes):
                plot_file = os.path.join(
                    config.output_dir,
                    era,
                    f"{json_file_basename}.validation_{args.validation_type}.{i}.png"
                )
                fig.savefig(plot_file)
                print(f"Saved validation plot to: {plot_file}")


def setup_plot_validation(subparsers: argparse._SubParsersAction):
    """
    Add the CLI parameters required for subcommand 'plot_validation'.
    """
    sp = subparsers.add_parser("plot_validation", help=plot_validation.__doc__)
    sp.set_defaults(command=plot_validation)

    setup_config_eras(sp)

    sp.add_argument(
        "-v",
        "--validation-type",
        default="simple",
        choices=("simple", "grid"),
        help="type of validation to perform; simple = take test points from config, "
        "grid = compute validation grid from correction binning",
    )
