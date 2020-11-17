# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"


import misc
import os
import numpy as np
import pathlib

import institutes
import systems


def main(__spec__=None):

    options = misc.json_to_dict("options/<NAME>.json")

    institute = institutes.choose_institute(options["institute"])
    system = systems.choose_system(options["system"])

    raw_data = institute.read_raw(options["filepath"])
    sweep_list = institute.read_sweep_list(options["filepath"])
    metadata = institute.read_metadata(options["filepath"])

    # below: assuming haven't processed already. Write into a couple of functions

    options["original_bin"] = int(metadata["Binning"])
    if not int(options["additional_bins"]):
        options["total_bin"] = options["original_bin"]
    else:
        options["total_bin"] = options["original_bin"] * int(options["additional_bins"])

    output_dir = pathlib.PurePosixPath(
        options["filepath"] + "_processed" + "_bin_" + str(options["total_bin"])
    )
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    options["output_dir"] = output_dir

    options["data_dir"] = output_dir / "data"
    if not os.path.exists(options["data_dir"]):
        os.mkdir(options["data_dir"])

    # ok now start transforming dataset
