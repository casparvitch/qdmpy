# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"

import numpy as np
import os
import misc
import re


class Institute:
    def __init__(self):
        pass

    def read_raw(self, filepath, **kwargs):
        raise NotImplementedError

    def read_sweep_list(self, filepath, **kwargs):
        raise NotImplementedError

    def read_metadata(self, filepath, **kwargs):
        raise NotImplementedError


class UniMelb(Institute):
    def __init__(self):
        pass

    def read_raw(self, filepath, **kwargs):
        with open(os.path.normpath(filepath), "r") as fid:
            raw_data = np.fromfile(fid, dtype=np.float32())[2:]
        return raw_data

    def read_sweep_list(self, filepath, **kwargs):
        with open(os.path.normpath(filepath + "_metaSpool.txt"), "r") as fid:
            sweep_str = fid.readline().rstrip().split("\t")
        return [float(i) for i in sweep_str]

    def read_metadata(self, filepath, **kwargs):
        # TODO want to add a process here to read 'old bin conversion' option etc.
        # to change binning

        # skip over sweep list
        with open(os.path.normpath(filepath + "_metaSpool.txt"), "r") as fid:
            _ = fid.readline().rstrip().split("\t")
            # ok now read the metadata
            rest_str = fid.read()
            matches = re.findall(
                r"^([a-zA-Z0-9_ _/+()#-]+):([a-zA-Z0-9_ _/+()#-]+)",
                rest_str,
                re.MULTILINE,
            )
            metadata = {a: misc.failfloat(b) for (a, b) in matches}
        return metadata


institutes_dict = {"unimelb": UniMelb}


def choose_institute(name):
    return institutes_dict[name]()
