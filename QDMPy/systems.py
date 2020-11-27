# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"

import numpy as np
import os
import misc
import re
import pathlib
import warnings


DIR_PATH = pathlib.Path(__file__).parent.absolute()


class System:
    name = "Unknown System"

    def __init__(self, *args, **kwargs):
        pass

    def read_raw(self, filepath, **kwargs):
        raise NotImplementedError

    def read_sweep_list(self, filepath, **kwargs):
        raise NotImplementedError

    def read_metadata(self, filepath, **kwargs):
        raise NotImplementedError

    def get_raw_pixel_size(self):
        raise NotImplementedError

    def option_choices(self, option_name):
        raise NotImplementedError

    def available_options(self):
        raise NotImplementedError


# Institute or university level
class UniMelb(System):
    name = "Unknown UniMelb System"

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

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

    def option_choices(self, option_name):
        return self.options_dict[option_name]["option_choices"]

    def option_characs(self, option_name):
        return self.options_dict[option_name]["option_characs"]

    def available_options(self):
        return self.options_dict.keys()


# 'system' level, inherits from broader institute class
# --> this is what should be passed around!
class Zyla(UniMelb):
    _raw_pixel_size = 1
    name = "Zyla"
    config_path = DIR_PATH + "options\\zyla_config.json"

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        # ensure all values default to None (at all levels of reading in json)
        self.option_dict = misc.json_to_dict(self.config_path, hook="dd")

    def get_pixel_size(self):
        return self._raw_pixel_size


def choose_system(name):
    return systems_dict[name]()


# ============================================================================


class OptionsError(Exception):
    def __init__(self, option_name, option_given, system, custom_msg=None):
        self.option_name = option_name
        self.option_given = option_given
        self.custom_msg = custom_msg
        super().__init__(custom_msg)

    def __str__(self):
        if self.custom_msg is not None:
            return str(self.custom_msg)
        else:
            # use system to get possible options (read specific json into dict etc.)
            return (
                f"Option {self.option_given} not a valid option for {self.option_name}"
                + f", pick from: {self.system.option_choices[self.option_name]}"
            )


def check_option(key, val, system):
    if key not in system.available_options():
        raise warnings.warn(
            f"Option {key} was not recognised by the {system.name} system, skipping."
        )
    elif system.option_choices(key) is not None and val not in system.option_choices(key):
        OptionsError(key, val, system)

    # TODO add here checks for specifics of each key etc. like list length etc.
    # can build a very comprehensive check on options! {use option_characs}

    # e.g. check fit_funcs are appropriate form


def check_options(options):
    system = options["system"]
    for key, val in options.items():
        check_option(key, val, system)
    options["cleaned"] = True


def clean_options(options):
    if "cleaned" in options.keys() and options["cleaned"]:
        return
    else:
        check_options(options)


systems_dict = {"zyla": Zyla}
