# -*- coding: utf-8 -*-
"""
This module holds classes and functions to define different systems.

This module allows users with different data saving/loading procedures to use
this package. Also defined are variables such as raw_pixel_size which may be
different for different systems at the same institution.

Also defined are functions to handle the checking/cleaning of the json options
file (and then dict) to ensure it is valid etc.

_Make sure_ that any valid systems you define here are placed
in the SYSTEMS_DICT defined near the bottom.

Classes
-------
 - `QDMPy.systems.System`
 - `QDMPy.systems.UniMelb`
 - `QDMPy.systems.Zyla`
 - `QDMPy.systems.OptionsError`


Functions
---------
 - `QDMPy.systems.check_option`
 - `QDMPy.systems.check_options`
 - `QDMPy.systems.clean_options`


Module variables
----------------
 - `QDMPy.systems.DIR_PATH`
 - `QDMPy.systems.SYSTEMS_DICT`

"""


# ============================================================================

__author__ = "Sam Scholten"

# ============================================================================

import numpy as np
import os
import re
import pathlib
import warnings
from multiprocessing import cpu_count

# ============================================================================

import QDMPy.misc as misc

# ============================================================================


DIR_PATH = pathlib.Path(__file__).parent.absolute()
"""
Path to QDMPy directory, so you can navigate to options directory (e.g. to read config json files)
"""


class System:
    """Abstract class defining what is expected for a system."""

    name = "Unknown System"
    """ """

    options_dict = None
    """ """

    def __init__(self, *args, **kwargs):
        """
        Initialisation of system. Must set options_dict.
        """
        pass

    def read_raw(self, filepath, **kwargs):
        """
        Method that must be defined to read raw data in from filepath.
        """
        raise NotImplementedError

    def read_sweep_list(self, filepath, **kwargs):
        """
        Method that must be defined to read sweep_list in from filepath.
        """
        raise NotImplementedError

    def read_metadata(self, filepath, **kwargs):
        """
        Method that must be defined to read metadata in from filepath.
        """
        raise NotImplementedError

    def get_raw_pixel_size(self):
        """
        Method that must be defined to return a raw_pixel_size.
        """
        raise NotImplementedError

    def get_default_options(self):
        """
        Method that must be defined to return an options dictionary of default values.
        """
        raise NotImplementedError

    def option_choices(self, option_name):
        """
        Method that must be defined to return the available option choices for a given option_name
        """
        raise NotImplementedError

    def available_options(self):
        """
        Method that must be defined to return what options are available for this system.
        """
        raise NotImplementedError

    def system_specific_option_update(self, options):
        """
        Hardcoded method that allows updating of the options at runtime with a custom script.

        In particular this defines some things that cannot be stored in a json.
        """
        # most systems will need these {you need to copy to your subclass method}
        # need to know number of threads to call (might be parallel fitting)
        options["threads"] = cpu_count() - options["scipy_sub_threads"]
        if "base_dir" in options:
            if options["base_dir"] == "test_datasets":
                # find tests path in this repo and prepend
                options["filepath"] = DIR_PATH / "tests/test_datasets/" / options["filepath"]
            elif options["base_dir"] != "":
                options["filepath"] = options["base_dir"] / options["filepath"]


# ============================================================================


# Institute or university level
class UniMelb(System):
    """
    University of Melbourne-wide properties of our systems.

    Inherited by specific systems defined as sub-classes.
    """

    name = "Unknown UniMelb System"

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def read_raw(self, filepath, **kwargs):
        with open(os.path.normpath(filepath), "r") as fid:
            raw_data = np.fromfile(fid, dtype=np.float32())[2:]
        return raw_data

    def read_sweep_list(self, filepath, **kwargs):
        with open(os.path.normpath(str(filepath) + "_metaSpool.txt"), "r") as fid:
            sweep_str = fid.readline().rstrip().split("\t")
        return [float(i) for i in sweep_str]

    def read_metadata(self, filepath, **kwargs):

        # skip over sweep list
        with open(os.path.normpath(str(filepath) + "_metaSpool.txt"), "r") as fid:
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

    def get_default_options(self):
        ret = {}
        for key, val in self.options_dict.items():
            ret[key] = val["option_default"]
        return ret

    def option_choices(self, option_name):
        return self.options_dict[option_name]["option_choices"]

    def available_options(self):
        return self.options_dict.keys()

    def system_specific_option_update(self, options):
        # set some things that cannot be stored in the json

        # need to know number of threads to call (might be parallel fitting)
        options["threads"] = cpu_count() - options["scipy_sub_threads"]

        options["filepath"] = os.path.normpath(options["filepath"])
        
        # ensure only useful (scipy) loss method is used
        if "scipy_fit_method" in options:
            if options["scipy_fit_method"] == "lm":
                options["loss"] = "linear"

        if "base_dir" in options:
            if options["base_dir"] == "test_datasets":
                # find tests path in this repo and prepend
                options["filepath"] = DIR_PATH / "tests/test_datasets/" / options["filepath"]
            elif options["base_dir"] != "":
                options["filepath"] = os.path.join(options["base_dir"], options["filepath"])


# ============================================================================


# 'system' level, inherits from broader institute class
# --> this is what should be passed around!
class Zyla(UniMelb):
    """
    Specific system details for the Zyla QDM.
    """

    name = "Zyla"
    config_path = DIR_PATH / "options/zyla_config.json"

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        # ensure all values default to None (at all levels of reading in json)
        self.options_dict = misc.json_to_dict(self.config_path, hook="dd")

    def get_raw_pixel_size(self):
        return self.options_dict["raw_pixel_size"]["option_default"]


class LiamsWidefield(UniMelb):
    """
    Specific system details for Liam's Widefield QDM. Currently a copy of Zyla
    """

    name = "Liams Widefield"
    config_path = DIR_PATH / "options/liam_widefield_config.json"

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        # ensure all values default to None (at all levels of reading in json)
        self.options_dict = misc.json_to_dict(self.config_path, hook="dd")

    def get_raw_pixel_size(self):
        return self.options_dict["raw_pixel_size"]["option_default"]


def choose_system(name):
    return SYSTEMS_DICT[name]()


# ============================================================================


class OptionsError(Exception):
    """
    Exception with custom messages for errors to do with options dictionary.
    """

    def __init__(self, option_name, option_given, system, custom_msg=None):

        self.custom_msg = custom_msg

        choices = system.option_choices(option_name)

        if choices is not None:
            self.default_msg = (
                f"Option {option_given} not a valid option for {option_name}"
                + f", pick from: {choices}"
            )
        else:
            self.default_msg = f"Option {option_given} not a valid option for {option_name}."

        super().__init__(custom_msg)

    def __str__(self):
        if self.custom_msg is not None:
            return str(self.custom_msg) + "\n" + self.default_msg
        else:
            # use system to get possible options (read specific json into dict etc.)
            return self.default_msg


# ===============================


def check_option(key, val, system):
    if key not in system.available_options():
        warnings.warn(f"Option {key} was not recognised by the {system.name} system.")
    elif system.option_choices(key) is not None and val not in system.option_choices(key):
        OptionsError(key, val, system)


# ===============================


def check_options(options):
    system = options["system"]
    for key, val in options.items():
        check_option(key, val, system)
    options["cleaned"] = True


# ===============================


def clean_options(options):
    if "cleaned" in options.keys() and options["cleaned"]:
        return
    else:
        check_options(options)


SYSTEMS_DICT = {"Zyla": Zyla, "Liams_Widefield": LiamsWidefield}
"""
Dictionary that defines systems available for use.

Add any systems you define here so you can use them.
"""
