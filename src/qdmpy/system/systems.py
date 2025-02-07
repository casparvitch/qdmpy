# -*- coding: utf-8 -*-
"""
This sub-package holds classes and functions to define different systems.

This sub-package allows users with different data saving/loading procedures to use
this package. Also defined are variables such as raw_pixel_size which may be
different for different systems at the same institution.

Also defined are functions to handle the checking/cleaning of the json options
file (and then dict) to ensure it is valid etc.

_Make sure_ that any valid systems you define here are placed
in the _SYSTEMS dict at the bottom of the file.

Classes
-------
 - `qdmpy.system.systems.System`
 - `qdmpy.system.systems.UniMelb`
 - `qdmpy.system.systems.Zyla`
 - `qdmpy.system.systems.cQDM`
 - `qdmpy.system.systems.CryoWidefield`


Functions
---------
 - `qdmpy.system.systems.choose_system`


Module variables
----------------
 - `qdmpy.system.systems._CONFIG_PATH`
 - `qdmpy.system.systems._SYSTEMS`

"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.system.systems.System": True,
    "qdmpy.system.systems.UniMelb": True,
    "qdmpy.system.systems.Zyla": True,
    "qdmpy.system.systems.cQDM": True,
    "qdmpy.system.systems.CryoWidefield": True,
    "qdmpy.system.systems._CONFIG_PATH": True,
    "qdmpy.system.systems._SYSTEMS": True,
    "qdmpy.system.systems.choose_system": True,
}

# ============================================================================

import numpy as np
import pandas as pd
import os
import re
import pathlib
from multiprocessing import cpu_count
from math import radians
from importlib.metadata import version

# ============================================================================

import qdmpy.shared.json2dict
from qdmpy.shared.misc import warn

# ============================================================================

_CONFIG_PATH = pathlib.Path(__file__).parent.absolute()
"""
Path to the system directory (e.g. /qdmpy/system)
Allows access to config json files.
"""

# ============================================================================


class System:
    """Abstract class defining what is expected for a system."""

    name = "Unknown System"
    """Name of the system."""

    options_dict = None
    """Dictionary of available options for this system (loaded from config file)"""

    filepath_joined = False
    """Used to ensure base_dir is not prepended to filepath twice!"""

    def __init__(self, *args, **kwargs):
        """
        Initialisation of system. Must set options_dict.
        """
        raise NotImplementedError(
            "System init must set options_dict: override default!"
        )

    def read_image(self, filepath, options):
        """
        Method that must be defined to read raw data in from filepath.

        Returns
        -------
        image : np array, 3D
            Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has
            not been rebinned. Unwanted sweep values not removed.
        options : dict
            Generic options dict holding all the user options.
        """
        raise NotImplementedError

    def determine_binning(self, options):
        """
        Method that must be defined to define the original_bin and total_bin options
        (in the options dict). original_bin equiv. to a camera binning, e.g. any binning
        before qdmpy was run.

        Arguments
        ---------
        options : dict
            Generic options dict holding all the user options.
        """
        raise NotImplementedError

    def read_sweep_list(self, filepath):
        """
        Method that must be defined to read sweep_list in from filepath.
        """
        raise NotImplementedError

    def get_raw_pixel_size(self, options):
        """
        Method that must be defined to return a raw_pixel_size.

        Arguments
        ---------
        options : dict
            Generic options dict holding all the user options.
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

    def get_bias_field(self, options):
        """
        Method to get magnet bias field from experiment metadata,
        i.e. if set with programmed electromagnet. Default: False, None.

        Arguments
        ---------
        options : dict
            Generic options dict holding all the user options.

        Returns
        -------
        bias_on : bool
            Was programmed bias field used?
        bias_field : tuple
            Tuple representing vector bias field (B_mag (gauss), B_theta (rad), B_phi (rad))
        """
        return False, None

    def system_specific_option_update(self, options):
        """
        Hardcoded method that allows updating of the options at runtime with a custom script.

        In particular this defines some things that cannot be stored in a json.
        """
        # most systems will need these {you need to copy to your subclass method}
        # need to know number of threads to call (might be parallel fitting)
        options["threads"] = cpu_count() - options["scipyfit_sub_threads"]
        if (
            "base_dir" in options
            and options["base_dir"] is not None
            and not self.filepath_joined
        ):
            options["filepath"] = options["base_dir"] / options["filepath"]
            self.filepath_joined = True

    def get_headers_and_read_csv(self, options, path):
        """
        Harcoded method that allows reading of a csv file with 'other' measurement
        data (e.g. temperature) from a csv file. Needs to return
        headers, csv_data (as a list of strings, and a numpy array).
        The 1st column should be some sort of independ. data e.g. time.
        """
        raise NotImplementedError


# ============================================================================


# Institute or university level
class UniMelb(System):
    """
    University of Melbourne-wide properties of our systems.

    Inherited by specific systems defined as sub-classes.
    """

    name = "Unknown UniMelb System"
    uni_defaults_path = _CONFIG_PATH / "unimelb_defaults.json"

    def __init__(self, *args, **kwargs):
        # super().__init__(self, *args, **kwargs) # don't super init, it's a NotImplemented guard.
        # ensure all values default to None (at all levels of reading in json)

        # global defaults
        self.options_dict = qdmpy.shared.json2dict.json_to_dict(
            self.uni_defaults_path, hook="dd"
        )
        # system specific options, then recursively update
        sys_spec_opts = qdmpy.shared.json2dict.json_to_dict(self.config_path, hook="dd")
        qdmpy.shared.json2dict.recursive_dict_update(self.options_dict, sys_spec_opts)
        # in case we want this info in future, set the qdmpy version currently running
        self.options_dict["qdmpy_version"] = {
            "option_default": version("qdmpy"),
            "option_choices": None,
        }

    def get_raw_pixel_size(self, options):  # in metres
        if "pixel_size" in options and options["pixel_size"]:
            return options["pixel_size"]
        # override keys available as options
        override_keys = [
            "objective_mag",
            "objective_reference_focal_length",
            "camera_tube_lens",
        ]
        # not the cleanest way to do this... eh it works
        default_keys = ["default_" + key for key in override_keys]
        default_keys.insert(0, "sensor_pixel_size")
        settings_dict = self.options_dict["microscope_setup"]["option_default"]

        def key_finder(key):
            if key in settings_dict:
                return settings_dict[key]
            else:
                return None  # avoid keyerrors

        settings = [key_finder(key) for key in default_keys]

        for i, s in enumerate(override_keys):
            if options[s] is not None:
                settings[i + 1] = options[s]  # +1 to skip sensor pixel size (set)

        if None in settings:
            raise ValueError("Insufficient microscope setup settings provided.")

        sensor_pixel_size, mag, f_ref, f_tube = settings

        f_obj = f_ref / mag

        cam_pixel_size = sensor_pixel_size * (f_obj / f_tube)

        # save into options so it can be read from disk (by user) when saved
        # DON'T USE THESE IN CODE -> USE THIS FN
        options["calculated_raw_pixel_size"] = cam_pixel_size
        options["calculated_binned_pixel_size"] = cam_pixel_size * options["total_bin"]

        return cam_pixel_size

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
        if "scipyfit_sub_threads" in options:
            options["threads"] = cpu_count() - options["scipyfit_sub_threads"]

        # ensure only useful (scipy) loss method is used
        if "scipy_fit_method" in options:
            if options["scipy_fit_method"] == "lm":
                options["loss"] = "linear"

        if "freqs_to_use" in options:
            options["freqs_to_use"] = list(map(bool, options["freqs_to_use"]))

        if (
            "base_dir" in options
            and options["base_dir"] is not None
            and not self.filepath_joined
        ):
            options["filepath"] = os.path.join(options["base_dir"], options["filepath"])
            self.filepath_joined = True  # just a flag so we don't do this twice

        # add metadata to options (so it's saved for output)
        if "metadata" not in options:
            options["metadata"] = self._read_metadata(options["filepath"])

        options["filepath"] = os.path.normpath(options["filepath"])

    def get_headers_and_read_csv(self, options, path):
        # first get headers
        headers = pd.read_csv(path, sep=None, engine="python").columns.tolist()
        headers = [h for h in headers if not h.startswith("Unnamed")]
        # then load dataset
        dataset = np.genfromtxt(path, skip_header=1, autostrip=True, delimiter="\t")

        return headers, dataset


# ============================================================================


class LVControl(UniMelb):
    name = "Unknown Labview controlled System"

    def read_image(self, filepath, options):
        with open(os.path.normpath(filepath), mode="r") as fid:
            raw_data = np.fromfile(fid, dtype=np.float32)[2:]
        with open(os.path.normpath(filepath), mode="r") as fid:
            expected_shape = np.fromfile(fid, dtype=np.int32)[:2]

        options["binary_expected_shape"] = list(expected_shape.astype(int))
        return self._reshape_raw(options, raw_data, self.read_sweep_list(filepath))

    def read_sweep_list(self, filepath):
        with open(os.path.normpath(str(filepath) + "_metaSpool.txt"), "r") as fid:
            sweep_str = fid.readline().rstrip().split("\t")
        return [float(i) for i in sweep_str]

    def determine_binning(self, options):
        metadata = self._read_metadata(options["filepath"])

        options["original_bin"] = int(metadata["Binning"])
        if not int(options["additional_bins"]):
            options["total_bin"] = options["original_bin"]
        else:
            options["total_bin"] = options["original_bin"] * int(
                options["additional_bins"]
            )

    def _read_metadata(self, filepath):
        """
        Reads metaspool text file into a metadata dictionary.
        Filepath argument is the filepath of the (binary) dataset.
        """
        with open(os.path.normpath(str(filepath) + "_metaSpool.txt"), "r") as fid:
            # skip the sweep list (freqs or taus)
            _ = fid.readline().rstrip().split("\t")
            # ok now read the metadata
            rest_str = fid.read()
            # any text except newlines and tabs, followed by a colon and a space
            # then any text except newlines and tabs
            # (match the two text regions)
            matches = re.findall(
                # r"^([a-zA-Z0-9_ /+()#-]+):([a-zA-Z0-9_ /+()#-]+)",
                r"([^\t\n]+):\s([^\t\n]+)",
                rest_str,
                re.MULTILINE,
            )

            def failfloat(a):
                try:
                    return float(a)
                except ValueError:
                    if a == "FALSE":
                        return False
                    elif a == "TRUE":
                        return True
                    else:
                        return a

            metadata = {a: failfloat(b) for (a, b) in matches}
        return metadata

    def _reshape_raw(self, options, raw_data, sweep_list):
        """
        Reshapes raw data into more useful shape, according to image size in metadata.
        Unimelb-specific data reshaping procedure (relies upon metadata)

        Arguments
        ---------
        options : dict
            Generic options dict holding all the user options.
        raw_data : np array, 1D (unshaped)
            Raw unshaped data read from binary file
        sweep_list : list
            List of sweep parameter values

        Returns
        -------
        image : np array, 3D
            Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has
            not been rebinned. Unwanted sweep values not removed.
        """

        options["used_ref"] = False  # flag for later

        metadata = self._read_metadata(options["filepath"])

        # NOTE: AOIHeight and AOIWidth are saved by labview the opposite of what you'd expect
        # -> LV rotates to give image as you'd expect standing in lab
        # -> thus, we ensure all images in this processing code matches LV orientation
        try:
            if not options["ignore_ref"]:
                data_pts = len(sweep_list)
                image = np.reshape(
                    raw_data,
                    [
                        data_pts,
                        int(metadata["AOIHeight"]),
                        int(metadata["AOIWidth"]),
                    ],
                )
                options["used_ref"] = False
            else:
                raise ValueError
        except ValueError:
            # if the ref is used then there's 2* the number of sweeps
            # i.e. auto-detect reference existence
            data_pts = 2 * len(sweep_list)
            if options["ignore_ref"]:
                options["used_ref"] = False
                image = np.reshape(
                    raw_data,
                    [
                        data_pts,
                        int(metadata["AOIHeight"]),
                        int(metadata["AOIWidth"]),
                    ],
                )[
                    ::2
                ]  # hmmm disregard ref -> use every second element.
            else:
                image = np.reshape(
                    raw_data,
                    [
                        data_pts,
                        int(metadata["AOIHeight"]),
                        int(metadata["AOIWidth"]),
                    ],
                )
                options["used_ref"] = True
        # Transpose the dataset to get the correct x and y orientations ([y, x])
        # will work for non-square images
        return image.transpose([0, 2, 1]).copy()

    def get_bias_field(self, options):
        """get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad))"""
        if "metadata" not in options:
            return False, None
        key_ars = [
            ["Field Strength (G)"],
            ["Theta (deg)"],
            ["Phi (def)", "Phi (deg)"],
        ]
        bias_field = []
        for ar in key_ars:
            found = False
            for key in ar:
                if key in options["metadata"]:
                    bias_field.append(options["metadata"][key])
                    found = True
            if not found:
                return False, None
        if len(bias_field) != 3:
            warn(
                f"Found {len(bias_field)} bias field params in metadata, "
                + "this shouldn't happen (expected 3)."
            )
            return False, None
        onoff_str = options["metadata"].get("Mag on/off", "")
        bias_on = onoff_str == " TRUE"
        return bias_on, (
            bias_field[0],
            radians(bias_field[1]),
            radians(bias_field[2]),
        )


# ============================================================================


class PyControl(UniMelb):
    name = "Unknown Python controlled System"

    def read_image(self, filepath, options):
        im = np.load(os.path.normpath(str(filepath) + ".npy"))
        if (
            options["ignore_ref"]
            and self._read_metadata(options["filepath"])["Measurement"]["ref_bool"]
        ):
            # ignore ref but have on
            options["used_ref"] = False
            return im[::2]
        elif not self._read_metadata(options["filepath"])["Measurement"]["ref_bool"]:
            # don't have a ref
            options["used_ref"] = False
            return im
        else:
            # do have a ref and want to use it
            options["used_ref"] = True
            return im

    def read_sweep_list(self, filepath):
        return qdmpy.shared.json2dict.json_to_dict(
            os.path.normpath(str(filepath) + ".json")
        )["freq_list"] # TODO won't work for tau sweeps

    def determine_binning(self, options):
        metadata = self._read_metadata(options["filepath"])

        binning = metadata["Devices"]["camera"]["bin"]
        if binning[0] != binning[1]:
            raise ValueError("qmdpy not setup to handle anisotropic binning.")
        options["original_bin"] = int(binning[0])

        if not int(options["additional_bins"]):
            options["total_bin"] = options["original_bin"]
        else:
            options["total_bin"] = options["original_bin"] * int(
                options["additional_bins"]
            )

    def _read_metadata(self, filepath):
        """
        Reads metaspool text file into a metadata dictionary.
        Filepath argument is the filepath of the (binary) dataset.
        """
        pth = os.path.normpath(str(filepath) + "_metadata.json")
        return qdmpy.shared.json2dict.json_to_dict(pth, hook="dd")

    def get_bias_field(self, options):
        """get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad))"""
        if "metadata" not in options:
            return False, None
        keys = [
            "bnorm",  # update with units?
            "theta",
            "phi",
        ]
        bias_field = []
        for key in keys:
            if key in options["metadata"]["Devices"]["null"]:
                bias_field.append(options["metadata"]["Devices"]["null"][key])
            else:
                return False, None
        if len(bias_field) != 3:
            warn(
                f"Found {len(bias_field)} bias field params in metadata, "
                + "this shouldn't happen (expected 3)."
            )
            return False, None

        return True, (
            10 * bias_field[0],  # convert mT to gauss
            radians(bias_field[1]),
            radians(bias_field[2]),
        )


# ============================================================================


# 'system' level, inherits from broader institute class
# --> this is what should be passed around!
class Zyla(LVControl):
    """
    Specific system details for the Zyla QDM.
    """

    name = "Zyla"
    config_path = _CONFIG_PATH / "zyla_config.json"


class cQDM(LVControl):  # noqa: N801
    """
    Specific system details for the cQDM QDM.
    """

    name = "cQDM"
    config_path = _CONFIG_PATH / "cqdm_config.json"


class CryoWidefield(LVControl):
    """
    Specific system details for Cryogenic (Attocube) widefield QDM.
    """

    name = "Cryo Widefield"
    config_path = _CONFIG_PATH / "cryo_widefield_config.json"


class LegacyCryoWidefield(LVControl):
    """
    Specific system details for cryogenic (Attocube) widefield QDM.
    - Legacy binning version
    """

    name = "Legacy Cryo Widefield"
    config_path = _CONFIG_PATH / "cryo_widefield_config.json"

    def determine_binning(self, options):
        # silly old binning convention -> change when labview updated to new binning
        bin_conversion = [1, 2, 3, 4, 8]
        metadata = self._read_metadata(options["filepath"])
        metadata_bin = int(metadata["Binning"])
        options["original_bin"] = bin_conversion[metadata_bin]

        if not int(options["additional_bins"]):
            options["total_bin"] = options["original_bin"]
        else:
            options["total_bin"] = options["original_bin"] * int(
                options["additional_bins"]
            )


class Argus(LVControl):
    """
    Specific system details for Argus room-temperature widefield QDM.
    """

    name = "Argus"
    config_path = _CONFIG_PATH / "argus_config.json"


class LegacyArgus(LVControl):
    name = "Legacy Argus"
    config_path = _CONFIG_PATH / "argus_config.json"

    def determine_binning(self, options):
        # silly old binning convention -> change when labview updated to new binning
        bin_conversion = [1, 2, 3, 4, 8]
        metadata = self._read_metadata(options["filepath"])
        metadata_bin = int(metadata["Binning"])
        options["original_bin"] = bin_conversion[metadata_bin]

        if not int(options["additional_bins"]):
            options["total_bin"] = options["original_bin"]
        else:
            options["total_bin"] = options["original_bin"] * int(
                options["additional_bins"]
            )


class PyCryoWidefield(PyControl):
    """
    Specific system details for Cryogenic (Attocube) widefield QDM.
    """

    name = "Py Cryo Widefield"
    config_path = _CONFIG_PATH / "cryo_widefield_config.json"


# ============================================================================

_SYSTEMS = {
    "Zyla": Zyla,
    "Cryo_Widefield": CryoWidefield,
    "Legacy_Cryo_Widefield": LegacyCryoWidefield,
    "cQDM": cQDM,
    "Argus": Argus,
    "Legacy_Argus": LegacyArgus,
    "Default": Argus,
    "Py_Cryo_Widefield": PyCryoWidefield,
}
"""Dictionary that defines systems available for use.

Add any systems you define here so you can use them.
"""


def choose_system(name):
    """Returns `qdmpy.system.systems.System` object called 'name'."""
    if name is None:
        raise RuntimeError("System chosen was 'None': override this default option!!!")
    return _SYSTEMS[name]()


# ============================================================================
