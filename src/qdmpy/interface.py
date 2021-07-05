# -*- coding: utf-8 -*-
"""
This module defines some ease-of-use methods for the qdmpy package.

Functions
---------
 - `qdmpy.interface.initialize`
 - `qdmpy.interface.load_options`
 - `qdmpy.interface.save_options`
 - `qdmpy.interface._add_bias_field`
 - `qdmpy.interface._get_bias_field`
 - `qdmpy.interface._spherical_deg_to_cart`
 - `qdmpy.interface._define_output_dir`
 - `qdmpy.interface._interpolate_option_str`
 - `qdmpy.interface.load_polygons`
 - `qdmpy.interface.check_option`
 - `qdmpy.interface.check_options`
 - `qdmpy.interface.clean_options`

Classes
-------
 - `qdmpy.interface.OptionsError`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.interface.initialize": True,
    "qdmpy.interface.load_options": True,
    "qdmpy.interface.save_options": True,
    "qdmpy.interface._add_bias_field": True,
    "qdmpy.interface._get_bias_field": True,
    "qdmpy.interface._spherical_deg_to_cart": True,
    "qdmpy.interface._define_output_dir": True,
    "qdmpy.interface._interpolate_option_str": True,
    "qdmpy.interface.load_polygons": True,
    "qdmpy.interface.check_option": True,
    "qdmpy.interface.check_options": True,
    "qdmpy.interface.clean_options": True,
    "qdmpy.interface.OptionsError": True,
}
# ============================================================================

import numpy as np
import warnings
from collections import OrderedDict  # insertion order is guaranteed for py3.7+, but to be safe!
import pathlib
import re

# ============================================================================

import qdmpy.shared.json2dict
import qdmpy.plot
import qdmpy.system
import qdmpy.shared.polygon
import qdmpy.pl

# ============================================================================


def initialize(
    options_dict=None,
    options_path=None,
    ref_options_dict=None,
    ref_options_dir=None,
    set_mpl_rcparams=True,
):
    """Helped function to initialise analysis program.

    Arguments
    ---------
    options_dict : dict, default=None
        Generic options dict holding all the user options (for the main/signal experiment).
    options_path : str or path object, default=None
        Direct path to options json, i.e. will run something like 'read(options_path)'.
    ref_options_dict : dict, default=None
        Generic options dict holding all the user options (for the reference experiment).
    ref_options_dir : str or path object, default=None
        Path to read reference options from,
        i.e. will run something like 'read('ref_options_dir / saved_options.json')'.

    Returns
    -------
    options_dict : dict
        (Processed) generic options dict holding all user options.
    ref_options_dict : dict
        As options_dict, but for reference experiment (assuming pl already fit).

    """

    options = load_options(
        options_dict=options_dict,
        options_path=options_path,
        check_for_prev_result=True,
        loading_ref=False,
    )

    if set_mpl_rcparams:
        qdmpy.plot.set_mpl_rcparams(options)

    ref_options = load_ref_options(
        options, ref_options=ref_options_dict, ref_options_dir=ref_options_dir
    )

    return options, ref_options


# ============================================================================


def load_options(
    options_dict=None, options_path=None, check_for_prev_result=False, loading_ref=False
):
    """
    Load and process options (from json file or dict) into generic options dict used everywhere.

    Also handles directory creation etc. to put results in.
    Provide either options_dict OR options_path (must provide one!).

    Note the system default options are loaded in, so you only need to set the things you need.
    In particular, filepath, fit_functions and system_name must be set

    Optional Arguments
    -------------------
    options_dict : dict, default: None
        Directly pass in a dictionary of options.
    path : string, default: None
        Path to fit options .json file. Can be absolute, or from qdmpy.
    check_for_prev_result : bool, default: false
        Check to see if there's a previous fit result for these options.
    loading_ref : bool
        Reloading reference fit result, so ensure we check for previous fit result.
        Passed on to check_if_already_fit.

    Returns
    -------
    options : dict
        Generic options dict holding all the user options.
    """

    # pass options_dict and/or options_path
    if options_dict is not None and options_path is not None:
        # options_dict takes precedence
        options_path = None
    if options_dict is None and options_path is None:
        raise RuntimeError("pass at least one of options_dict and options_path to load_options")

    if options_path is not None:
        if not pathlib.Path(options_path).is_file():
            raise ValueError("options file at `options_path` not found?")
        prelim_options = qdmpy.shared.json2dict.json_to_dict(options_path)
    else:
        prelim_options = OrderedDict(options_dict)  # unnescessary py3.7+, leave to describe intent

    required_options = ["filepath", "fit_functions"]
    for key in required_options:
        if key not in prelim_options:
            raise RuntimeError(f"Must provide these options: {required_options}")

    chosen_system = qdmpy.system.choose_system(prelim_options["system_name"])

    chosen_system.system_specific_option_update(prelim_options)

    options = chosen_system.get_default_options()  # first load in default options
    # now update with what has been decided upon by user
    options = qdmpy.shared.json2dict.recursive_dict_update(options, prelim_options)

    chosen_system.determine_binning(options)

    chosen_system.system_specific_option_update(
        options
    )  # do this again on full options to be sure

    options["system"] = chosen_system

    _add_bias_field(options)

    clean_options(options)  # check all the options make sense

    _define_output_dir(options)
    pathlib.Path(options["output_dir"]).mkdir(exist_ok=True)
    pathlib.Path(options["data_dir"]).mkdir(exist_ok=True)

    load_polygons(options)

    # don't always check for prev. results (so we can use this fn in other contexts)
    if check_for_prev_result or loading_ref:
        qdmpy.pl.check_if_already_fit(options, loading_ref=loading_ref)
    return options


# ============================================================================


def load_ref_options(options, ref_options=None, ref_options_dir=None):
    """ref_options dict -> pixel_fit_params dict.

    Provide one of ref_options and ref_options_dir. If both are None, returns None (with a
    warning). If both are supplied, ref_options takes precedence.

    Also (importantly) creates output directories for field results etc. now that we know
    the context of any reference.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options (for the main/signal experiment).
    ref_options : dict, default=None
        Generic options dict holding all the user options (for the reference experiment).
    ref_options_dir : str or path object, default=None
        Path to read reference options from, i.e. will read 'ref_options_dir / saved_options.json'.

    Returns
    -------
    ref_options : dict
        Processed generic options dict for reference experiment.

    """
    if not ref_options or options["exp_reference_type"] is None:
        ref_options = None
    if not ref_options_dir or options["exp_reference_type"] is None:
        ref_options_dir = None
    if ref_options is None and ref_options_dir is None:
        warnings.warn(
            "Continuing without reference. (No reference chosen or exp_referece_type was 'None')"
        )
        options["field_dir"] = options["output_dir"].joinpath("field")
        options["field_sig_dir"] = options["field_dir"].joinpath("sig")
        options["field_ref_dir"] = options["field_dir"].joinpath("ref_nothing")
        options["field_sig_sub_ref_dir"] = options["field_dir"].joinpath("sig_sub_ref")

        pathlib.Path(options["field_dir"]).mkdir(exist_ok=True)
        pathlib.Path(options["field_sig_dir"]).mkdir(exist_ok=True)
        pathlib.Path(options["field_ref_dir"]).mkdir(exist_ok=True)
        pathlib.Path(options["field_sig_sub_ref_dir"]).mkdir(exist_ok=True)
        return None

    if ref_options_dir is not None:
        ref_options_path = pathlib.Path(ref_options_dir) / "saved_options.json"
    else:
        ref_options_path = None

    ref_options = load_options(
        options_dict=ref_options,
        options_path=ref_options_path,
        check_for_prev_result=True,
        loading_ref=True,
    )
    # copy reference bias to sig options.
    options["ref_bias_field_cartesian_gauss"] = ref_options["bias_field_cartesian_gauss"]
    options["ref_bias_field_spherical_deg_gauss"] = ref_options["bias_field_spherical_deg_gauss"]

    ref_name = pathlib.Path(ref_options["filepath"]).stem
    options["field_dir"] = options["output_dir"].joinpath("field")
    options["field_sig_dir"] = options["field_dir"].joinpath("sig")
    options["field_ref_dir"] = options["field_dir"].joinpath(f"ref_{ref_name}")
    options["field_sig_sub_ref_dir"] = options["field_dir"].joinpath("sig_sub_ref")

    pathlib.Path(options["field_dir"]).mkdir(exist_ok=True)
    pathlib.Path(options["field_sig_dir"]).mkdir(exist_ok=True)
    pathlib.Path(options["field_ref_dir"]).mkdir(exist_ok=True)
    pathlib.Path(options["field_sig_sub_ref_dir"]).mkdir(exist_ok=True)

    return ref_options


# ============================================================================


def save_options(options):
    """
    Saves generic options dict to harddrive as json file (in options["output_dir"])

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    """

    keys_to_remove = ["system", "polygons"]
    save_options = {}

    for key, val in options.items():
        if key.endswith("dir") or key.endswith("path"):
            val = str(val).replace("\\", "\\\\")
        if key not in keys_to_remove:
            save_options[key] = val
    qdmpy.shared.json2dict.dict_to_json(
        save_options, "saved_options.json", path_to_dir=options["output_dir"]
    )


# ============================================================================


def _add_bias_field(options):
    """Adds bias field in-place to options."""
    bias_cart = _get_bias_field(options)
    bias_spherical_deg = _get_bias_field(options, spherical_deg=True)
    options["bias_field_cartesian_gauss"] = bias_cart
    options["bias_field_spherical_deg_gauss"] = bias_spherical_deg


# ============================================================================


def _get_bias_field(options, spherical_deg=False):
    """
    Returns (bx, by, bz) guess for the bias field in Gauss.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    spherical_deg : bool, default=False
        Return as (magnitude, theta, phi) in Gauss & degrees.

    Returns
    -------
    bxyz : tuple
        (bx, by, bz) for the bias field, in Gauss. (or in spherical polar Gauss, degrees).
    """
    bias_field = None
    if options["auto_read_bias"]:
        bias_on, bias_field = options["system"].get_bias_field(options)
        if not bias_on:
            bias_field = None
    if bias_field is not None:
        b_mag_gauss, b_theta_rad, b_phi_rad = bias_field
        b_theta_deg = np.rad2deg(b_theta_rad)
        b_phi_deg = np.rad2deg(b_phi_rad)
    else:
        b_mag_gauss = options["bias_mag"]
        b_theta_deg = options["bias_theta"]
        b_phi_deg = options["bias_phi"]

    if spherical_deg:
        return b_mag_gauss, b_theta_deg, b_phi_deg
    else:
        return _spherical_deg_to_cart(b_mag_gauss, b_theta_deg, b_phi_deg)


# ============================================================================


def _spherical_deg_to_cart(b_mag_gauss, b_theta_deg, b_phi_deg):
    """Field vector in spherical polar degrees -> cartesian (gauss)

    Parameters
    ----------
    b_ag_gauss, b_theta_deg, b_phi_deg : float
        Field components in spherical polar (degrees)
    Returns
    -------
    b_x, b_y, b_z : tuple
        All floats in gauss, cartesian field components.
    """
    b_theta_rad = np.deg2rad(b_theta_deg)
    b_phi_rad = np.deg2rad(b_phi_deg)
    b_x_gauss = b_mag_gauss * np.sin(b_theta_rad) * np.cos(b_phi_rad)
    b_y_gauss = b_mag_gauss * np.sin(b_theta_rad) * np.sin(b_phi_rad)
    b_z_gauss = b_mag_gauss * np.cos(b_theta_rad)
    return b_x_gauss, b_y_gauss, b_z_gauss


# ============================================================================


def _define_output_dir(options):
    """
    Defines output_dir and data_dir in options.
    """
    if options["custom_output_dir"] is not None:
        output_dir = pathlib.PurePosixPath(
            _interpolate_option_str(str(options["custom_output_dir"]), options)
        )
    else:
        output_dir = pathlib.PurePosixPath(str(options["filepath"]))

    if options["custom_output_dir_prefix"] is not None:
        prefix = _interpolate_option_str(str(options["custom_output_dir_prefix"]), options)
    else:
        prefix = ""

    if options["custom_output_dir_suffix"] is not None:
        suffix = _interpolate_option_str(str(options["custom_output_dir_suffix"]), options)
    else:
        suffix = ""

    options["output_dir"] = output_dir.parent.joinpath(prefix + output_dir.stem + suffix)
    options["data_dir"] = options["output_dir"].joinpath("data")


# ============================================================================


def _interpolate_option_str(interp_str, options):
    """
    Interpolates any options between braces in interp_str.
    I.e. "{fit_backend}" -> f"{options['fit_backend']"
    (this is possibly possible directly through f-strings but I didn't want to
    play with fire)

    Arguments
    ---------
    interp_str : str
        String (possibly containing option names between braces) to be interpolated.
    options : dict
        Generic options dict holding all the user options.
    Returns
    -------
    interp_str : str
        String, now with interpolated values (option between braces).
    """

    # convert whitespace to underscore
    interp_str = interp_str.replace(" ", "_")
    if not ("{" in interp_str and "}" in interp_str):
        return interp_str
    pattern = r"\{(\w+)\}"  # all text between braces
    match_substr_lst = re.findall(pattern, interp_str)
    locs = []
    for match_substr in match_substr_lst:
        start_loc = interp_str.find(match_substr)
        end_loc = start_loc + len(match_substr) - 1  # both fence posts inclusive
        locs.append((start_loc, end_loc))

    # ok so now we have the option names (keys) to interpolate (match_substr_lst)
    # as well as the locations in the string of the same (locs)
    # first we convert those option names to their variables
    option_lst = []
    for option_name in match_substr_lst:
        try:
            option_lst.append(str(options[option_name]))
        except KeyError as e:
            warnings.warn(
                "\n"
                + "KeyError caught interpolating custom output_dir.\n"
                + f"You gave: {option_name}.\n"
                + "Avoiding this issue by placing 'option_name' in the dir instead. KeyError msg:"
                + f"{e}"
            )
            option_lst.append(option_name)

    # this block is a bit old-school, but does the trick...
    locs_passed = 0  # how many match locations we've passed
    new_str = []  # ok actually a lst but will convert back at end

    for i, c in enumerate(interp_str):
        # don't want braces in output
        if c in ["{", "}"]:
            continue

        #  'inside' a brace -> don't copy these chars
        if locs[locs_passed][0] <= i <= locs[locs_passed][1]:
            # when we've got to end of this brace location, copy in option
            if i == locs[locs_passed][1]:
                new_str.append(option_lst[locs_passed])
                locs_passed += 1
        else:
            new_str.append(c)

    return "".join(new_str)


# ============================================================================


def load_polygons(options):
    if options["polygon_nodes_path"]:
        options["polygon_nodes"] = [
            np.array(polygon)
            for polygon in qdmpy.shared.json2dict.json_to_dict(options["polygon_nodes_path"])[
                "nodes"
            ]
        ]
        options["polygons"] = [
            qdmpy.shared.polygon.Polygon(nodes[:, 0], nodes[:, 1])
            for nodes in options["polygon_nodes"]
        ]
    else:
        options["polygon_nodes"] = None
        options["polygons"] = None


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


# ============================================================================
