# -*- coding: utf-8 -*-
"""
This module holds tools for drift correction.
Does not depend on any of the rest of the qdmpy package.
Most of the functions are not documented, but the API is only 2 funcs:

Functions
---------
 - `qdmpy.driftcorrect.bnv.drift_correct_test`
 - `qdmpy.driftcorrect.drift_correct_measurement`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.driftcorrect.drift_correct_test": True,
    "qdmpy.driftcorrect.drift_correct_measurement": True,
}

# ============================================================================

import re
from rebin import rebin
import numpy as np
from warnings import warn
import os
from tqdm.autonotebook import tqdm
from skimage.registration import phase_cross_correlation as cross_correl
from skimage.transform import EuclideanTransform, warp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

# ============================================================================


def read_image(
    filepath,
    additional_bins=0,
    norm="div",
    rem_start_pts=1,
    rem_end_pts=0,
    ROI=None,
):
    with open(os.path.normpath(filepath), mode="r") as fid:
        raw_data = np.fromfile(fid, dtype=np.float32)[2:]

    # pl_image, sig, ref, sig_norm, sweep_list, roi
    return _reshape_raw(
        raw_data,
        filepath,
        additional_bins,
        norm,
        rem_start_pts,
        rem_end_pts,
        ROI,
    )


# ============================================================================


def read_metadata(filepath):
    with open(os.path.normpath(str(filepath) + "_metaSpool.txt"), "r") as fid:
        _ = fid.readline().rstrip().split("\t")
        rest_str = fid.read()
        matches = re.findall(
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


# ============================================================================


def _read_sweep_list(filepath):
    with open(os.path.normpath(str(filepath) + "_metaSpool.txt"), "r") as fid:
        sweep_str = fid.readline().rstrip().split("\t")
    return [float(i) for i in sweep_str]


# ============================================================================


def _check_start_end_rectangle(
    name, start_x, start_y, end_x, end_y, full_size_w, full_size_h
):
    if start_x >= end_x:
        warn(f"{name} Rectangle ends before it starts (in x), swapping them")
        start_x, end_x = end_x, start_x
    if start_y >= end_y:
        warn(f"{name} Rectangle ends before it starts (in y), swapping them")
        start_y, end_y = end_y, start_y
    if start_x >= full_size_w:
        warn(
            f"{name} Rectangle starts outside image (too large in x), setting to zero."
        )
        start_x = 0
    elif start_x < 0:
        warn(
            f"{name} Rectangle starts outside image (negative in x), setting to zero.."
        )
        start_x = 0

    if start_y >= full_size_h:
        warn(
            f"{name} Rectangle starts outside image (too large in y), setting to zero."
        )
        start_y = 0
    elif start_y < 0:
        warn(
            f"{name}  Rectangle starts outside image (negative in y), setting to zero."
        )
        start_y = 0

    if end_x >= full_size_w:
        warn(
            f"{name} Rectangle too big in x, cropping to image.\n"
            + f"Image dimensions (x,y): ({full_size_w},{full_size_h})"
        )
        end_x = full_size_w - 1
    if end_y >= full_size_h:
        warn(
            f"{name} Rectangle too big in y, cropping to image.\n"
            + f"Image dimensions (x,y): ({full_size_w},{full_size_h})"
        )
        end_y = full_size_h - 1

    return (start_x, start_y), (end_x, end_y)


# ============================================================================


def _define_roi(ROI, full_size_h, full_size_w):
    if ROI is None:
        roi = _define_area_roi(0, 0, full_size_w - 1, full_size_h - 1)
    else:
        start_x, start_y = ROI["start"]
        end_x, end_y = ROI["end"]
        roi = _define_area_roi(start_x, start_y, end_x, end_y)
    return roi


# ============================================================================


def _define_area_roi(start_x, start_y, end_x, end_y):
    x = np.linspace(start_x, end_x, end_x - start_x + 1, dtype=int)
    y = np.linspace(start_y, end_y, end_y - start_y + 1, dtype=int)
    xv, yv = np.meshgrid(x, y)
    return [
        yv,
        xv,
    ]  # arrays are indexed in image convention, e.g. sig[sweep_param, y, x]


# ============================================================================


def _remove_unwanted_data(
    image_rebinned,
    sweep_list,
    sig,
    ref,
    sig_norm,
    rem_start=1,
    rem_end=0,
    ROI=None,
):
    end = -rem_end if rem_end > 0 else None

    roi = _define_roi(ROI, *image_rebinned.shape[1:])

    if rem_start < 0:
        warn("remove_start_sweep must be >=0, setting to zero now.")
        rem_start = 0
    if rem_end < 0:
        warn("remove_end_sweep must be >=0, setting to zero now.")
        rem_end = 0

    pl_image = np.sum(image_rebinned, axis=0)

    return (
        pl_image,
        sig[rem_start:end].copy(),
        ref[rem_start:end].copy(),
        sig_norm[rem_start:end].copy(),
        np.asarray(sweep_list[rem_start:end]).copy(),
        roi,
    )


# ============================================================================


def crop_roi(seq, roi):
    if len(np.shape(seq)) == 2:
        return seq[roi[0], roi[1]].copy()
    else:
        return seq[:, roi[0], roi[1]].copy()


# ============================================================================


def _reshape_raw(
    raw_data,
    filepath,
    additional_bins=0,
    norm="div",
    rem_start_pts=1,
    rem_end_pts=0,
    ROI=None,
):
    sweep_list = _read_sweep_list(filepath)
    metadata = read_metadata(filepath)
    try:
        data_pts = len(sweep_list)
        image = np.reshape(
            raw_data,
            [
                data_pts,
                int(metadata["AOIHeight"]),
                int(metadata["AOIWidth"]),
            ],
        )
        used_ref = False
    except ValueError:
        data_pts = 2 * len(sweep_list)
        image = np.reshape(
            raw_data,
            [
                data_pts,
                int(metadata["AOIHeight"]),
                int(metadata["AOIWidth"]),
            ],
        )
        used_ref = True

    im = image.transpose([0, 2, 1]).copy()
    return _reshape_dataset(
        im,
        sweep_list,
        additional_bins,
        used_ref,
        norm,
        rem_start_pts,
        rem_end_pts,
        ROI,
    )


# ============================================================================


def _rebin_image(image, additional_bins=0, used_ref=True, norm="div"):
    if not additional_bins:
        image_rebinned = np.array(image)
    else:
        if additional_bins != 1 and additional_bins % 2:
            raise RuntimeError("The binning parameter needs to be a multiple of 2.")

        image_rebinned = np.array(
            rebin(
                image,
                factor=(1, additional_bins, additional_bins),
                func=np.mean,
            )
        )

    # define sig and ref differently if we're using a ref
    if used_ref:
        sig = image_rebinned[::2, :, :]
        ref = image_rebinned[1::2, :, :]
        if norm == "sub":
            sig_norm = 1 + (sig - ref) / (sig + ref)
        elif norm == "div":
            sig_norm = sig / ref
        elif norm == "true_sub":
            sig_norm = (sig - ref) / np.nanmax(sig - ref, axis=0)
        else:
            raise KeyError("bad normalisation option, use: ['sub', 'div', 'true_sub']")
    else:
        sig = image_rebinned
        ref = image_rebinned
        sig_norm = sig / np.max(sig, 0)

    return image_rebinned, sig, ref, sig_norm


# ============================================================================


def _reshape_dataset(
    image,
    sweep_list,
    additional_bins=0,
    used_ref=True,
    norm="div",
    rem_start_pts=1,
    rem_end_pts=0,
    ROI=None,
):
    image_rebinned, sig, ref, sig_norm = _rebin_image(
        image, additional_bins, used_ref, norm
    )
    try:
        size_h, size_w = image_rebinned.shape[1:]
    except Exception:
        size_h, size_w = image_rebinned.shape

    # check options to ensure ROI is in correct format (now we have image size)
    if ROI != None:
        ROI["start"], ROI["end"] = _check_start_end_rectangle(
            "ROI", *ROI["start"], *ROI["end"], size_w, size_h
        )  # opposite convention here, [x, y]

    (
        pl_image,
        sig,
        ref,
        sig_norm,
        sweep_list,
        roi,
    ) = _remove_unwanted_data(
        image_rebinned,
        sweep_list,
        sig,
        ref,
        sig_norm,
        rem_start_pts,
        rem_end_pts,
        ROI,
    )  # also cuts sig etc. down to ROI

    return (
        pl_image,
        sig,
        ref,
        sig_norm,
        sweep_list,
        roi,
    )


# ============================================================================


def _add_colorbar(im, fig, ax, aspect=20, pad_fraction=1, **kwargs):
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1.0 / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = fig.colorbar(im, cax=cax, **kwargs)
    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.linewidth = 0.5
    return cbar


def plot_image_on_ax(
    fig, ax, image_data, title="", c_map="gray", c_range=None, c_label=""
):
    if c_range is None:
        mx = np.nanmax(np.abs([np.nanmin(image_data), np.nanmax(image_data)]))
        c_range = [-mx, mx]
    im = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1])

    ax.set_title(title)

    cbar = _add_colorbar(im, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    return fig, ax


# ============================================================================


def drift_correct_single(refr_pl, move_pl, target_pl):
    # refr_pl & move_pl should probably be cropped to some feature
    # then the shift is applied to target_pl
    with np.errstate(all="ignore"):
        shift_calc = cross_correl(refr_pl, move_pl, normalization=None)[0]
    tform = EuclideanTransform(translation=[-shift_calc[1], -shift_calc[0]])
    return warp(target_pl, tform, mode="edge"), shift_calc


def drift_correct_stack(refr_pl, move_pl, move_sig_norm):
    # calc's shift in 'mov_pl' to match 'refr_pl' (image regulation) via cross-corr method
    # then applies that to move_sig_norm.
    with np.errstate(all="ignore"):
        shift_calc = cross_correl(refr_pl, move_pl, normalization=None)[0]
    tform = EuclideanTransform(translation=[-shift_calc[1], -shift_calc[0]])

    reg_sig_norm = np.empty(move_sig_norm.shape)
    for i in range(move_sig_norm.shape[0]):
        reg_sig_norm[i, :, :] = warp(move_sig_norm[i, :, :], tform, mode="edge")
    return reg_sig_norm, shift_calc


# ============================================================================


def read_and_drift_correct(
    base_path,
    stub,
    image_seq,
    additional_bins=0,
    rem_start_pts=0,
    rem_end_pts=0,
    ROI=None,  # Used for image registration only (no crop on output)
    mask=None,  # True where you want to include image in accum.
):
    first = True
    for i in tqdm(image_seq):
        _, sig, ref, _, _, roi = read_image(
            base_path + stub(i),
            additional_bins,
            "div",  # norm - we don't care what it is
            rem_start_pts,
            rem_end_pts,
            ROI,
        )
        if first:
            first = False
            refr_pl = np.sum(sig, axis=0)
            accum_sig = sig.copy()
            accum_ref = ref.copy()
            prev_sig = sig.copy()
            prev_ref = ref.copy()
            continue

        this_sig = sig - prev_sig
        this_ref = ref - prev_ref
        this_pl = np.sum(this_sig, axis=0)
        prev_sig = sig
        prev_ref = ref

        this_sig, shift_calc_stack = drift_correct_stack(
            crop_roi(refr_pl, roi), crop_roi(this_pl, roi), this_sig
        )
        this_ref, shift_calc_stack = drift_correct_stack(
            crop_roi(refr_pl, roi), crop_roi(this_pl, roi), this_ref
        )

        if mask is None or mask[i - image_seq[0]]:
            accum_sig += this_sig
            accum_ref += this_ref

    return accum_sig, accum_ref


# ============================================================================


def drift_correct_measurement(
    directory,
    start_num,
    end_num,
    stub,
    output_file,
    feature_roi=None,
    additional_bins=0,  # TODO  need to save in metadata?? for scalebars
    rem_start_pts=0,
    rem_end_pts=0,
    image_nums_mask=None,
    reader_mode=None,
    writer_mode=None,
):
    """

    Arguments
    ---------
    directory : str
        Path to directory that contains all the measurements.
    start_num : int
        Image/measurement to start accumulating from.
    end_num : int
        Image/measurement to end accumulation.
    stub : function
        Function that takes image num and returns path to that measurement
        from directory. I.e. directory + stub(X) for filepath ass. with 'X'
    output_file : str
        Output will be stored in directory + output file
    feature_roi : dict
        ROI to use for drift-correction.
        Form: {"start": [X, Y], "end": [X, Y]}
    additional_bins : int
        Binning to do before drift-correction. Saved in this form also.
    rem_start_pts : int
        Points in freq/tau dimension to remove from sweep at start.
    rem_end_pts : int
        Points in freq/tau dimension to remove from sweep at end.
    image_nums_mask : Sequence of bools
        For each val in [start_sum, end_sum],
        do we accumulate this image or not? True = keep that idx.
    reader_mode : str or None (default)
        Which type of reader function to apply.
        Default (only implemented) is the old labview reader.
    writer_mode : str or None (default)
        Which type of reader function to apply to output.
        Default (only implemented) is to compat with the old labview reader.
    """

    if reader_mode is None or reader_mode == "labview":
        sig, ref = read_and_drift_correct(
            directory,
            stub,
            list(range(start_num, end_num + 1)),
            ROI=feature_roi,
            mask=image_nums_mask,
        )
    else:
        raise ValueError(
            "Only None/default/labview reader_mode is defined at the moment."
        )

    if writer_mode is None or writer_mode == "labview":
        # save output. bit slow but haven't worked out a vectorised version'
        s = sig.transpose([0, 2, 1])
        r = ref.transpose([0, 2, 1])

        output = []
        for f in range(s.shape[0]):
            output.append(s[f, ::])
            output.append(r[f, ::])
        output_array = np.array(output).flatten()

        with open(output_file, "wb") as fid:
            np.array([0, 0]).astype(np.float32).tofile(fid)
            output_array.astype(np.float32).tofile(fid)

        # read metaspool, and then write with updated binning.
        lines = []
        with open(directory + stub(start_num) + "_metaSpool.txt", "r") as fid:
            for line in fid:
                if line.startswith("Binning:"):
                    old_binning = int(re.match(r"Binning: (\d+)\n", line)[1])
                    new_binning = (
                        old_binning
                        if not additional_bins
                        else old_binning * additional_bins
                    )
                    lines.append(f"Binning: {new_binning}\n")
                else:
                    lines.append(line)

        with open(output_file + "_metaSpool.txt", "w") as fid:
            for line in lines:
                fid.write(line)
    else:
        raise ValueError(
            "Only None/default/laview writer_mode is defined at the moment."
        )


# ============================================================================


def drift_correct_test(
    directory,
    start_num,
    end_num,
    comparison_nums,
    stub,
    feature_roi=None,
    additional_bins=0,
    reader_mode=None,
):
    """

    Arguments
    ---------
    directory : str
        Path to directory that contains all the measurements.
    start_num : int
        Image/measurement to start accumulating from.
        (Also the reference frame)
    end_num : int
        Image/measurement to end accumulation.
    comparison_nums : list of ints
        List of image/measurment nums to compare drift calc on.
    stub : function
        Function that takes image num and returns path to that measurement
        from directory. I.e. directory + stub(X) for filepath ass. with 'X'
    feature_roi : dict
        ROI to use for drift-correction.
        Form: {"start": [X, Y], "end": [X, Y]}
    additional_bins : int
        Binning to do before drift-correction. Saved in this form also.
    reader_mode : str or None (default)
        Which type of reader function to apply.
        Default (only implemented) is the old labview reader.

    Returns
    -------
    crop_fig, crop_axs
        Matplotlib figure and axes, for further editing/saving etc.
    """

    # prep fig
    nrows = len(comparison_nums)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 4 * nrows))

    # read in all frames, only pull out (non-accum) pl frames we want to compare
    raw_comp_frames = []
    if reader_mode is None or reader_mode == "labview":
        image_seq = list(range(start_num, end_num + 1))
        first = True
        for i in image_seq:
            accum_pl, _, _, _, _, roi = read_image(
                directory + stub(i),
                additional_bins,
                "div",  # norm - we don't care what it is
                0,
                0,
                feature_roi,
            )
            if first:
                first = False
                prev_accum_pl = accum_pl.copy()
                if i in comparison_nums:
                    raw_comp_frames.append(accum_pl)
                continue
            this_pl = accum_pl - prev_accum_pl
            prev_accum_pl = accum_pl
            if i in comparison_nums:
                raw_comp_frames.append(this_pl)
            if i > max(comparison_nums):
                break
    else:
        raise ValueError(
            "Only None/default/labview reader_mode is defined at the moment."
        )

    # plot cropped sig frames in left column
    for i, frame, ax in zip(comparison_nums, raw_comp_frames, axs[:, 0]):
        plot_image_on_ax(
            fig,
            ax,
            crop_roi(frame, roi),
            title=f"raw   {i}",
            c_range=None,
            c_label="Counts",
        )

    # do cross-corr on comparison frames
    refr_frame = raw_comp_frames[0]
    corrected_frames = [
        refr_frame,
    ]
    shift_calcs = [
        [0, 0],
    ]
    for frame in raw_comp_frames[1:]:
        corrected_frame, shift_calc = drift_correct_single(
            crop_roi(refr_frame, roi), crop_roi(frame, roi), frame
        )
        corrected_frames.append(corrected_frame)
        shift_calcs.append(shift_calc)

    # plot cropped corrected frames in right column
    for i, frame, shift_calc, ax in zip(
        comparison_nums, corrected_frames, shift_calcs, axs[:, 1]
    ):
        plot_image_on_ax(
            fig,
            ax,
            crop_roi(frame, roi),
            title=f"shftd {i}: {shift_calc}",
            c_range=None,
            c_label="Counts",
        )

    return fig, axs
