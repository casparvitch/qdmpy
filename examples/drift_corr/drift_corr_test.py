import re
from rebin import rebin
import numpy as np
from warnings import warn
import os
from skimage.registration import phase_cross_correlation as cross_correl
from skimage.transform import EuclideanTransform, warp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib as mpl

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
            raise RuntimeError(
                "The binning parameter needs to be a multiple of 2."
            )

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
            raise KeyError(
                "bad normalisation option, use: ['sub', 'div', 'true_sub']"
            )
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

def plot_image_on_ax(fig, ax, image_data, title="", c_map="gray", c_range=None, c_label=""):
    if c_range is None:
        mx = np.nanmax(np.abs([np.nanmin(image_data), np.nanmax(image_data)]))
        c_range = [-mx, mx]
    im = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1])

    ax.set_title(title)

    cbar = _add_colorbar(im, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    return fig, ax

# ============================================================================

def drift_correct(refr_pl, move_pl, target_pl):
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


exec(open("./TEST_DATA_PATH.py").read())

base_path = TEST_DATA_PATH + "drift_corr_test/"
stub = lambda x: f"ODMR - CW_{x}"
imns = [9, 10, 11, 69]

ROI = {"start": [150, 100], "end": [400, 350]}

(refr_pl, _, _, refr_sig_norm, sweep_list, roi) = read_image(
    base_path + stub(imns[0]), additional_bins=0, norm="div", ROI=ROI
)

(move_pl, _, _, move_sig_norm, sweep_list, roi) = read_image(
    base_path + stub(imns[1]), additional_bins=0, norm="div", ROI=ROI
)

crop_refr_pl = crop_roi(refr_pl, roi)
crop_move_pl = crop_roi(move_pl, roi)

crop_regr_pl, shift_calc_pl = drift_correct(crop_refr_pl, crop_move_pl, crop_move_pl)
print(shift_calc_pl)
tfig, taxs = plt.subplots(ncols=3, figsize=(14,4), sharex=True, sharey=True)
plot_image_on_ax(tfig, taxs[0], crop_refr_pl)
plot_image_on_ax(tfig, taxs[1], crop_move_pl)
plot_image_on_ax(tfig, taxs[2], crop_regr_pl)

fig, ax = plt.subplots()
plot_image_on_ax(fig, ax, crop_roi(refr_pl, roi), title="ROI")

sig_norms = [refr_sig_norm,]
old_pls = [refr_pl,]
new_pls = [refr_pl,]
for i in imns[1:]:
    move_pl, _, _, sig_norm, _, _ = read_image(base_path + stub(i))
    
    old_pls.append(move_pl)
    regr_pl, shift_calc_pl = drift_correct(crop_roi(refr_pl, roi), crop_roi(move_pl, roi), move_pl)
    new_pls.append(regr_pl)
    stack, shift_calc_stack = drift_correct_stack(crop_roi(refr_pl, roi), crop_roi(move_pl, roi), sig_norm)
    sig_norms.append(stack)
    print(shift_calc_pl, shift_calc_stack)
    
fig, axs = plt.subplots(ncols=len(imns), nrows=2, figsize=(16,12), sharex=True, sharey=True)
for i, f in enumerate(imns):
    plot_image_on_ax(fig, axs[0, i], old_pls[i], title=f"old {f}")
    plot_image_on_ax(fig, axs[1, i], new_pls[i], title=f"new {f}")

# now crop with: crop_roi(seq, roi) as required

# NOTE that all works, but doesn't correctly handle subtracting off previous image frame -> bit tricky eh!

plt.show()
