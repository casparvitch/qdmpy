# -*- coding: utf-8 -*-
"""Interface to mag simulations.

# FIXME needs better documentation eh!

Functions
---------
 - `qdmpy.magsim.interface._plot_image_on_ax`
 - `qdmpy.magsim.interface._add_cbar`

Classes
-------
 - `qdmpy.magsim.interface.MagSim`
 - `qdmpy.magsim.interface.SandboxMagSim`
 - `qdmpy.magsim.interface.ComparisonMagSim`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.magsim.interface.MagSim": True,
    "qdmpy.magsim.interface.SandboxMagSim": True,
    "qdmpy.magsim.interface.ComparisonMagSim": True,
    "qdmpy.magsim.interface._plot_image_on_ax": True,
    "qdmpy.magsim.interface._add_cbar": True,
}
# ============================================================================

import numpy as np
import numpy.linalg as LA  # noqa: N812
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from collections import defaultdict as dd
from copy import copy

from pyfftw.interfaces import numpy_fft
from scipy.ndimage import gaussian_filter

# ============================================================================

import qdmpy.shared.polygon
import qdmpy.shared.json2dict

# ============================================================================


class MagSim:
    polygon_nodes = None
    mag = None
    template_polygon_nodes = None
    bfield = None
    standoff = None
    magnetizations_lst = None
    unit_vectors_lst = None

    def _load_polys(self, polys, check_size=False):
        """ polys: either path to json, or dict containing 'nodes' key. """
        if polys is not None:
            if isinstance(polys, dict):
                if check_size and "image_size" in polys:
                    if self.ny != polys["image_size"][0] or self.nx != polys["image_size"][1]:
                        raise RuntimeError(
                            "Image size polygons were defined on as passed to add_polygons does "
                            + "not match this MagSim's mesh."
                        )
                return [np.array(p) for p in polys["nodes"]]
            elif isinstance(polys, str):
                return [np.array(p) for p in qdmpy.shared.json2dict.json_to_dict(polys)["nodes"]]
            else:
                raise TypeError("polygons argument was not a dict or string?")
        return None

    def _load_image(self, image):
        if image is not None:
            if isinstance(image, np.ndarray):
                return image
            elif isinstance(image, str):
                return np.loadtxt(image)
            else:
                raise TypeError("image argument must be an np.ndarray or string?")
        return None

    def _polygon_gui(self, polygon_nodes=None, mean_plus_minus=None, base_scale=1.25, **kwargs):
        fig, ax = plt.subplots()

        if mean_plus_minus is not None and isinstance(mean_plus_minus, (float, int)):
            mean = np.mean(self.base_image)
            vmin, vmax = mean - mean_plus_minus, mean + mean_plus_minus
        else:
            vmin = vmax = None

        img = ax.imshow(self.base_image, aspect="equal", cmap="bwr", vmin=vmin, vmax=vmax)
        ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
        ax.tick_params(
            axis="y",  # changes apply to the y-axis
            which="both",  # both major and minor ticks are affected
            left=False,
            right=False,
            labelleft=False,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax)
        ax.set_title("Select polygons")

        psw = qdmpy.shared.polygon.PolygonSelectionWidget(ax, base_scale=base_scale, style=kwargs)

        if polygon_nodes is not None:
            psw.load_nodes(polygon_nodes)

        plt.show(block=True)
        psw.disconnect()

        pgons = psw.get_polygons_lst()
        if len(pgons) < 1:
            raise RuntimeError("You didn't define any polygons")

        pgon_lst = [pgon.get_nodes() for pgon in pgons if np.shape(pgon.get_nodes())[0] > 2]
        output_dict = {"nodes": pgon_lst, "image_shape": (self.ny, self.nx)}

        return output_dict

    def add_polygons(self, polys=None):
        """polygons is dict (polygons directly) or str (path to)"""
        self.polygon_nodes = self._load_polys(polys, check_size=True)

    def define_magnets(self, magnetizations, unit_vectors):
        """
        magnetizations: int/float if the same for all polygons, or an iterable of len(polygon_nodes)
            -> in units of mu_b / nm^2 (or mu_b / PX^2 for SandboxMagSim)
        unit_vectors: 3-iterable if the same for all polygons (cartesian coords),
            or an iterable of len(polygon_nodes) each element a 3-iterable
        """
        # todo: do we want to be able to add _noise_ here too? / other imperfections?
        if isinstance(magnetizations, (float, int)):
            self.magnetizations_lst = [magnetizations for m, _ in enumerate(self.polygon_nodes)]
        else:
            if len(magnetizations) != len(self.polygon_nodes):
                raise ValueError("Number of magnetizations does not match number of magnets.")
            self.magnetizations_lst = magnetizations

        if isinstance(unit_vectors, (np.ndarray, list, tuple)):
            if len(np.shape(unit_vectors)) == 1:
                if len(unit_vectors) == 3:
                    uv_abs = LA.norm(unit_vectors)
                    self.unit_vectors_lst = [
                        tuple(np.array(unit_vectors) / uv_abs)
                        for m, _ in enumerate(self.polygon_nodes)
                    ]
                else:
                    raise RuntimeError("I don't recognise that shape of unit_vectors.")
            else:
                # ensure unit vectors
                self.unit_vectors_lst = [tuple(np.array(uv) / LA.norm(uv)) for uv in unit_vectors]
        else:
            raise TypeError("unit_vectors wrong type :(")

        if len(self.magnetizations_lst) != len(self.unit_vectors_lst):
            raise RuntimeError("magnetizations_lst not the same length as unit_vectors_lst. :(")

        # now construct mag
        self.mag = dd(lambda: np.zeros((self.ny, self.nx)))
        grid_y, grid_x = np.meshgrid(range(self.ny), range(self.nx), indexing="ij")

        for i, p in enumerate(self.polygon_nodes):
            polygon = qdmpy.shared.polygon.Polygon(p[:, 0], p[:, 1])
            in_or_out = polygon.is_inside(grid_y, grid_x)
            self.mag[self.unit_vectors_lst[i]][in_or_out >= 0] = self.magnetizations_lst[i]

    def run(self, standoff, resolution=None, pad_mode="mean", pad_factor=2, k_vector_epsilon=1e-6):
        """standoff: units of pixels for sandbox, units of m for comparison."""
        self.standoff = standoff
        # in future could be generalised to a range of standoffs
        # e.g. if we wanted to average over an nv-depth distribution that would be easy

        # get shape so we can define kvecs
        dummy_img, _ = qdmpy.shared.fourier.pad_image(
            np.empty(np.shape(self.mag[self.unit_vectors_lst[0]])), pad_mode, pad_factor
        )
        ky, kx, k = qdmpy.shared.fourier.define_k_vectors(
            dummy_img.shape, self.pixel_size, k_vector_epsilon
        )

        # opposite sign on the exponential/standoff as we're upward-propagating.
        d_matrix = np.exp(
            -1 * k * standoff
        ) * qdmpy.shared.fourier.define_magnetization_transformation(ky, kx, k, standoff=False)

        d_matrix = qdmpy.shared.fourier.set_naninf_to_zero(d_matrix)

        self.bfield = [
            np.zeros((self.ny, self.nx)),
            np.zeros((self.ny, self.nx)),
            np.zeros((self.ny, self.nx)),
        ]

        # convert to A from mu_b / nm^2 magnetization units
        m_scale = 1 / qdmpy.shared.fourier.MAG_UNIT_CONV

        for uv in self.unit_vectors_lst:
            mx, my, mz = (
                self.mag[uv] * uv[0] * m_scale,
                self.mag[uv] * uv[1] * m_scale,
                self.mag[uv] * uv[2] * m_scale,
            )

            mx_pad, p = qdmpy.shared.fourier.pad_image(mx, pad_mode, pad_factor)
            my_pad, _ = qdmpy.shared.fourier.pad_image(my, pad_mode, pad_factor)
            mz_pad, _ = qdmpy.shared.fourier.pad_image(mz, pad_mode, pad_factor)

            fft_mx = numpy_fft.fftshift(numpy_fft.fft2(mx_pad))
            fft_my = numpy_fft.fftshift(numpy_fft.fft2(my_pad))
            fft_mz = numpy_fft.fftshift(numpy_fft.fft2(mz_pad))

            fft_mag_vec = np.stack((fft_mx, fft_my, fft_mz))

            fft_b_vec = np.einsum(
                "ij...,j...->i...", d_matrix, fft_mag_vec
            )  # matrix mul b = d * m (d and m are stacked in last 2 dimensions)

            # take back to real space, unpad & convert bfield to Gauss (from Tesla)
            self.bfield[0] += (
                qdmpy.shared.fourier.unpad_image(
                    numpy_fft.ifft2(numpy_fft.ifftshift(fft_b_vec[0])).real, p
                )
                * 1e4
            )
            self.bfield[1] += (
                qdmpy.shared.fourier.unpad_image(
                    numpy_fft.ifft2(numpy_fft.ifftshift(fft_b_vec[1])).real, p
                )
                * 1e4
            )
            self.bfield[2] += (
                qdmpy.shared.fourier.unpad_image(
                    numpy_fft.ifft2(numpy_fft.ifftshift(fft_b_vec[2])).real, p
                )
                * 1e4
            )

        # for resolution convolve with width=resolution gaussian?
        # yep but width = sigma = in units of pixels. {so do some maths eh}
        # just do it after the fft I think.
        # add noise too? -> add to magnetisation or what?
        if resolution is not None:
            # would be faster to do while in k-space already above.
            sigma = resolution / self.pixel_size
            # in-place
            gaussian_filter(self.bfield[0], sigma, output=self.bfield[0])
            gaussian_filter(self.bfield[1], sigma, output=self.bfield[1])
            gaussian_filter(self.bfield[2], sigma, output=self.bfield[2])

    def _scale_for_fft(self, ars):
        """Norm arrays so fft doesn't freak out"""
        mx = np.max(np.abs(ars))
        return [ar / mx for ar in ars], mx

    # image outputs: bfield and magnetization
    def get_bfield_im(self, projection=(0, 0, 1)):
        if self.bfield is None:
            raise AttributeError("simulation not run yet.")
        # access bfield_sensor_plane, project onto projection

        # reshape bfield to be [bx, by, bz] for each pixel of image (as 1 stacked ndarray)
        bfield_reshaped = np.stack(self.bfield, axis=-1)

        proj_vec = np.array(projection)

        return np.apply_along_axis(lambda bvec: np.dot(proj_vec, bvec), -1, bfield_reshaped)

    def get_magnetization_im(self, unit_vector):
        if self.mag is None:
            raise AttributeError("magnetization not defined yet.")
        return self.mag[unit_vector]

    def plot_magsim_magnetization(
        self, unit_vector, annotate_polygons=True, polygon_patch_params=None
    ):
        fig, ax = plt.subplots()
        mag_image = self.get_magnetization_im(unit_vector)
        mx = np.max(np.abs(mag_image))
        c_range = (-mx, mx)
        if annotate_polygons:
            polys = self.polygon_nodes
        else:
            polys = None
        _plot_image_on_ax(
            fig,
            ax,
            mag_image,
            str(unit_vector),
            "PuOr",
            c_range,
            r"M ($\mu_B$ nm$^{-2}$)",
            polygon_nodes=polys,
            polygon_patch_params=polygon_patch_params,
        )
        return fig, ax

    def plot_magsim_magnetizations(self, annotate_polygons=True, polygon_patch_params=None):
        # use single colorbar, different plots
        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/colorbar_placement.html
        # calculate c_range smartly.
        if self.magnetizations_lst is None:
            raise AttributeError("no magnetizations_lst found, define it first aye.")

        unique_uvs = dd(list)
        for i, uv in enumerate(self.unit_vectors_lst):
            unique_uvs[uv].append(i)

        mag_images = [self.get_magnetization_im(uv) for uv in unique_uvs]

        mx = max([np.max(mag) for mag in mag_images])
        c_range = (-mx, mx)

        figsize = mpl.rcParams["figure.figsize"].copy()
        nrows = 1
        ncols = len(unique_uvs)
        figsize[0] *= nrows

        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

        for i, (mag, uvs) in enumerate(zip(mag_images, unique_uvs)):
            if annotate_polygons:
                polys = [self.polygon_nodes[j] for j in unique_uvs[uvs]]
            else:
                polys = None

            title = ", ".join([str(self.unit_vectors_lst[uv]) for uv in unique_uvs[uvs]])
            _plot_image_on_ax(
                fig,
                axs[i] if isinstance(axs, np.ndarray) else axs,
                mag,
                title,
                "PuOr",
                c_range,
                self._get_mag_unit_str(),
                polygon_nodes=polys,
                polygon_patch_params=polygon_patch_params,
            )

        return fig, axs

    def plot_magsim_bfield_at_nvs(
        self,
        projection=(0, 0, 1),
        annotate_polygons=True,
        polygon_patch_params=None,
        c_map="bwr",
        strict_range=None,
        c_label=None,
    ):
        if self.bfield is None:
            raise AttributeError("No bfield found: no simulation run.")

        if strict_range is not None:
            c_range = strict_range
        else:
            c_range = (np.nanmin(self.bfield), np.nanmax(self.bfield))

        polys = None if annotate_polygons is None else self.polygon_nodes

        fig, ax = plt.subplots()
        proj_name = f"({projection[0]:.2f},{projection[1]:.2f},{projection[2]:.2f})"
        c_label_ = f"B . {proj_name}, (G)" if c_label is None else c_label
        _plot_image_on_ax(
            fig,
            ax,
            self.get_bfield_im(projection),
            f"B . {proj_name} at z = {self.standoff}{self._get_dist_unit_str()}",
            c_map,
            c_range,
            c_label_,
            polygon_nodes=polys,
            polygon_patch_params=polygon_patch_params,
        )
        return fig, ax

    def _get_mag_unit_str(self):
        return r"M ($\mu_B$ nm$^{-2}$)"

    def _get_dist_scaling(self):
        # nm = 1e-9
        return 1e-9

    def _get_dist_unit_str(self):
        return "m"


# standoff in units of pixels!!
# {ensure magnetization is in units of mu_b / pixel !!}
class SandboxMagSim(MagSim):
    def __init__(self, mesh_shape, fov_dims):
        """Image conventions: first index is height."""
        self.ny, self.nx = mesh_shape
        self.fov_dims = fov_dims
        self.base_image = np.full(mesh_shape, np.nan)
        pxl_y = fov_dims[0] / self.ny
        pxl_x = fov_dims[1] / self.nx
        if pxl_y != pxl_x:
            raise ValueError("fov_dims ratio height:width does not match mesh height:width ratio.")
        self.pixel_size = pxl_y

    def add_template_polygons(self, polygons=None):
        """polygons takes precedence."""
        self.template_polygon_nodes = self._load_polys(polygons, check_size=True)

    def adjust_template(self, output_path=None, mean_plus_minus=None, **kwargs):
        if self.template_polygon_nodes is None:
            raise AttributeError("Add template polygons before adjusting.")
        pgon_dict = self._polygon_gui(
            polygon_nodes=self.template_polygon_nodes, mean_plus_minus=mean_plus_minus, **kwargs
        )
        if output_path is not None:
            qdmpy.shared.json2dict.dict_to_json(pgon_dict, output_path)

        self.template_polygon_nodes = [np.array(p) for p in pgon_dict["nodes"]]

    def set_template_as_polygons(self):
        if self.template_polygon_nodes is None:
            raise AttributeError("No template set.")
        self.polygon_nodes = self.template_polygon_nodes


# define FOV size (height, width), standoff height
# resolution: increase size of polygon via this (grab it's image size)
# -> user cannot scroll etc., define to match some other/bnv image
# would be best if image is plane-subtracted
class ComparisonMagSim(MagSim):
    unscaled_polygon_nodes = None

    def __init__(
        self,
        image,  # array-like (image directly) or string (path to)
        fov_dims,  # (the height, width of the image in m)
    ):
        if fov_dims is None:
            raise ValueError("You need to supply fov_dims (the height, width of the image in m).")
        if (
            not isinstance(fov_dims, (tuple, list, np.ndarray))
            or len(fov_dims) != 2
            or not isinstance(fov_dims[0], (int, float))
            or not isinstance(fov_dims[1], (int, float))
        ):
            raise TypeError("fov_dims needs to be length 2 array-like of int/floats")

        # check for path etc. here
        self.base_image = self._load_image(image)

        self.ny, self.nx = self.base_image.shape
        pxl_y = fov_dims[0] / self.ny
        pxl_x = fov_dims[1] / self.nx
        if pxl_y != pxl_x:
            raise ValueError(
                "fov_dims ratio height:width does not match image height:width ratio."
            )
        self.pixel_size = pxl_y

    def rescale(self, factor):
        if self.polygon_nodes is None:
            raise RuntimeError("Add/define polygon_nodes before rescaling.")

        if self.unscaled_polygon_nodes is None:
            self.unscaled_polygon_nodes = copy(self.polygon_nodes)

        self.ny *= factor
        self.nx *= factor
        self.pixel_size *= 1 / factor
        for polygon in self.polygon_nodes:
            for node in polygon:
                node[0] *= factor
                node[1] *= factor

    def select_polygons(
        self, polygon_nodes=None, output_path=None, mean_plus_minus=None, **kwargs
    ):
        pgon_dict = self._polygon_gui(
            polygon_nodes=polygon_nodes, mean_plus_minus=mean_plus_minus, **kwargs
        )
        if output_path is not None:
            qdmpy.shared.json2dict.dict_to_json(pgon_dict, output_path)

        self.polygon_nodes = [np.array(p) for p in pgon_dict["nodes"]]

    def plot_comparison(
        self,
        projection=(0, 0, 1),
        strict_range=None,
        annotate_polygons=True,
        c_label_meas=None,
        c_label_sim=None,
    ):
        # show:
        # - source image
        # - simulated field
        # (needs cmap/crange arguments/options, same for both images.)

        if self.bfield is None:
            raise AttributeError("No bfield found: no simulation run.")

        bfield_proj = self.get_bfield_im(projection)

        if strict_range is not None:
            c_range = strict_range
        else:
            mn = min((np.nanmin(bfield_proj), np.nanmin(self.base_image)))
            mx = max((np.nanmax(bfield_proj), np.nanmax(self.base_image)))
            c_range = (mn, mx)

        c_label_meas_ = "B (G)" if c_label_meas is None else c_label_meas

        proj_name = f"({projection[0]:.2f},{projection[1]:.2f},{projection[2]:.2f})"
        c_label_sim_ = f"B . {proj_name}, (G)" if c_label_sim is None else c_label_sim

        if annotate_polygons is None:
            unscaled_polys = None
            scaled_polys = None
        else:
            unscaled_polys = (
                self.polygon_nodes
                if self.unscaled_polygon_nodes is None
                else self.unscaled_polygon_nodes
            )
            scaled_polys = self.polygon_nodes

        figsize = mpl.rcParams["figure.figsize"].copy()
        nrows = 1
        ncols = 2
        figsize[0] *= nrows

        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

        _plot_image_on_ax(
            fig,
            axs[0],
            self.base_image,
            "measurement",
            "bwr",
            c_range,
            c_label_meas_,
            polygon_nodes=unscaled_polys,
            polygon_patch_params=None,
        )
        _plot_image_on_ax(
            fig,
            axs[1],
            bfield_proj,
            "simulated",
            "bwr",
            c_range,
            c_label_sim_,
            polygon_nodes=scaled_polys,
            polygon_patch_params=None,
        )

        return fig, axs


# below: adjusted versions of fns in qdmpy.plot.common.
def _plot_image_on_ax(
    fig,
    ax,
    image_data,
    title,
    c_map,
    c_range,
    c_label,
    polygon_nodes=None,
    polygon_patch_params=None,
):

    im = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1], aspect="equal")

    ax.set_title(title)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    cbar = _add_cbar(im, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    if polygon_nodes is not None:
        if polygon_patch_params is None:
            polygon_patch_params = {
                "facecolor": None,
                "edgecolor": "k",
                "linestyle": "dashed",
                "fill": False,
            }
        for p in polygon_nodes:
            # polygons reversed to (x,y) indexing for patch
            ax.add_patch(
                matplotlib.patches.Polygon(
                    np.dstack((p[:, 1], p[:, 0]))[0], **polygon_patch_params
                )
            )

    return fig, ax


def _add_cbar(im, fig, ax, aspect=20, pad_fraction=1, **kwargs):
    """
    Adds a colorbar to matplotlib axis

    Arguments
    ---------
    im : image as returned by ax.imshow
    fig : matplotlib Figure object
    ax : matplotlib Axis object

    Returns
    -------
    cbar : matplotlib colorbar object

    Optional Arguments
    ------------------
    aspect : int
        Reciprocal of aspect ratio passed to new colorbar axis width. Default: 20.
    pad_fraction : int
        Fraction of new colorbar axis width to pad from image. Default: 1.

    **kwargs : other keyword arguments
        Passed to fig.colorbar.

    """
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


# def _to_nested_np_array(list):
#     return np.array([np.array(i) for i in list])


# def _recursive_nparray(obj):
#     if not isinstance(obj[0], (list, tuple, np.ndarray)):
#         return np.array([np.array(o) for o in obj])
#     return np.array([_recursive_nparray(o) for o in obj])
