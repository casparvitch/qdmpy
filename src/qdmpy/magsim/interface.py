# -*- coding: utf-8 -*-
"""Interface to mag simulations


Functions
---------
 - `qdmpy.magsim.interface._plot_image_on_ax`
 - `qdmpy.magsim.interface._add_cbar`

Classes
-------
 - `qdmpy.magsim.interface.MagSim`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.magsim.interface.MagSim": True,
    "qdmpy.magsim.interface._plot_image_on_ax": True,
    "qdmpy.magsim.interface._add_cbar": True,
}
# ============================================================================

import numpy as np
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from collections import defaultdict as dd

from pyfftw.interfaces import numpy_fft


# ============================================================================

import qdmpy.shared.polygon
import qdmpy.shared.json2dict

# ============================================================================


class MagSim:
    polygon_nodes = None
    mag = None

    # TODO
    # define FOV size (height, width), standoff height
    # resolution: increase size of polygon via this (grab it's image size)
    # -> user cannot scroll etc., define to match some other/bnv image

    # add another 'mode': just draw some polygon,
    # define FOV, standoff etc. on some ny,nx mesh.
    # --> define two MagSim sub-classes? Sandbox, Comparison

    # gaussian filter to blur (set sigma)

    def __init__(self, ny, nx):
        """Image conventions: first index is height."""
        self.ny, self.nx = ny, nx
        self.base_image = np.full((ny, nx), np.nan)

    def add_template_polygons(self, path=None, polys=None):
        """polys takes precedence."""
        if path is None and polys is None:
            return
        if polys is not None:
            self.template_polygon_nodes = polys["nodes"]
        else:
            self.template_polygon_nodes = qdmpy.shared.json2dict.json_to_dict(path)["nodes"]

    def fix_template(self, output_path=None, **kwargs):

        fig, ax = plt.subplots(constrained_layout=True)
        img = ax.imshow(self.base_image, aspect="equal", cmap="bwr")
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
        ax.set_title("Select polygons to exclude from background fit")

        psw = qdmpy.shared.polygon.PolygonSelectionWidget(ax, style=kwargs)

        if self.template_polygon_nodes is not None:
            psw.load_nodes(self.template_polygon_nodes)

        plt.show(block=True)
        psw.disconnect()

        pgons = psw.get_polygons_lst()
        if len(pgons) < 1:
            raise RuntimeError("You didn't define any polygons")

        pgon_lst = [pgon.get_nodes() for pgon in pgons if np.shape(pgon.get_nodes())[0] > 2]
        output_dict = {"nodes": pgon_lst, "image_shape": (self.ny, self.nx)}

        if output_path is not None:
            qdmpy.shared.json2dict.dict_to_json(output_dict, output_path)

        self.polygon_nodes = pgon_lst
        return pgon_lst

    def specify_polygons(self, path=None, polys=None):
        """polys takes precedence."""
        if path is None and polys is None:
            warnings.warn("Both path and polys passed to add_polygons were None.")
            return
        if polys is not None:
            if "image_size" in polys:
                if self.ny != polys["image_size"][0] or self.nx != polys["image_size"][1]:
                    raise RuntimeError(
                        """Image size polygons were defined on as passed to add_polygons does not 
                        match this MagSim's mesh."""
                    )
            self.polygon_nodes = polys["nodes"]
        else:
            self.polygon_nodes = qdmpy.shared.json2dict.json_to_dict(path)["nodes"]

    def define_magnets(self, magnetizations, unit_vectors):
        """
        magnetizations: int/float if the same for all polygons, or an iterable of len(polygon_nodes)
        unit_vectors: 3-iterable if the same for all polygons (cartesian coords),
            or an iterable of len(polygon_nodes) each element a 3-iterable
        """
        if isinstance(magnetizations, (float, int)):
            self.magnetizations_lst = [magnetizations for m, _ in enumerate(self.polygon_nodes)]
        else:
            if len(magnetizations) != len(self.polygon_nodes):
                raise ValueError("Number of magnetizations does not match number of magnets.")
            self.magnetizations_lst = [magnetizations]

        if isinstance(unit_vectors, (np.ndarray, list, tuple)):
            if len(unit_vectors) == 3:
                self.unit_vectors_lst = [unit_vectors for m, _ in enumerate(self.polygon_nodes)]
            else:
                self.unit_vectors_lst = [unit_vectors]
        else:
            raise TypeError("unit_vectors wrong type :(")

        # now construct mag
        self.mag = dd(lambda: np.zeros((self.ny, self.nx)))
        grid_y, grid_x = np.meshgrid(range(self.ny), range(self.nx), indexing="ij")

        for i, p in enumerate(self.polygon_nodes):
            in_or_out = p.is_inside(grid_y, grid_x)
            self.mag[self.unit_vectors_lst[i]][in_or_out >= 0] = self.magnetizations_lst[i]

    # TODO better k_vector_epsilon default?? 1e-3 below smallest k value?
    def run(self, standoff=0, pad_mode="mean", pad_factor=2, k_vector_epsilon=1e-6):
        """standoff: fraction of FOV width (i.e. 500um width, standoff=1/500 => standoff=1um)."""

        self.bfield_at_nvs = [
            np.zeros((self.ny, self.nx)),
            np.zeros((self.ny, self.nx)),
            np.zeros((self.ny, self.nx)),
        ]
        for unit_vector in self.unit_vectors_lst:
            # calculate vector field.
            mag_pad, padder = qdmpy.shared.fourier.pad_image(
                self.mag[unit_vector], pad_mode, pad_factor
            )
            fft_mag = numpy_fft.fftshift(numpy_fft.fft2(mag_pad))
            ky, kx, k = qdmpy.shared.fourier.define_k_vectors(
                fft_mag.shape, self.pixel_size, k_vector_epsilon
            )
        pass

    # image outputs: bfield and magnetization
    def get_bfield_im(self, projection=(0, 0, 1)):
        # access bfield_sensor_plane, project onto projection

        # # reshape bnvs to be [bnv_1, bnv_2, bnv_3] for each pixel of image
        # bnvs_reshaped = np.stack(bnvs_to_use, axis=-1)

        # # unv_inv * [bnv_1, bnv_2, bnv_3] for pxl in image -> VERY fast. (applied over last axis)
        # bxyzs = np.apply_along_axis(lambda bnv_vec: np.matmul(unv_inv, bnv_vec), -1, bnvs_reshaped)
        pass

    def get_magnetization_im(self, unit_vector):
        return self.mag[unit_vector]

    def plot_magsim_magnetization(self, unit_vector):
        pass

    def plot_magsim_bfield_at_nvs(self, projection=(0, 0, 1)):
        pass


# below: copies of fns in qdmpy.plot.common.
def _plot_image_on_ax(
    fig, ax, options, image_data, title, c_map, c_range, c_label, annotate_polygons=True
):

    im = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1])

    ax.set_title(title)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    cbar = _add_cbar(im, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    if annotate_polygons:
        for p in options["polygon_nodes"]:
            # polygons reversed to (x,y) indexing for patch
            ax.add_patch(
                matplotlib.patches.Polygon(
                    np.dstack((p[:, 1], p[:, 0]))[0], **options["polygon_patch_params"]
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
