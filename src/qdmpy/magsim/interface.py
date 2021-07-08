# -*- coding: utf-8 -*-
"""Interface to mag simulations


Functions
---------
 - `qdmpy.magsim.interface.`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.magsim.interface.": True,
}
# ============================================================================

import numpy as np
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

# ============================================================================

import qdmpy.shared.polygon
import qdmpy.shared.json2dict

# ============================================================================

# 1. define image resolution (nx, ny)
# 2. base_polygons (if any)
# 3. reload those polygons on new mesh, resize/add new/etc.
# 4. save/load those finished polygons on this mesh
# 5. define magnet properties (magnitude, unit_vec)
# 6. define sim. properties -> bnv projection etc., standoff {standoff as decimal of FOV}
# 7. run simulation to get field maps

# image outputs: bfield and magnetization


class MagSim:
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

    # load template polygons if they exist and allow user to rescale etc.
    def fix_template(self, output_path=None, **kwargs):
        # load template polygons (if they exist)
        # run gui etc. & save in output_path if specified
        # --> load result into self.polygon_nodes

        fig, ax = plt.subplots()
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
            self.magnetizations_lst = list(magnetizations)

        if isinstance(unit_vectors, (np.ndarray, list, tuple)):
            if len(unit_vectors) == 3:
                self.unit_vectors = [unit_vectors for m, _ in enumerate(self.polygon_nodes)]
        else:
            raise TypeError("unit_vectors wrong type :(")

    def run(self):
        pass

    def get_bfield_im(self):
        pass

    def get_magnetization_im(self):
        pass
