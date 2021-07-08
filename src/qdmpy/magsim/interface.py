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
    def fix_template(self, output_path=None):
        # load template polygons (if they exist)
        # run gui etc. & save in output_path if specified
        # --> load result into self.polygon_nodes
        pass

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
