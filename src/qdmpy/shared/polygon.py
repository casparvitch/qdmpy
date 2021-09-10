# -*- coding: utf-8 -*-
"""
This module holds the Polygon class: a class to compute if a point
lies inside/outside/on-side of a polygon. Also defined is a function
(polygon_gui) that can be called to select a polygon region on an image.

Polygon-GUI
-----------
Function to select polygons on an image. Ensure you have the required
gui backends for matplotlib. Best ran seperately/not within jupyter.
E.g. open python REpl (python at cmd), 'import qdmpy.itool', then
run qdmpy.shared.polygon.Polygonpolygon_gui() & follow the prompts.

An optional array (i.e. the image used to define regions) can be passed
to polygon_gui.

The output json path can then be specified in the usual way (there's an
option called 'polygon_nodes_path') to utilize these regions in the main
processing code.

Polygon
-------
This is a Python 3 implementation of the Sloan's improved version of the
Nordbeck and Rystedt algorithm, published in the paper:

SLOAN, S.W. (1985): A point-in-polygon program.
    Adv. Eng. Software, Vol 7, No. 1, pp 45-47.

This class has 1 method (is_inside) that returns the minimum distance to the
nearest point of the polygon:

If is_inside < 0 then point is outside the polygon.
If is_inside = 0 then point in on a side of the polygon.
If is_inside > 0 then point is inside the polygon.

Sam Scholten copied from:
http://code.activestate.com/recipes/578381-a-point-in-polygon-program-sw-sloan-algorithm/
-> swapped x & y args order (etc.) for image use.

Classes
-------
 - `qdmpy.shared.polygon.Polygon`

Functions
---------
 - `qdmpy.shared.polygon.polygon_selector`
 - `qdmpy.shared.polygon.polygon_gui`
 - `qdmpy.shared.polygon.Polygon`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.shared.polygon.polygon_gui": True,
    "qdmpy.shared.polygon.Polygon": True,
}

# ============================================================================


import numpy as np
import PySimpleGUI as sg  # noqa: N813
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from polylabel import polylabel
import numba
from numba import jit

# ============================================================================

from qdmpy.shared.json2dict import json_to_dict, dict_to_json
import qdmpy.shared.widget
from qdmpy.shared.misc import warn

# ============================================================================

CMAP_OPTIONS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "binary",
    "gist_yarg",
    "gist_gray",
    "gray",
    "bone",
    "pink",
    "spring",
    "summer",
    "autumn",
    "winter",
    "cool",
    "Wistia",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
    "twilight",
    "twilight_shifted",
    "hsv",
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    "flag",
    "prism",
    "ocean",
    "gist_earth",
    "terrain",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "CMRmap",
    "cubehelix",
    "brg",
    "gist_rainbow",
    "rainbow",
    "jet",
    "nipy_spectral",
    "gist_ncar",
]

# ============================================================================


# def _tri_2area_det(yvert, xvert):
#     """
#     Compute twice the area of the triangle defined by points with using
#     determinant formula.

#     Arguments
#     ---------

#     yvert : array-like
#         A vector of nodal y-coords (all unique).

#     xvert : array-like
#         A vector of nodal x-coords (all unique).

#     Returns
#     -------
#     Twice the area of the triangle defined by the points.

#     Notes:asfarray

#     _tri_2area_det is positive if asfarraypoints define polygon in anticlockwise order.
#     _tri_2area_det is negative if asfarraypoints define polygon in clockwise order.
#     _tri_2area_det is zero if at least two of the points are coincident or if
#         all points are collinear.

#     """
#     xvert = np.asfarray(xvert)
#     yvert = np.asfarray(yvert)
#     x_prev = np.concatenate(([xvert[-1]], xvert[:-1]))
#     y_prev = np.concatenate(([yvert[-1]], yvert[:-1]))
#     return np.sum(yvert * x_prev - xvert * y_prev, axis=0)  # good or no?
#     # return np.sum(xvert * y_prev - yvert * x_prev, axis=0)

# ============================================================================


@jit("int8(float64[:], float64[:,:])", nopython=True, cache=True)
def _is_inside_sm(point, polygon):
    # https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
    # note this fn works in (x,y) coords (but if point/polygon is consistent all is g)
    conv_map = {0: -1, 1: 1, 2: 0}
    length = polygon.shape[0] - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = (
                    dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]
                )  # noqa: N806

                if (
                    point[0] > F
                ):  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (
                point[0] == polygon[jj][0]
                or (dy == 0 and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0)
            ):
                return 2

        ii = jj
        jj += 1

    # print 'intersections =', intersections
    return conv_map[intersections & 1]


# ============================================================================


@jit(nopython=True, parallel=True, cache=True)
def _is_inside_sm_parallel(points, polygon):
    # https://stackoverflow.com/questions/36399381/ \
    #   whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
    # note this fn works in (x,y) coords (but if point/polygon is consistent all is g.)
    p_ar = np.asfarray(points)
    pts_shape = p_ar.shape[:-1]
    p_ar_flat = p_ar.reshape(-1, 2)  # shape: (len_y * len_x, 2), i.e. long list of coords (y, x)
    d = np.zeros(p_ar_flat.shape[0], dtype=numba.int8)
    for i in numba.prange(p_ar_flat.shape[0]):
        d[i] = _is_inside_sm(p_ar_flat[i], polygon)
    d = d.reshape(pts_shape)
    return d


# ============================================================================


class Polygon:
    """
    Polygon object.

    Arguments
    ---------
    y : array-like
        A sequence of nodal y-coords (all unique).

    x : array-like
        A sequence of nodal x-coords (all unique).
    """

    def __init__(self, y, x):
        if len(y) != len(x):
            raise IndexError("y and x must be equally sized.")
        self.y = np.asfarray(y)
        self.x = np.asfarray(x)
        # Closes the polygon if were open
        y1, x1 = y[0], x[0]
        yn, xn = y[-1], x[-1]
        if x1 != xn or y1 != yn:
            self.y = np.concatenate((self.y, [y1]))
            self.x = np.concatenate((self.x, [x1]))
        # # Anti-clockwise coordinates # irrelevant...?
        # if _tri_2area_det(self.y, self.x) < 0:
        #     self.y = self.y[::-1]
        #     self.x = self.x[::-1]

    # =============================================== #

    def get_center(self):
        """Get center point that is inside polygon."""
        # polylabel uses opposite indexing convention, but it doesn't effect result!
        return polylabel([self.get_nodes()])

    # =============================================== #

    def get_nodes(self):
        # get nodes as a list [[y1,x1], [y2,x2]] etc.
        return [[y, x] for y, x in zip(self.y, self.x)]

    # =============================================== #

    def get_yx(self):
        return np.stack((self.y, self.x), axis=-1)

    # =============================================== #

    def is_inside(self, y, x):
        # return value:
        # <0 - the point is outside the polygon
        # =0 - the point is one edge (boundary)
        # >0 - the point is inside the polygon
        xs = np.asfarray(x)
        ys = np.asfarray(y)
        # Check consistency
        if xs.shape != ys.shape:
            raise IndexError("x and y has different shapes")
        # check if single point
        if xs.shape is tuple():
            return _is_inside_sm((y, x), self.get_yx())
        else:
            return _is_inside_sm_parallel(np.stack((ys, xs), axis=-1), self.get_yx())

    # def is_inside(self, ypoint, xpoint, smalld=1e-12):
    #     """
    #     Check if point is inside a general polygon.

    #     Arguments
    #     ---------

    #     ypoint : array-like
    #         The y-coord of the point to be tested.
    #     xpoint : array-like
    #         The x-coords of the point to be tested.
    #     smalld : float
    #         A small float number.

    #     ypoint and xpoint could be scalars or array-like sequences.

    #     Returns
    #     -------
    #     mindst : scalar
    #         The distance from the point to the nearest point of the
    #         polygon.
    #         If mindst < 0 then point is outside the polygon.
    #         If mindst = 0 then point in on a side of the polygon.
    #         If mindst > 0 then point is inside the polygon.

    #     Notes
    #     -----

    #     An improved version of the algorithm of Nordbeck and Rydstedt.

    #     REF: SLOAN, S.W. (1985): A point-in-polygon program. Adv. Eng.
    #          Software, Vol 7, No. 1, pp 45-47.

    #     """
    #     xpoint = np.asfarray(xpoint)
    #     ypoint = np.asfarray(ypoint)
    #     # Scalar to array
    #     if xpoint.shape is tuple():
    #         xpoint = np.array([xpoint], dtype=float)
    #         ypoint = np.array([ypoint], dtype=float)
    #         scalar = True
    #     else:
    #         scalar = False
    #     # Check consistency
    #     if xpoint.shape != ypoint.shape:
    #         raise IndexError("x and y has different shapes")
    #     # If snear = True: Dist to nearest side < nearest vertex
    #     # If snear = False: Dist to nearest vertex < nearest side
    #     snear = np.ma.masked_all(xpoint.shape, dtype=bool)
    #     # Initialize arrays
    #     mindst = np.ones_like(ypoint, dtype=float) * np.inf
    #     j = np.ma.masked_all(ypoint.shape, dtype=int)
    #     x = self.x
    #     y = self.y
    #     n = len(y) - 1  # Number of sides/vertices defining the polygon

    #     # Loop over each side defining polygon
    #     for i in range(n):
    #         d = np.ones_like(ypoint, dtype=float) * np.inf
    #         # Start of side has coords (y1, x1)
    #         # End of side has coords (y2, x2)
    #         # Point has coords (xpoint, ypoint)
    #         x1 = x[i]
    #         y1 = y[i]
    #         x21 = x[i + 1] - x1
    #         y21 = y[i + 1] - y1
    #         x1p = x1 - xpoint
    #         y1p = y1 - ypoint

    #         # Points on infinite line defined by
    #         #     y = y1 + t * (y1 - y2)
    #         #     x = x1 + t * (x1 - x2)
    #         # where
    #         #     t = 0    at (y1, x1)
    #         #     t = 1    at (y2, x2)
    #         # Find where normal passing through (xpoint, ypoint) intersects
    #         # infinite line
    #         t = -(x1p * x21 + y1p * y21) / (x21 ** 2 + y21 ** 2)
    #         tlt0 = t < 0
    #         tle1 = (0 <= t) & (t <= 1)  # this looks silly but don't change it
    #         # Normal intersects side
    #         d[tle1] = (x1p[tle1] + t[tle1] * x21) ** 2 + (y1p[tle1] + t[tle1] * y21) ** 2
    #         # Normal does not intersects side
    #         # Point is closest to vertex (y1, x1)
    #         # Compute square of distance to this vertex
    #         d[tlt0] = x1p[tlt0] ** 2 + y1p[tlt0] ** 2
    #         # Store distances
    #         mask = d < mindst
    #         mindst[mask] = d[mask]
    #         j[mask] = i
    #         # Point is closer to (y1, x1) than any other vertex or side
    #         snear[mask & tlt0] = False
    #         # Point is closer to this side than to any other side or vertex
    #         snear[mask & tle1] = True

    #     if np.ma.count(snear) != snear.size:
    #         raise IndexError("Error computing distances")

    #     mindst **= 0.5
    #     # Point is closer to its nearest vertex than its nearest side, check if
    #     # nearest vertex is concave

    #     # If the nearest vertex is concave then point is inside the polygon,
    #     # else the point is outside the polygon.
    #     jo = j.copy()
    #     jo[j == 0] -= 1
    #     area = _tri_2area_det([y[j + 1], y[j], y[jo - 1]], [x[j + 1], x[j], x[jo - 1]])
    #     mindst[~snear] = np.copysign(mindst, area)[~snear]
    #     # Point is closer to its nearest side than to its nearest vertex, check
    #     # if point is to left or right of this side.
    #     # If point is to left of side it is inside polygon, else point is
    #     # outside polygon.
    #     area = _tri_2area_det([y[j], y[j + 1], ypoint], [x[j], x[j + 1], xpoint])
    #     mindst[snear] = np.copysign(mindst, area)[snear]
    #     # Point is on side of polygon
    #     mindst[np.fabs(mindst) < smalld] = 0
    #     # If input values were scalar then the output should be too
    #     if scalar:
    #         mindst = float(mindst)
    #     return mindst


# ============================================================================


def polygon_selector(
    numpy_txt_file_path,
    json_output_path=None,
    json_input_path=None,
    mean_plus_minus=None,
    strict_range=None,
    print_help=False,
    **kwargs,
):
    """
    Generates mpl (qt) gui for selecting a polygon.

    Arguments
    ---------
    numpy_txt_file_path : path
        Path to (numpy) .txt file to load as image.
    json_output_path : str or path-like, default="~/poly.json"
        Path to put output json, defaults to home/poly.json.
    json_input_path : str or path-like, default=None
        Loads previous polygons at this path for editing.
    mean_plus_minus : float, default=None
        Plot image with color scaled to mean +- this number.
    strict_range: length 2 list, default=None
        Plot image with color scaled between these values. Precedence over mean_plus_minus.
    print_help : bool, default=False
        View this message.
    **kwargs : dict
        Other keyword arguments to pass to plotters. Currently implemented:
            cmap : string
                Passed to imshow.
            lineprops : dict
                Passed to PolygonSelectionWidget.
            markerprops : dict
                Passed to PolygonSelectionWidget.


    GUI help
    --------
    In the mpl gui, select points to draw polygons.
    Press 'enter' to continue in the program.
    Press the 'esc' key to reset the current polygon
    Hold 'shift' to move all of the vertices (from all polygons)
    'ctrl' to move a single vertex in the current polygon
    'alt' to start a new polygon (and finalise the current one)
    'del' to clear all lines from the graphic  (thus deleting all polygons).

    """
    if print_help:
        print(
            """
        Help
        ====

        Input help
        ----------
        numpy_txt_file_path : path 
            Path to (numpy) .txt file to load as image.
        json_output_path : str or path-like, default="~/poly.json"
            Path to put output json, defaults to home/poly.json.
        json_input_path : str or path-like, default=None
            Loads previous polygons at this path for editing.
        mean_plus_minus : float, default=None
            Plot image with color scaled to mean +- this number.
        strict_range: length 2 list, default=None
            Plot image with color scaled between these values. Precedence over mean_plus_minus.
        help : bool, Default=False
            View this message.
        **kwargs : dict
        Other keyword arguments to pass to plotters. Currently implemented:
            cmap : string
                Passed to imshow.
            lineprops : dict
                Passed to PolygonSelectionWidget.
            markerprops : dict
                Passed to PolygonSelectionWidget.

        GUI help
        --------
        In the mpl gui, select points to draw polygons.
        Press 'enter' to continue in the program.
        Press the 'esc' key to reset the current polygon
        Hold 'shift' to move all of the vertices (from all polygons)
        Hold 'r' and scroll to resize all of the polygons.
        'ctrl' to move a single vertex in the current polygon
        'alt' to start a new polygon (and finalise the current one)
        'del' to clear all lines from the graphic  (thus deleting all polygons).
        'right click' on a vertex (of a finished polygon) to remove it.
        """
        )
        return []

    image = np.loadtxt(numpy_txt_file_path)

    if json_input_path is None:
        polys = None
        polygon_nodes = None
    else:
        polys = json_to_dict(json_input_path)
        polygon_nodes = polys["nodes"]
        if "image_shape" in polys:
            shp = polys["image_shape"]
            if shp[0] != image.shape[0] or shp[1] != image.shape[1]:
                warn("Image shape loaded polygons were defined on does not match current image.")

    fig, ax = plt.subplots()
    minimum = np.nanmin(image)
    maximum = np.nanmax(image)
    if (
        strict_range is not None
        and isinstance(strict_range, (list, np.ndarray, tuple))
        and len(strict_range) == 2
    ):
        vmin, vmax = strict_range
    elif mean_plus_minus is not None and isinstance(mean_plus_minus, (float, int)):
        mean = np.mean(image)
        vmin, vmax = mean - mean_plus_minus, mean + mean_plus_minus
    else:
        vmin = vmax = [minimum, maximum]
    img = ax.imshow(
        image,
        aspect="equal",
        cmap=kwargs["cmap"] if "cmap" in kwargs else "bwr",
        vmin=vmin,
        vmax=vmax,
    )

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

    psw = PolygonSelectionWidget(ax, style=kwargs)

    if polygon_nodes is not None:
        psw.load_nodes(polygon_nodes)

    plt.tight_layout()
    plt.show(block=True)
    psw.disconnect()

    pgons = psw.get_polygons_lst()
    if len(pgons) < 1:
        raise RuntimeError("You didn't define any polygons")

    # exclude polygons with nodes < 3
    pgon_lst = [pgon.get_nodes() for pgon in pgons if np.shape(pgon.get_nodes())[0] > 2]
    output_dict = {"nodes": pgon_lst, "image_shape": image.shape}

    dict_to_json(output_dict, json_output_path)

    return output_dict


# ===============================================================================================


def polygon_gui(image=None):
    """Load gui to select polygon regions. Follow the prompts.

    Arguments
    ---------
    image : 2D array-like, optional
        image to use for select polygons on.
        (the default is None, forces user to specify path in gui)

    Returns
    -------
    polygon_lst : list
        list of polygons (each defined by a list of nodes [y, x])

    Also saves polygon_lst into a json at a given path (in the gui).
    """
    sg.change_look_and_feel("Material1")
    sg.SetOptions(margins=(10, 10), text_justification="r")

    cmap_lst = []
    for cmap in CMAP_OPTIONS:
        cmap_lst.append(cmap)
        cmap_lst.append(cmap + "_r")

    cmap_lst = sorted(cmap_lst)
    size = (3, 1)

    if image is not None:
        layout0 = [sg.Text("Image file chosen: " + str(image.__class__) + " @ " + hex(id(image)))]
    else:
        layout0 = [
            sg.Text("Choose .txt numpy array File"),
            sg.In("", key="image_path", size=(40, 1), justification="left"),
            sg.FileBrowse(target="image_path"),
        ]

    layout = [
        layout0,
        [
            sg.Txt("Polygon input (json) path:"),
            sg.In("", key="polygon_path", size=(40, 1), justification="left"),
            sg.FileBrowse(target="polygon_path"),
        ],
        [
            sg.Txt("Polygon output (json) path:"),
            sg.In("", key="output_path", size=(40, 1), justification="left"),
        ],
        [
            sg.Checkbox("Load the polygon set", key="do_load_polys"),
            sg.Submit(),
        ],
        [sg.Text("Cmap:"), sg.Combo(cmap_lst, default_value="viridis", key="cmap")],
        [sg.Text(" ")],
        [sg.Text("Line style:")],
        [
            sg.Text("Width:"),
            sg.In("1.0", key="lw", size=size),
            sg.Text("Colour:"),
            sg.In("g", key="lc", size=size),
            sg.Text("Alpha (in [0, 1]):"),
            sg.In("0.5", key="la", size=size),
        ],
        [sg.Text("Marker style:")],
        [
            sg.Text("Type:"),
            sg.In("o", key="mt", size=size),
            sg.Text("Size:"),
            sg.In("2.0", key="ms", size=size),
            sg.Text("Colour:"),
            sg.In("g", key="mc", size=size),
            sg.Text("Alpha (in [0, 1]):"),
            sg.In("0.5", key="ma", size=size),
        ],
        [sg.Text(" ")],
        [sg.Text("In the mpl gui, select points to draw polygons.")],
        [sg.Text("Press 'enter' to continue in the program.")],
        [sg.Text("Press the 'esc' key to reset the current polygon")],
        [sg.Text("Hold 'shift' to move all of the vertices (from all polygons)")],
        [sg.Text("Hold 'r' and scroll to resize all of the polygons.")],
        [sg.Text("'ctrl' to move a single vertex in the current polygon")],
        [sg.Text("'alt' to start a new polygon (and finalise the current one)")],
        [sg.Text("'del' to clear all lines from the graphic  (thus deleting all polygons).")],
        [sg.Text("'right click' on a vertex (of a closer polygon) to remove it.")],
    ]
    window = sg.Window("Polygon selection tool", layout, grab_anywhere=False)
    event, values = window.read()
    # if event is None:
    #     # sys.exit()
    #     pass
    window.close()
    del window

    style = {
        "lineprops": {
            "color": values["lc"],
            "linestyle": "-",
            "linewidth": float(values["lw"]),
            "alpha": float(values["la"]),
        },
        "markerprops": {
            "marker": values["mt"],
            "markersize": float(values["ms"]),
            "mec": values["mc"],
            "mfc": values["mc"],
            "alpha": float(values["ma"]),
        },
        "cmap": values["cmap"],
    }
    # set user style...
    # set cmap

    if image is None:
        image = np.loadtxt(values["image_path"])

    if values["do_load_polys"]:
        polys = json_to_dict(values["polygon_path"])
        polygon_nodes = polys["nodes"]
        if "image_shape" in polys:
            shp = polys["image_shape"]
            if shp[0] != image.shape[0] or shp[1] != image.shape[1]:
                warn("Image shape loaded polygons were defined on does not match current image.")
    else:
        polygon_nodes = None

    fig, ax = plt.subplots()
    minimum = np.nanmin(image)
    maximum = np.nanmax(image)
    mean = np.mean(image)
    max_distance_from_mean = np.max([abs(maximum - mean), abs(minimum - mean)])
    img = ax.imshow(
        image,
        aspect="equal",
        cmap=values["cmap"],
        vmin=mean - max_distance_from_mean,
        vmax=mean + max_distance_from_mean,
    )

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

    psw = PolygonSelectionWidget(ax, style=style)

    if polygon_nodes is not None:
        psw.load_nodes(polygon_nodes)

    plt.tight_layout()
    plt.show(block=True)
    psw.disconnect()

    pgons = psw.get_polygons_lst()
    if len(pgons) < 1:
        raise RuntimeError("You didn't define any polygons")

    # exclude polygons with nodes < 3
    pgon_lst = [pgon.get_nodes() for pgon in pgons if np.shape(pgon.get_nodes())[0] > 2]
    output_dict = {"nodes": pgon_lst, "image_shape": image.shape}

    dict_to_json(output_dict, values["output_path"])

    return output_dict


# ================================================================================================ #


class PolygonSelectionWidget:
    """
    How to Use
    ----------
    selector = PolygonSelectionWidget(ax, ...)
    plt.show()
    selector.disconnect()
    polygon_lst = selector.get_polygon_lst()

    """

    def __init__(self, ax, style=None, base_scale=1.5):
        self.canvas = ax.figure.canvas

        dflt_style = {
            "lineprops": {"color": "g", "linestyle": "-", "linewidth": 1.0, "alpha": 0.5},
            "markerprops": {
                "marker": "o",
                "markersize": 2.0,
                "mec": "g",
                "mfc": "g",
                "alpha": 0.5,
            },
        }

        self.lp = dflt_style["lineprops"]
        self.mp = dflt_style["markerprops"]
        if style is not None:
            if "lineprops" in style and isinstance(style["lineprops"], dict):
                for key, item in style["lineprops"]:
                    self.lp[key] = item
            if "markerprops" in style and isinstance(style["markerprops"], dict):
                for key, item in style["markerprops"]:
                    self.mp[key] = item

        vsr = 7.5 * self.mp["markersize"]  # linear scaling on what our select radius is
        self.ax = ax
        self.polys = qdmpy.shared.widget.PolygonSelector(
            ax,
            self.onselect,
            lineprops=self.lp,
            markerprops=self.mp,
            vertex_select_radius=vsr,
            base_scale=base_scale,
        )
        self.pts = []

    def onselect(self, verts):
        # only called when polygon is finished
        self.pts.append(verts)
        self.canvas.draw_idle()

    def disconnect(self):
        self.polys.disconnect_events()
        self.canvas.draw_idle()

    def get_polygons_lst(self):
        lst = []
        for p in self.polys.xy_verts:
            # NOTE opposite indexing convention here (xy_verts -> yx Polygon)
            new_polygon_obj = Polygon(p[1], p[0])
            lst.append(new_polygon_obj)
        return lst

    def load_nodes(self, polygon_nodes):
        # polygon nodes: list of polygons, each polygon is a list of nodes
        for polygon in polygon_nodes:
            nodes_ar = np.array(polygon)

            # x & y convention swapped here
            new_line = Line2D(nodes_ar[:, 1], nodes_ar[:, 0], **self.lp)
            self.ax.add_line(new_line)

            new_line_dict = dict(line_obj=new_line, xs=nodes_ar[:, 1], ys=nodes_ar[:, 0])

            self.polys.artists.append(new_line)
            self.polys.lines.append(new_line_dict)  # list of line dicts

        self.polys.draw_polygon()
