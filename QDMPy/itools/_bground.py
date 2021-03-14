# -*- coding: utf-8 -*-
"""
This module holds functions to get the background of an image.

Functions
---------
 - `QDMPy.itools._bground.zero_background`
 - `QDMPy.itools._bground.equation_plane`
 - `QDMPy.itools._bground.three_point_background`
 - `QDMPy.itools._bground.mean_background`
 - `QDMPy.itools._bground.residual_poly`
 - `QDMPy.itools._bground.poly_background`
 - `QDMPy.itools._bground.gaussian`
 - `QDMPy.itools._bground.moments`
 - `QDMPy.itools._bground.residual_gaussian`
 - `QDMPy.itools._bground.gaussian_background`
 - `QDMPy.itools._bground.interpolated_background`
 - `QDMPy.itools._bground.filtered_background`

"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.itools._bground.zero_background": True,
    "QDMPy.itools._bground.equation_plane": True,
    "QDMPy.itools._bground.three_point_background": True,
    "QDMPy.itools._bground.mean_background": True,
    "QDMPy.itools._bground.residual_poly": True,
    "QDMPy.itools._bground.poly_background": True,
    "QDMPy.itools._bground.gaussian": True,
    "QDMPy.itools._bground.moments": True,
    "QDMPy.itools._bground.residual_gaussian": True,
    "QDMPy.itools._bground.gaussian_background": True,
    "QDMPy.itools._bground.interpolated_background": True,
    "QDMPy.itools._bground.filtered_background": True,
}

# ============================================================================

import numpy as np
from numpy.polynomial.polynomial import polyval2d
from scipy.optimize import least_squares
from scipy.interpolate import griddata
import warnings

# ============================================================================

import QDMPy.itools._filter

# ============================================================================


def zero_background(image, zero):
    """Background defined by z level of 'zero'"""
    if type(zero) != int and type(zero) != float:
        return TypeError("'zero' must be an int or a float.")
    mn = zero
    bg = np.empty(image)
    bg[:] = mn
    return bg


# ============================================================================


def equation_plane(params, y, x):
    """params: [a, b, c, d] s.t. d = a*y + b*x + c*z
    so z = (1/c) * (d - (ay + bx)) -> return this."""
    return (1 / params[2]) * (params[3] - params[0] * y - params[1] * x)


def points_to_params(points):
    """
    http://pi.math.cornell.edu/~froh/231f08e1a.pdf
    points: iterable of 3 iterables: [x, y, z]
    returns a,b,c,d parameters (see equation_plane)
    """
    rearranged_points = [[p[1], p[0], p[2]] for p in points]  # change to [y, x, z]
    pts = np.array(rearranged_points)
    vec1_in_plane = pts[1] - pts[0]
    vec2_in_plane = pts[2] - pts[0]
    a_normal = np.cross(vec1_in_plane, vec2_in_plane)

    d = np.dot(pts[2], a_normal)
    return *a_normal, d


def three_point_background(image, points):
    """points: len 3 iterable of len 2 iterables: [[x1, y1], [x2, y2], [x3, y3]]
    https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points
    https://www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/
    """

    if len(points) != 3:
        raise ValueError("points needs to be len 3 of format: [x, y] (int or floats).")
    for p in points:
        if len(p) != 2:
            raise ValueError("points needs to be len 3 of format: [x, y] (int or floats).")
        for c in p:
            if type(c) != int and type(c) != float:
                raise ValueError("points needs to be len 3 of format: [x, y] (int or floats).")
        if image.mask[p[1], p[0]]:
            warnings.warn(
                "One of the input points was masked (inside a polygon?), "
                + "falling back on polyfit, order 1"
            )
            return poly_background(image, order=1)

    points = np.array([np.append(p, image[p[1], p[0]]) for p in points])
    Y, X = np.indices(image.shape)
    return equation_plane(points_to_params(points), Y, X)


# ============================================================================


def mean_background(image):
    """Background defined by mean of image."""
    bg = np.empty(image.shape)
    mn = np.nanmean(image)
    bg[:] = mn
    return bg


# ============================================================================


def residual_poly(params, y, x, z, order):
    """
    z = image data, order = highest polynomial order to go to
    y, x: index meshgrids
    """
    # get params to matrix form (as expected by polyval)
    params = np.append(
        params, 0
    )  # add on c[-1, -1] term we don't want (i.e. cross term of next order)
    c = params.reshape((order + 1, order + 1))
    return polyval2d(y, x, c) - z  # note z is flattened, as is polyval


def poly_background(image, order):
    """Background defined by a polynomial fit up to order 'order'."""

    init_params = np.zeros((order + 1, order + 1))
    init_params[0, 0] = np.nanmean(image)  # set zeroth term to be mean to get it started
    Y, X = np.indices(image.shape)
    good_vals = np.logical_and(~np.isnan(image), ~image.mask)
    y = Y[good_vals]
    x = X[good_vals]

    data = image[good_vals]  # flattened
    best_c = least_squares(residual_poly, init_params.flatten()[:-1], args=(y, x, data, order)).x
    best_c = np.append(best_c, 0)
    c = best_c.reshape((order + 1, order + 1))
    return polyval2d(Y.flatten(), X.flatten(), c).reshape(image.shape)  # eval over full image


# ============================================================================


def gaussian(p, y, x):
    """Simple Gaussian function, height, center_y, center_x, width_y, width_x = p ."""
    height, center_y, center_x, width_y, width_x = p
    return height * np.exp(
        -((((center_y - y) / width_y) ** 2 + (center_x - x) / width_x) ** 2) / 2
    )


def moments(image):
    """Calculate moments of image (get initial guesses on gaussian function)"""
    total = np.nansum(image)
    Y, X = np.indices(image.shape)

    center_y = np.nansum(Y * image) / total
    center_x = np.nansum(X * image) / total
    if center_y > np.max(Y) or center_y < 0:
        center_y = np.max(Y) / 2
    if center_x > np.max(X) or center_x < 0:
        center_x = np.max(X) / 2

    col = image[int(center_y), :]
    row = image[:, int(center_x)]
    width_x = np.nansum(np.sqrt(abs((np.arange(col.size) - center_y) ** 2 * col)) / np.nansum(col))
    width_y = np.nansum(np.sqrt(abs((np.arange(row.size) - center_x) ** 2 * row)) / np.nansum(row))
    height = np.nanmax(image)
    return height, center_y, center_x, width_y, width_x


def residual_gaussian(p, y, x, data):
    """Residual of data with a gaussian model."""
    return gaussian(p, y, x) - data


def gaussian_background(image):
    """Background defined by a Gaussian function."""
    params = moments(image)
    Y, X = np.indices(image.shape)
    good_vals = np.logical_and(~np.isnan(image), ~image.mask)
    y = Y[good_vals]
    x = X[good_vals]
    data = image[good_vals]
    p = least_squares(residual_gaussian, params, method="lm", args=(y, x, data)).x
    return gaussian(p, Y, X)


# ============================================================================


def interpolated_background(image, interp_method, polygons, sigma):
    """Background defined by the dataset smoothed via a sigma-gaussian filtering,
    and method-interpolation over masked (polygon) regions.

    method available: nearest, linear, cubic.
    """
    if type(polygons) != list or type(polygons[0]) != QDMPy.itools._polygon.Polygon:
        raise TypeError("polygons were not None, a list or a list of Polygon objects")

    ylen, xlen = image.shape
    isnt_poly = np.full(image.shape, True)  # all masked to start with

    # coordinate grid for all coordinates
    grid_y, grid_x = np.meshgrid(range(ylen), range(xlen), indexing="ij")

    for p in polygons:
        in_or_out = p.is_inside(grid_y, grid_x)
        # mask all vals that are not background
        is_this_poly = np.ma.masked_greater_equal(in_or_out, 0).mask  # >= 0 => inside/on poly
        isnt_poly = np.logical_and(isnt_poly, ~is_this_poly)  # prev isnt_poly and isnt this poly

    # now we want to send all of the values in indexes that is_bg is True to griddata
    pts = []
    vals = []
    for i in range(ylen):
        for j in range(xlen):
            if isnt_poly[i, j]:
                pts.append([i, j])
                vals.append(image[i, j])

    bg_interp = griddata(pts, vals, (grid_y, grid_x), method=interp_method)

    return QDMPy.itools._filter.get_im_filtered(bg_interp, "gaussian", sigma=sigma)


# ============================================================================


def filtered_background(image, filter_type, **kwargs):
    """Background defined by a filter_type-filtering of the image.
    Passed to `QDMPy.itools._filter.get_im_filtered`."""
    return QDMPy.itools._filter.get_im_filtered(image, filter_type, **kwargs)


# ============================================================================
