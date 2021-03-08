# -*- coding: utf-8 -*-
"""
This module holds ...

Functions
---------
 - `QDMPy.itools._bground.`


Notes
-----
 - ensure all methods work with masked arrays (numpy.ma)
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.itools._bground.": True,
}

# ============================================================================

import numpy as np
from numpy.polynomial.polynomial import polyval2d
from scipy.optimize import least_squares
from scipy.interpolate import griddata

# ============================================================================

import QDMPy.i

# ============================================================================


def zero_background(image, zero):
    if type(zero) != int and type(zero) != float:
        return TypeError("'zero' must be an int or a float.")
    return zero


# ============================================================================
# TODO check LH convention works ok here (i.e. on the cross product etc.)


def equation_plane(params, y, x):
    # params: [a, b, c, d] s.t. d = a*y + b*x + c*z
    # so z = (1/c) * (d - (ay + bx)) -> return this.
    return (1 / params[2]) * (params[3] - params[0] * y - params[1] * x)


def points_to_params(points):
    # http://pi.math.cornell.edu/~froh/231f08e1a.pdf
    # points: iterable of 3 iterables: [y, x, z]
    points = np.array(points)
    vec1_in_plane = points[1] - points[0]
    vec2_in_plane = points[2] - points[0]
    a_normal = np.cross(vec1_in_plane, vec2_in_plane)

    d = np.dot(points[2], a_normal)
    return *a_normal, d


def three_point_background(image, points):
    """ points: iterable of 3 iterables: [y, x, z] """
    # https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points
    # https://www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/

    # FIXME
    # if one of the points is masked, do poly fit and warn user!

    if len(points) != 3:
        raise ValueError("points needs to be iterable of 3 iterables: [y, x, z].")
    for p in points:
        if len(p) != 3:
            raise ValueError("points needs to be iterable of 3 iterables: [y, x, z].")
        for c in p:
            if type(c) != int and type(c) != float:
                raise ValueError("points needs to be iterable of 3 iterables: [y, x, z].")
    Y, X = np.indices(image.shape)
    return equation_plane(points_to_params(points), Y, X)


# ============================================================================


def mean_background(image):
    return np.nanmean(image)


# ============================================================================


def residual_poly(params, y, x, z, order):
    # z = image data, order = highest polynomial order to go to
    # get params to matrix form (as expected by polyval)
    c = params.reshape((order + 1, order + 1))
    return polyval2d(y, x, c) - z


def poly_background(image, order):

    init_params = np.zeros((order + 1, order + 1))
    init_params[0, 0] = np.nanmean(image)  # set zeroth term to be mean to get it started
    Y, X = np.indices(image.shape)
    mask = np.logical_or(~np.isnan(image), image.mask)
    y = Y[mask]
    x = X[mask]
    data = image[mask]
    best_c = least_squares(residual_poly, init_params, method="lm", args=(y, x, data, order)).x
    return polyval2d(Y, X, best_c)  # eval at non-masked regions too!


# ============================================================================


def gaussian(p, y, x):
    height, center_y, center_x, width_y, width_x = p
    return height * np.exp(
        -((((center_y - y) / width_y) ** 2 + (center_x - x) / width_x) ** 2) / 2
    )


def moments(image):
    total = np.nansum(image)
    Y, X = np.indices(image.shape)

    center_y = np.nansum(Y * image) / total
    center_x = np.nansum(X * image) / total
    if center_y > np.max(X) or center_x < 0:
        center_x = np.max(X) / 2
    if center_y > np.max(X) or center_y < 0:
        center_y = np.max(Y) / 2

    col = image[int(center_y), :]
    row = image[:, int(center_x)]
    width_x = np.nansum(np.sqrt(abs((np.arange(row.size) - center_x) ** 2 * col)) / np.nansum(row))
    width_y = np.nansum(np.sqrt(abs((np.arange(col.size) - center_y) ** 2 * row)) / np.nansum(col))
    height = np.nanmax(image)
    return height, center_y, center_x, width_y, width_x


def residual_gaussian(p, y, x, data):
    return gaussian(p, y, x) - data


def gaussian_background(image):
    params = moments(image)
    Y, X = np.indices(image.shape)
    mask = np.logical_or(~np.isnan(image), image.mask)
    y = Y[mask]
    x = X[mask]
    data = image[mask]
    p = least_squares(residual_gaussian, params, method="lm", args=(y, x, data)).x
    return gaussian(p, Y, X)


# ============================================================================


def interpolated_background(image, method, polygons, sigma):
    if not polygons:
        return np.ma.masked_array(image)
    if type(polygons) != list or type(polygons[0]) != QDMPy.itools._polygon.Polygon:
        raise TypeError("polygons were not None, a list or a list of Polygon objects")

    ylen, xlen = image.shape
    masked_area = np.full(image.shape, True)  # all masked to start with

    # coordinate grid for all coordinates
    grid_y, grid_x = np.meshgrid(range(ylen), range(xlen), indexing="ij")

    for p in polygons:
        in_or_out = p.is_inside(grid_y, grid_x)
        # mask all vals that are not background
        m = np.ma.masked_greater_equal(in_or_out, 0).mask
        masked_area = np.logical_and(masked_area, ~m)

    # now we want to send all of the values in indexes that is_bg is True to griddata
    pts = []
    vals = []
    for i in range(xlen):
        for j in range(ylen):
            if ~masked_area[i, j]:
                pts.append([i, j])
                vals.append(image[j, i])  # this seems wrong? Check this... why indexing ij???

    bg_interp = griddata(pts, vals, (grid_x, grid_y), method=method)  # method=interp_method

    return = filter_background(bg_interp, "gaussian", sigma)


# ============================================================================


def filtered_background(image, filter_type, **kwargs):
    return QDMPy.itools._filter.get_im_filtered(image, filter_type, **kwargs)


# ============================================================================
