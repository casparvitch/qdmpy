# -*- coding: utf-8 -*-
"""
This module holds the Polygon class: a class to compute if a point
lies inside/outside/on-side of a polygon.

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

Functions
---------
 - `QDMPy.itools._polygon.`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "QDMPy.itools._polygon.": True,
}

# ============================================================================

import numpy as np

# ============================================================================

# ============================================================================


def _tri_2area_det(yvert, xvert):
    """
    Compute twice the area of the triangle defined by points with using
    determinant formula.

    Arguments
    ---------

    yvert : array-like
        A vector of nodal x-coords (all unique).

    xvert : array-like
        A vector of nodal y-coords (all unique).

    Returns
    -------
    Twice the area of the triangle defined by the points.

    Notes:asfarray

    _tri_2area_det is positive if asfarraypoints define polygon in anticlockwise order.
    _tri_2area_det is negative if asfarraypoints define polygon in clockwise order.
    _tri_2area_det is zero if at least two of the points are coincident or if
        all points are collinear.

    """
    xvert = np.asfarray(xvert)
    yvert = np.asfarray(yvert)
    x_prev = np.concatenate(([xvert[-1]], xvert[:-1]))
    y_prev = np.concatenate(([yvert[-1]], yvert[:-1]))
    return np.sum(xvert * y_prev - yvert * x_prev, axis=0)


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
        # Anti-clockwise coordinates
        if _tri_2area_det(self.y, self.x) < 0:
            self.y = self.y[::-1]
            self.x = self.x[::-1]

    # =============================================== #

    def is_inside(self, ypoint, xpoint, smalld=1e-12):
        """
        Check if point is inside a general polygon.

        Arguments
        ---------

        ypoint : array-like
            The y-coord of the point to be tested.
        xpoint : array-like
            The x-coords of the point to be tested.
        smalld : float
            A small float number.

        ypoint and xpoint could be scalars or array-like sequences.

        Returns
        -------
        mindst : scalar
            The distance from the point to the nearest point of the
            polygon.
            If mindst < 0 then point is outside the polygon.
            If mindst = 0 then point in on a side of the polygon.
            If mindst > 0 then point is inside the polygon.

        Notes
        -----

        An improved version of the algorithm of Nordbeck and Rydstedt.

        REF: SLOAN, S.W. (1985): A point-in-polygon program. Adv. Eng.
             Software, Vol 7, No. 1, pp 45-47.

        """
        xpoint = np.asfarray(xpoint)
        ypoint = np.asfarray(ypoint)
        # Scalar to array
        if xpoint.shape is tuple():
            xpoint = np.array([xpoint], dtype=float)
            ypoint = np.array([ypoint], dtype=float)
            scalar = True
        else:
            scalar = False
        # Check consistency
        if xpoint.shape != ypoint.shape:
            raise IndexError("x and y has different shapes")
        # If snear = True: Dist to nearest side < nearest vertex
        # If snear = False: Dist to nearest vertex < nearest side
        snear = np.ma.masked_all(xpoint.shape, dtype=bool)
        # Initialize arrays
        mindst = np.ones_like(ypoint, dtype=float) * np.inf
        j = np.ma.masked_all(ypoint.shape, dtype=int)
        x = self.x
        y = self.y
        n = len(y) - 1  # Number of sides/vertices defining the polygon

        # Loop over each side defining polygon
        for i in range(n):
            d = np.ones_like(ypoint, dtype=float) * np.inf
            # Start of side has coords (y1, x1)
            # End of side has coords (y2, x2)
            # Point has coords (xpoint, ypoint)
            x1 = x[i]
            y1 = y[i]
            x21 = x[i + 1] - x1
            y21 = y[i + 1] - y1
            x1p = x1 - xpoint
            y1p = y1 - ypoint

            # Points on infinite line defined by
            #     y = y1 + t * (y1 - y2)
            #     x = x1 + t * (x1 - x2)
            # where
            #     t = 0    at (y1, x1)
            #     t = 1    at (y2, x2)
            # Find where normal passing through (xpoint, ypoint) intersects
            # infinite line
            t = -(x1p * x21 + y1p * y21) / (x21 ** 2 + y21 ** 2)
            tlt0 = t < 0
            tle1 = (0 <= t) & (t <= 1)
            # Normal intersects side
            d[tle1] = (x1p[tle1] + t[tle1] * x21) ** 2 + (y1p[tle1] + t[tle1] * y21) ** 2
            # Normal does not intersects side
            # Point is closest to vertex (y1, x1)
            # Compute square of distance to this vertex
            d[tlt0] = x1p[tlt0] ** 2 + y1p[tlt0] ** 2
            # Store distances
            mask = d < mindst
            mindst[mask] = d[mask]
            j[mask] = i
            # Point is closer to (y1, x1) than any other vertex or side
            snear[mask & tlt0] = False
            # Point is closer to this side than to any other side or vertex
            snear[mask & tle1] = True

        if np.ma.count(snear) != snear.size:
            raise IndexError("Error computing distances")

        mindst **= 0.5
        # Point is closer to its nearest vertex than its nearest side, check if
        # nearest vertex is concave.
        # If the nearest vertex is concave then point is inside the polygon,
        # else the point is outside the polygon.
        jo = j.copy()
        jo[j == 0] -= 1
        area = _tri_2area_det([y[j + 1], y[j], y[jo - 1]], [x[j + 1], x[j], x[jo - 1]])
        mindst[~snear] = np.copysign(mindst, area)[~snear]
        # Point is closer to its nearest side than to its nearest vertex, check
        # if point is to left or right of this side.
        # If point is to left of side it is inside polygon, else point is
        # outside polygon.
        area = _tri_2area_det([y[j], y[j + 1], ypoint], [x[j], x[j + 1], xpoint])
        mindst[snear] = np.copysign(mindst, area)[snear]
        # Point is on side of polygon
        mindst[np.fabs(mindst) < smalld] = 0
        # If input values were scalar then the output should be too
        if scalar:
            mindst = float(mindst)
        return mindst
