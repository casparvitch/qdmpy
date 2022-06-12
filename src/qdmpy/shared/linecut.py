# -*- coding: utf-8 -*-
"""
This module contains...

FIXME TODO docs, once code settles
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {"qdmpy.shared.": True}

# ============================================================================

import numpy as np
from matplotlib.image import AxesImage
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pathlib
import matplotlib.patches

# ============================================================================

import qdmpy.shared.widget
import qdmpy.shared.json2dict
import qdmpy.shared.itool

# ============================================================================


def bulk_vert_linecut_vs_position(
    image_to_show, images, times, averaging_width=1, sigma=1, linecut_coords=None
):
    # prev args: image_to_show, images, times, middle_idx, averaging_width=1, sigma=1

    if not isinstance(averaging_width, int) or not (averaging_width % 2):
        raise TypeError("averaging_width must be an odd int.")
    avg_steps = averaging_width - 1 // 2
    fig, axs = plt.subplots(ncols=2, figsize=(12, 8))

    imgsf = [
        qdmpy.shared.itool.get_im_filtered(image, "gaussian", sigma=sigma)
        for image in images
    ]
    height, width = image_to_show.shape

    # if middle_idx > width:
    # raise ValueError(f"start_idx too large for shape: ({width, height}).")

    furthest = max([np.nanmin(image_to_show), np.nanmax(image_to_show)])
    axs[0].imshow(image_to_show, cmap="bwr", vmin=-furthest, vmax=furthest)
    # axs[0].axvline(middle_idx, color="k", ls="--", lw=2)

    if linecut_coords is not None:
        start_y, end_y = linecut_coords
        p = matplotlib.patches.Polygon(
            [[width // 2, start_y], [width // 2, end_y]], closed=False, ec="k", ls="--"
        )
        axs[0].add_patch(p)

    cols = list(range(avg_steps, width - avg_steps))
    col_labels = [col_idx for col_idx in cols]
    x_pos = [col_labels for _ in times]
    x_pos = np.transpose(x_pos)

    img_profiles = []
    for data in imgsf:
        profiles = []
        for centre_col in cols:
            if linecut_coords is None:
                prof = np.zeros(height)
            else:
                prof = np.zeros(1 + end_y - start_y)
            for col in range(centre_col - avg_steps, centre_col + avg_steps + 1):
                if linecut_coords is None:
                    prof += data[:, col]
                else:
                    prof += data[start_y : (end_y + 1), col]  # noqa: E203
            prof /= averaging_width
            profiles.append(prof)
        img_profiles.append(profiles)

    img_integrals = []
    for img in img_profiles:
        integrals = [integrate.simpson(prof) for prof in img]
        img_integrals.append(integrals)

    integral_series = np.transpose(img_integrals)  # shape for plotting

    colors = cm.plasma_r(np.linspace(0.1, 0.9, len(times)))
    axs[1].set_prop_cycle("color", colors)

    axs[1].plot(x_pos, integral_series, "o--", ms=2.5, label=times)
    axs[1].axvline(0, color="k", zorder=0)
    axs[1].axhline(0, color="k", zorder=0)

    handles, _ = axs[1].get_legend_handles_labels()
    # tot_num = len(handles)
    # interval = tot_num // 5
    # new_handles = [handles[i] for i in range(0, tot_num, interval)]
    # new_labels = [cols[i] for i in range(0, tot_num, interval)]
    # axs[1].legend(handles, new_labels, title="x position")
    axs[1].legend(title="time (us)")
    axs[1].set_xlabel("Position in x (PX)")
    axs[1].set_ylabel("I_x (prop. to A)")

    return integral_series


# ============================================================================


def vert_linecut_vs_position(
    inpt, output_path, averaging_width=1, alpha=0.5, sigma=1, linecut_coords=None
):
    if not isinstance(averaging_width, int) or not (averaging_width % 2):
        raise TypeError("averaging_width must be an odd int.")
    avg_steps = averaging_width - 1 // 2

    fig, axs = plt.subplots(ncols=3, figsize=(16, 8))

    data = np.loadtxt(inpt) if isinstance(inpt, (str, pathlib.PurePath)) else inpt
    data = qdmpy.shared.itool.get_im_filtered(data, "gaussian", sigma=sigma)

    height, width = data.shape

    furthest = max([np.nanmin(data), np.nanmax(data)])
    axs[0].imshow(data, cmap="bwr", vmin=-furthest, vmax=furthest)

    cols = list(range(avg_steps, width - avg_steps))
    col_labels = [col_idx for col_idx in cols]

    if (
        linecut_coords is not None
    ):  # clean this stuff up, assume a linecut and fall back to 0->height
        start_y, end_y = linecut_coords
        p = matplotlib.patches.Polygon(
            [[width // 2, start_y], [width // 2, end_y]], closed=False, ec="k", ls="--"
        )
        axs[0].add_patch(p)

    profiles = []
    for centre_col in cols:
        if linecut_coords is None:
            prof = np.zeros(height)
        else:
            prof = np.zeros(1 + end_y - start_y)
        for col in range(centre_col - avg_steps, centre_col + avg_steps + 1):
            if linecut_coords is None:
                prof += data[:, col]
            else:
                prof += data[start_y : (end_y + 1), col]  # noqa: E203
        prof /= averaging_width
        profiles.append(prof)

    integrals = [integrate.simpson(prof) for prof in profiles]
    profiles = np.transpose(profiles)  # shape for plotting

    colors = cm.cool(np.linspace(0, 1, len(cols)))
    axs[1].set_prop_cycle("color", colors)
    axs[1].plot(profiles, "--", alpha=alpha, label=cols)

    handles, _ = axs[1].get_legend_handles_labels()
    tot_num = len(handles)
    interval = tot_num // 5
    new_handles = [handles[i] for i in range(0, tot_num, interval)]
    new_labels = [cols[i] for i in range(0, tot_num, interval)]
    axs[1].legend(new_handles, new_labels, title="x position")

    axs[2].scatter(col_labels, integrals)
    axs[2].axvline(0, color="k", zorder=0)
    axs[2].axhline(0, color="k", zorder=0)

    odict = {
        "integral_column_labels": col_labels,
        "integral_values": integrals,
        "profiles_formatted_for_mpl_plot": profiles,
    }
    qdmpy.shared.json2dict.dict_to_json(odict, output_path)


# ============================================================================


class BulkLinecutWidget:
    """
    How to use
    ----------
    import matplotlib.pyplot as plt
    import numpy as np
    from qdmpy.shared.linecut import BulkLinecutWidget

    path = "<WHATEVER>"
    times = [0.325, 1, 5, 10, 20, 21, 22, 25, 30, 40]
    paths = [f"{path}/{t}.txt" for t in times]
    images = [np.loadtxt(p) for p in paths]
    selector_image = images[4]

    fig, axs = plt.subplots(ncols=3, figsize=(12, 6))
    axs[0].imshow(selector_image)  # (data can be nans if you want an empty selector)
    selector = BulkLinecutWidget(*axs, images, times)
    plt.show()
    selector.disconnect(path="/home/samsc/share/result.json")
    """

    def __init__(self, imax, profax, resax, images, xlabels, style=None, useblit=False):
        # check that input ax has an imshow (else not so useful eh)
        if not any([[isinstance(t, AxesImage) for t in imax.get_children()]]):
            raise ValueError("input axis does not contain an AxesImage (imshow).")
        self.images = images
        self.xlabels = xlabels

        self.pts = []

        dflt_style = {
            "lineprops": {
                "color": "k",
                "linestyle": "-",
                "linewidth": 1.0,
                "alpha": 0.5,
            },
            "markerprops": {
                "marker": "o",
                "markersize": 2.0,
                "mec": "k",
                "mfc": "k",
                "alpha": 0.5,
            },
        }

        self.lp = dflt_style["lineprops"]
        self.mp = dflt_style["markerprops"]

        if style is not None:
            if "lineprops" in style and isinstance(style["lineprops"], dict):
                for key, item in style["lineprops"].items():
                    self.lp[key] = item
            if "markerprops" in style and isinstance(style["markerprops"], dict):
                for key, item in style["markerprops"].items():
                    self.mp[key] = item

        self.line_selector = qdmpy.shared.widget.LineSelector(
            imax,
            self.onselect,
            ondraw=self.ondraw,
            lineprops=self.lp,
            markerprops=self.mp,
            vertex_select_radius=7.5 * 2.0,
            useblit=useblit,
        )
        self.imax = imax
        self.profax = profax
        self.resax = resax
        self.integrals = [0 for i in xlabels]
        self.integral = 0

        self.canvas = self.imax.figure.canvas

        dummy_x = np.zeros((5, len(xlabels)))
        dummy_y = np.zeros((5, len(xlabels)))
        self.profiles = self.profax.plot(
            dummy_x, dummy_y, marker="o", ls="-", label=xlabels
        )
        # handles, _ = self.profax.get_legend_handles_labels()
        # self.profax.legend(handles, self.xlabels, loc="upper left")
        self.profax.legend()

        (self.integrals_plot,) = self.resax.plot(xlabels, self.integrals, "ko-")

    def ondraw(self, verts):
        if len(verts) == 1:
            # change all profiles
            for p, prof in enumerate(self.profiles):
                prof.set_xdata([0])
                prof.set_ydata(self.images[p][int(verts[0][1]), int(verts[0][0])])
        else:
            idxs, jdxs = zip(*verts)
            pxl_ar = [0]
            i_lst = []
            j_lst = []
            for n in range(len(idxs) - 1):
                i0, i1 = idxs[n], idxs[n + 1]
                j0, j1 = jdxs[n], jdxs[n + 1]
                num = int(np.sqrt((i1 - i0) ** 2 + (j1 - j0) ** 2) * 2)  # *2 to be safe
                if not num:
                    continue

                ivec = np.linspace(i0, i1, num).astype(int)
                jvec = np.linspace(j0, j1, num).astype(int)

                new_ij = [
                    (ivec[v], jvec[u])
                    for v, u in zip(range(len(ivec) - 1), range(len(jvec) - 1))
                    if ivec[v + 1] != ivec[v] or jvec[u + 1] != jvec[u]
                ]
                new_ij.append((ivec[-1], jvec[-1]))
                new_is, new_js = zip(*new_ij)
                if not i_lst:
                    i_lst, j_lst = list(new_is), list(new_js)
                    i_ar, j_ar = np.asarray(new_is), np.asarray(new_js)
                else:
                    i_lst.extend(list(new_is[1:]))
                    j_lst.extend(list(new_js[1:]))
                    i_ar, j_ar = np.asarray(new_is[1:]), np.asarray(new_js[1:])
                if not (i_ar.ndim and i_ar.size):
                    continue
                pxl_ar.extend(
                    (
                        pxl_ar[-1]
                        + np.sqrt((i_ar - i_ar[0]) ** 2 + (j_ar - j_ar[0]) ** 2)
                    ).tolist()
                )

            for p, prof in enumerate(self.profiles):
                z = self.images[p][j_lst, i_lst]
                if z.ndim and z.size:  # ensure no empty array
                    prof.set_xdata(
                        pxl_ar[1:]
                    )  # get rid of initial 0 on pxl_ar (bit hacky)
                    prof.set_ydata(list(z))
                    self.integrals[p] = integrate.simpson(z, pxl_ar[1:])
                else:
                    # dummy data for this guy...
                    prof.set_xdata((np.nan,))
                    prof.set_ydata((np.nan,))
                    self.integrals[p] = np.nan
            self.integrals_plot.set_ydata(self.integrals)
        self.profax.relim()
        self.profax.autoscale_view()
        self.resax.relim()
        self.resax.autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()

    def onselect(self, verts):
        # only called when line is finished
        self.pts = verts  # list of vertices `[(Ax1, Ay1), (Ax2, Ay2)]`
        self.canvas.draw_idle()

    def disconnect(self, path=None):
        if path is not None:
            output_dict = {
                "xlabels": self.xlabels,
                "integrals": self.integrals,
                "profile_x": np.transpose(
                    [prof.get_xdata() for prof in self.profiles]
                ).tolist(),
                "profile_y": np.transpose(
                    [prof.get_ydata() for prof in self.profiles]
                ).tolist(),
            }
        qdmpy.shared.json2dict.dict_to_json(output_dict, path)
        self.line_selector.disconnect_events()
        self.canvas.draw_idle()


class LinecutSelectionWidget:
    """
    How to Use
    ----------
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(data) # (data may be nans if you want empty selector)
    selector = LinecutSelectionWidget(axs[0], axs[1], ...)
    plt.show()
    selector.disconnect()

    """

    def __init__(self, imax, lineax, data, style=None, useblit=False):
        # check that input ax has an imshow (else not so useful eh)
        if not any([[isinstance(t, AxesImage) for t in imax.get_children()]]):
            raise ValueError("input axis does not contain an AxesImage (imshow).")

        self.data = data
        self.imax = imax
        self.lineax = lineax
        self.integral = 0

        self.canvas = self.imax.figure.canvas

        dflt_style = {
            "lineprops": {
                "color": "k",
                "linestyle": "-",
                "linewidth": 1.0,
                "alpha": 0.5,
            },
            "markerprops": {
                "marker": "o",
                "markersize": 2.0,
                "mec": "k",
                "mfc": "k",
                "alpha": 0.5,
            },
        }

        self.lp = dflt_style["lineprops"]
        self.mp = dflt_style["markerprops"]
        if style is not None:
            if "lineprops" in style and isinstance(style["lineprops"], dict):
                for key, item in style["lineprops"].items():
                    self.lp[key] = item
            if "markerprops" in style and isinstance(style["markerprops"], dict):
                for key, item in style["markerprops"].items():
                    self.mp[key] = item

        vsr = 7.5 * self.mp["markersize"]  # linear scaling on what our select radius is

        (self.profile,) = self.lineax.plot([1, 2, 3], [1, 2, 3], "ko-")
        self.lineax.title.set_text(f"Integral: {self.integral}")

        self.pts = []
        self.line_selector = qdmpy.shared.widget.LineSelector(
            imax,
            self.onselect,
            ondraw=self.ondraw,
            lineprops=self.lp,
            markerprops=self.mp,
            vertex_select_radius=vsr,
            useblit=useblit,
        )

    def ondraw(self, verts):
        if len(verts) == 1:
            self.profile.set_xdata([0])
            self.profile.set_ydata(self.data[int(verts[0][1]), int(verts[0][0])])
            self.lineax.title.set_text(f"Integral: {self.integral}")
        else:
            idxs, jdxs = zip(*verts)
            t_ar = [0]
            i_lst = []
            j_lst = []
            for n in range(len(idxs) - 1):
                i0, i1 = idxs[n], idxs[n + 1]
                j0, j1 = jdxs[n], jdxs[n + 1]
                num = int(np.sqrt((i1 - i0) ** 2 + (j1 - j0) ** 2) * 2)  # *2 to be safe
                if not num:
                    continue

                ivec = np.linspace(i0, i1, num).astype(int)
                jvec = np.linspace(j0, j1, num).astype(int)

                new_ij = [
                    (ivec[v], jvec[u])
                    for v, u in zip(range(len(ivec) - 1), range(len(jvec) - 1))
                    if ivec[v + 1] != ivec[v] or jvec[u + 1] != jvec[u]
                ]
                new_ij.append((ivec[-1], jvec[-1]))
                new_is, new_js = zip(*new_ij)
                if not i_lst:
                    i_lst, j_lst = list(new_is), list(new_js)
                    i_ar, j_ar = np.asarray(new_is), np.asarray(new_js)
                else:
                    i_lst.extend(list(new_is[1:]))
                    j_lst.extend(list(new_js[1:]))
                    i_ar, j_ar = np.asarray(new_is[1:]), np.asarray(new_js[1:])
                if not (i_ar.ndim and i_ar.size):
                    continue
                t_ar.extend(
                    (
                        t_ar[-1]
                        + np.sqrt((i_ar - i_ar[0]) ** 2 + (j_ar - j_ar[0]) ** 2)
                    ).tolist()
                )

            z = self.data[j_lst, i_lst]
            if z.ndim and z.size:  # ensure no empty array
                self.profile.set_xdata(
                    t_ar[1:]
                )  # get rid of initial 0 on t_ar (bit hacky)
                self.profile.set_ydata(list(z))
                self.integral = integrate.simpson(z, t_ar[1:])
                self.lineax.title.set_text(f"Integral: {self.integral:.6e}")
        self.lineax.relim()
        self.lineax.autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()

    def onselect(self, verts):
        # only called when line is finished
        self.pts = verts  # list of vertices `[(Ax1, Ay1), (Ax2, Ay2)]`
        self.canvas.draw_idle()

    def disconnect(self):
        self.line_selector.disconnect_events()
        self.canvas.draw_idle()
        print()
        print("profile xdata:")
        print(self.profile.get_xdata())
        print("profile ydata:")
        print(self.profile.get_ydata())
        print("integral:")
        print(self.integral)
        print()

    # if want to include this func. change 'lines' to 'line' (etc.)
    # def load_nodes(self, polygon_nodes):
    #     # polygon nodes: list of polygons, each polygon is a list of nodes
    #     for polygon in polygon_nodes:
    #         nodes_ar = np.array(polygon)

    #         # x & y convention swapped here
    #         new_line = Line2D(nodes_ar[:, 1], nodes_ar[:, 0], **self.lp)
    #         self.ax.add_line(new_line)

    #         new_line_dict = dict(line_obj=new_line, xs=nodes_ar[:, 1], ys=nodes_ar[:, 0])

    #         self.line_selector.artists.append(new_line)
    #         self.line_selector.lines.append(new_line_dict)  # list of line dicts

    #     self.line_selector.draw_polygon()
