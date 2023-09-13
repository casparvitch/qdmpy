import matplotlib.pyplot as plt
import numpy as np
from qdmpy.shared.linecut import (
    vert_linecut_vs_position,
    bulk_vert_linecut_vs_position,
)
import qdmpy.shared.json2dict

SINGLE = False

if SINGLE:
    # input_path = "/home/samsc/share/tseries_anu/22.txt"
    # output_path = "/home/samsc/share/tseries_anu/vert_linecuts_res.txt"
    input_path = "C:\\Temp\\Jx_without_ft.txt"
    output_path = "C:\\Temp\\vert_linecuts_vs_time.json"

    # vert_linecut_vs_position(input_path, output_path, 165, averaging_width=11, alpha=0.8, sigma=2)
    vert_linecut_vs_position(
        input_path, output_path, 170, averaging_width=11, alpha=0.8, sigma=2
    )
else:
    averaging_width = 1
    folder = "no_bsub"
    times = [0.325, 1, 5, 10, 20, 21, 22, 25, 30, 40]
    # times = [20, 21, 22, 25, 30, 40]
    # times = [0.325, 1, 5, 10, 20]
    paths = [f"C:\\Temp\\qdmpy_output\\{folder}\\Jx_without_ft_{t}.txt" for t in times]
    output_path = f"C:\\Temp\\qdmpy_output\\{folder}\\integral_results_width_{averaging_width}.json"
    images = [np.loadtxt(p) for p in paths]
    image_to_show = images[2]
    integral_series = bulk_vert_linecut_vs_position(
        image_to_show,
        images,
        times,
        165,
        averaging_width=averaging_width,
        sigma=2,
    )
    output_dict = {"xlabels": times, "integrals": integral_series}
    qdmpy.shared.json2dict.dict_to_json(output_dict, output_path)

plt.show()
