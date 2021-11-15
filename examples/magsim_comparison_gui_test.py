# run me as: python3 magsim_comparison_test.py
import qdmpy
import qdmpy.magsim
import qdmpy.shared.json2dict

import numpy as np
import matplotlib.pyplot as plt

numpy_txt_file_path = (
    "/home/samsc/ResearchData/test_images/mz_test/ODMR -"
    " Pulsed_10_Rectangle_bin_8/field/sig_sub_ref/sig_sub_ref_bnv_0.txt"
)
json_output_path = (
    "/home/samsc/ResearchData/test_images/mz_test/polys_mz_comparison_gui.json"
)
# json_input_path = "/home/samsc/ResearchData/test_images/mz_test/polys.json"
# json_input_path = "/home/samsc/ResearchData/test_images/mz_test/polys_mz_comparison_gui.json"
mean_plus_minus = 0.25


pgon_patch = {
    "facecolor": None,
    "edgecolor": "xkcd:grass green",
    "linestyle": "dashed",
    "fill": False,
    "linewidth": 2,
}

sim = qdmpy.magsim.ComparisonMagSim(numpy_txt_file_path, (30e-6, 30e-6))
# sim.add_polygons(json_input_path)
sim.select_polygons(output_path=json_output_path, mean_plus_minus=mean_plus_minus)
sim.rescale(3)
sim.define_magnets(5, (0, 0, 1))
sim.plot_magsim_magnetizations(annotate_polygons=True, polygon_patch_params=pgon_patch)
sim.run(290e-9, pad_mode="constant", resolution=700e-9)
unv = [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]
sim.plot_magsim_bfield_at_nvs(
    strict_range=(-0.25, 0.25), projection=unv
)  # these return fig, ax
sim.plot_comparison(
    strict_range=(-0.25, 0.25), projection=unv
)  # so you could e.g. run: fig, _ = sim.plot_comparison(); fig.savefig(path)
plt.show()
