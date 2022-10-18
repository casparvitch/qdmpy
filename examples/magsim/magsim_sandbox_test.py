# run me as: python3 magsim_sandbox_test.py
import qdmpy
import qdmpy.magsim
import qdmpy.shared.json2dict
import qdmpy.shared.fourier

import numpy as np
import matplotlib.pyplot as plt

numpy_txt_file_path = (
    "/home/samsc/ResearchData/test_images/mz_test/ODMR -"
    " Pulsed_10_Rectangle_bin_8/field/sig_sub_ref/sig_sub_ref_bnv_0.txt"
)
json_output_path = (
    "/home/samsc/ResearchData/test_images/mz_test/polys_mz_sandbox.json"
)
json_input_path = "/home/samsc/ResearchData/test_images/mz_test/polys.json"
mean_plus_minus = 0.25

pgon_patch = {
    "facecolor": None,
    "edgecolor": "xkcd:grass green",
    "linestyle": "dashed",
    "fill": False,
    "linewidth": 2,
}

mesh_size = 512
height = 290e-9
res = 700e-9
fov_size = 30e-6

sim = qdmpy.magsim.SandboxMagSim((mesh_size, mesh_size), (fov_size, fov_size))
sim.add_template_polygons(json_input_path)

sim.adjust_template(
    output_path=json_output_path, mean_plus_minus=mean_plus_minus
)
sim.set_template_as_polygons()

sim.define_magnets(5, (0, 1, 0))  # mag unit: mu_b/nm^2
sim.plot_magsim_magnetizations(
    annotate_polygons=True, polygon_patch_params=pgon_patch
)

sim.run(
    height, pad_mode="constant", resolution=res
)  # height: 'PX' equivalent in z, res the same

unv = [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]
sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=unv)
sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=(1, 0, 0))
sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=(0, 1, 0))
sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=(0, 0, 1))

# plotters return fig, ax so e.g. run: fig, _ = sim.plot_magsim_bfield_at_nvs(...); fig.savefig(path)

plt.show()
