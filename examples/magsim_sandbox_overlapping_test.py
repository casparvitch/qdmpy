# run me as: python3 magsim_sandbox_test.py
import qdmpy
import qdmpy.magsim
import qdmpy.shared.json2dict
import qdmpy.shared.fourier

import numpy as np
import matplotlib.pyplot as plt

overlapping_path = "/home/samsc/ResearchData/test_images/mz_test/overlapping.json"
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
# sim.select_polygons(output_path=overlapping_path)
sim.add_polygons(overlapping_path)

sim.define_magnets((5, 2), (0, 0, 1))  # mag unit: mu_b/nm^2
sim.plot_magsim_magnetizations(annotate_polygons=True, polygon_patch_params=pgon_patch)

sim.run(
    height, pad_mode="constant", resolution=res
)  # height: 'PX' equivalent in z, res the same

unv = [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]
sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=unv)

# plotters return fig, ax so e.g. run: fig, _ = sim.plot_magsim_bfield_at_nvs(...); fig.savefig(path)

plt.show()
