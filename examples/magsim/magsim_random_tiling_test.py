# run me as: python3 <name>.py
import qdmpy
import qdmpy.magsim
import qdmpy.shared.json2dict
import qdmpy.shared.fourier

import numpy as np
import matplotlib.pyplot as plt
import random

polys_path = "/home/samsc/ResearchData/test_images/mz_test/random_polys.pickle"
mag_path = "/home/samsc/ResearchData/test_images/mz_test/random_mag.pickle"
mean_plus_minus = 0.25

base_patch = {
    "facecolor": None,
    "linestyle": "-",
    "fill": False,
    "linewidth": 0.5,  # 3,
}

mesh_size = 1024
height = 290e-9
res = 700e-9
# res = 1e-6
fov_size = 30e-6
poly_shape = 6  # note: hexagons not so useful, unless using random mag direction.
domain_side_len = 0.1e-6

# colors = [f"C{i % 10}" for i in range(num_domains)]
# pgon_patches = [{**base_patch, **{"edgecolor": c}} for c in colors]
pgon_patches = {**base_patch, "edgecolor": "k"}

sim = qdmpy.magsim.TilingMagSim((mesh_size, mesh_size), (fov_size, fov_size))

# sim.define_tiling(poly_shape, domain_side_len)
# sim.plot_domains(polygon_patch_params=pgon_patches, fontsize=5, markersize=50)

# sim.crop_polygons_gui(show_polygons=False)

sim.add_polygons(polys_path)

# num_domains = len(sim.polygon_nodes)
# mags = [5 * (2 * (i % 2) - 1) for i in range(num_domains)]  # uniformly +- for neighbours
# mags = [5 * (random.randint(0, 1) * 2 - 1) for i in range(num_domains)]  # randomly up/down
# sim.define_magnets(mags, (0, 0, 1))  # mag unit: mu_b/nm^2

# sim.crop_magnetization_gui()

# sim.save_polygons(polys_path)
# sim.save_magnets(mag_path)  # only save/load mag data, ensure compatible with domains/polygons.
sim.load_magnets(mag_path)

sim.plot_magsim_magnetizations(
    annotate_polygons=False, polygon_patch_params=pgon_patches
)

sim.run(height, pad_mode="constant", resolution=res)

unv = [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]
vrange = (-0.05, 0.05)
vrange = None
fig, _ = sim.plot_magsim_bfield_at_nvs(
    projection=unv, polygon_patch_params=pgon_patches, strict_range=vrange
)
fig.savefig("/home/samsc/Desktop/fig_random.svg")


plt.show()
