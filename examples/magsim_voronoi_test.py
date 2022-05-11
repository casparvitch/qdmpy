# run me as: python3 <name>.py
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
json_output_path = "/home/samsc/ResearchData/test_images/mz_test/polys_mz_sandbox.json"
json_input_path = "/home/samsc/ResearchData/test_images/mz_test/polys.json"
domains_output_path = "/home/samsc/ResearchData/test_images/mz_test/polys_domain.json"
voronoi_output_path = "/home/samsc/ResearchData/test_images/mz_test/tesselate.pickle"
mag_path = "/home/samsc/ResearchData/test_images/mz_test/tess_mag.pickle"
mean_plus_minus = 0.25

base_patch = {
    "facecolor": None,
    "linestyle": "-",
    "fill": False,
    "linewidth": 1,  # 3,
}

mesh_size = 1024
# mesh_size = 256
height = 290e-9
res = 700e-9
# res = 2e-6
fov_size = 30e-6
num_domains = 1000

# colors = [f"C{i % 10}" for i in range(num_domains)]
# pgon_patches = [{**base_patch, **{"edgecolor": c}} for c in colors]
pgon_patches = {**base_patch, "edgecolor": "k"}

sim = qdmpy.magsim.VoronoiMagSim((mesh_size, mesh_size), (fov_size, fov_size))

# sim.add_template_polygons(json_input_path)
# sim.adjust_template(output_path=json_output_path, mean_plus_minus=mean_plus_minus)
# sim.set_template_as_polygons()

# sim.add_polygons(json_output_path)
sim.select_polygons(output_path=json_output_path)

sim.add_domain_sources(num_domains, polygon_idx=0)
# sim.load_domain_sources(domains_output_path)
# sim.save_domain_sources(domains_output_path)

# sim.plot_domains(polygon_patch_params=pgon_patches, fontsize=3, markersize=5)

# sim.save_voronoi(voronoi_output_path)
# sim.load_voronoi(voronoi_output_path)

# n = functools.reduce(operator.add, (1 for u in sim.domain_sources.values() for i in u))
n = len(sim.polygon_nodes)

# mags = [5 * (2 * (i % 2) - 1) for i in range(num_domains + 2)]
mags = [5 * (2 * (i % 2) - 1) for i in range(n)]

sim.define_magnets(
    mags,
    (0, 0, 1),
)  # mag unit: mu_b/nm^2
# sim.save_magnets(mag_path)  # only save/load mag data, ensure compatible with domains/polygons.
# sim.load_magnets(mag_path)

sim.crop_magnetization_gui()

sim.plot_magsim_magnetizations(
    annotate_polygons=True, polygon_patch_params=pgon_patches
)

sim.run(height, pad_mode="constant", resolution=res)

unv = [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]
sim.plot_magsim_bfield_at_nvs(projection=unv, polygon_patch_params=pgon_patches)
# sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=(1, 0, 0))
# sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=(0, 1, 0))
# sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=(0, 0, 1))

# plotters return fig, ax so e.g. run: fig, _ = sim.plot_magsim_bfield_at_nvs(...); fig.savefig(path)

plt.show()
