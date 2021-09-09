import matplotlib.pyplot as plt
from qdmpy.shared.linecut import vert_linecut_vs_position

input_path = "/home/samsc/share/tseries_anu/22.txt"
output_path = "/home/samsc/share/tseries_anu/vert_linecuts_res.txt"

vert_linecut_vs_position(input_path, output_path, 165, averaging_width=11, alpha=0.8, sigma=2)

plt.show()
