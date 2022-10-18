import matplotlib.pyplot as plt
import numpy as np
from qdmpy.shared.linecut import BulkLinecutWidget

FOLDER = "sigma_2_bsub"

times = [0.325, 1, 5, 10, 20, 21, 22, 25, 30, 40]
# paths = [f"/home/samsc/share/tseries_anu/{t}.txt" for t in times]
# paths = [f"C:\\Temp\\qdmpy_output\\{FOLDER}\\Jx_without_ft_{t}.txt" for t in times]
paths = [
    f"/home/samsc/ResearchData/Photovoltaics/paper_data/tseries/{t}_Jx_without_ft.txt"
    for t in times
]
images = [np.loadtxt(p) for p in paths]
selector_image = images[4]

fig, axs = plt.subplots(ncols=3, figsize=(12, 6))
axs[0].imshow(
    selector_image
)  # (data can be nans if you want an empty selector)
selector = BulkLinecutWidget(*axs, images, times)
plt.show()
selector.disconnect(
    path="/home/samsc/ResearchData/Photovoltaics/paper_data/tseries/bulk_integral.json"
)
