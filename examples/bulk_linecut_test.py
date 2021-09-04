import matplotlib.pyplot as plt
import numpy as np
from qdmpy.shared.linecut import BulkLinecutWidget

times = [0.325, 1, 5, 10, 20, 21, 22, 25, 30, 40]
paths = [f"/home/samsc/share/{t}.txt" for t in times]
images = [np.loadtxt(p) for p in paths]
selector_image = images[4]

fig, axs = plt.subplots(ncols=3, figsize=(12, 6))
axs[0].imshow(selector_image)  # (data may be nans if you want empty selector)
selector = BulkLinecutWidget(*axs, images, times)
plt.show()
selector.disconnect()
