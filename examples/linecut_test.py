import matplotlib.pyplot as plt
import numpy as np
from qdmpy.shared.linecut import LinecutSelectionWidget

fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
data = np.loadtxt("/home/samsc/share/Jx_without_ft.txt")
axs[0].imshow(data)  # (data may be nans if you want empty selector)
selector = LinecutSelectionWidget(axs[0], axs[1], data, useblit=False)
plt.show()
selector.disconnect()
