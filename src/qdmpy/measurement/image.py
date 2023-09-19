# Script to compare different ODMR spectra

import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt


class PL:
    def __init__(self, path, pixel_size=None, roi=None, label=None):
        self.path = path
        self.label = label
        self.image = self.read_image(path)
        self.max_intensity = np.max(self.image)
        self.pixel_size = pixel_size # in nm
        if roi is not None:
            self.image = self.define_region_of_interest(roi)
        # self.norm_vals = self.norm_vals / np.max(self.norm_vals)

    def read_image(self, path):
        # reads the ODMR spectrum from a .txt file
        # path: path to the .txt file
        # label: label for the plot
        # outputs:
        #   freqs: frequencies in MHz
        #   norm_vals: normalized counts

        # check if the .txt extention is given
        if '.txt' not in path:
            path += '.txt'  
        # Read the data
        image = np.genfromtxt(path)
        return image

    def define_region_of_interest(self, roi):
        # This function defines the region of interest and returns the image of this region
        # roi: list of the form [x0, x1, y0, y1]
        # outputs:
        #   image_roi: image of the region of interest
        #   roi: list of the form [x0, x1, y0, y1]

        # Define the region of interest
        x0, x1, y0, y1 = roi
        image_roi = self.image[y0:y1, x0:x1]
        return image_roi



    ########################################
    #           FITTING FUNCTIONS          #
    ########################################

    ########################################
    #           PLOTTING FUNCTIONS         #
    ########################################

    def plot_image(self, ax=None, cmap='inferno', vmin=None, vmax=None, **kwargs):
        # This function plots the image
        # ax: axis to plot the image on
        # cmap: colormap
        # vmin: minimum value for the colormap
        # vmax: maximum value for the colormap
        # kwargs: additional arguments for the plot function
        # outputs:
        #   ax: axis with the plot

        if ax is None:
            fig, ax = plt.subplots()

        plt.imshow(self.image, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        plt.title(self.label)


        # Add a colorbar
        cbar = plt.colorbar()
        cbar.set_label('Intensity (a.u.)')

        # add a scale bar
        if self.pixel_size is not None:
            scalebar = ScaleBar(self.pixel_size, 'nm', length_fraction=0.25, location='lower right', color='w', box_color='k', border_pad=0.5, sep=5, frameon=False)
            ax.add_artist(scalebar)
            
            ax.set_xticks([])
            ax.set_yticks([])
        return ax
        
