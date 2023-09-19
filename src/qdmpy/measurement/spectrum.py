# -*- coding: utf-8 -*-
"""
This module holds tools for reading in and plotting an ODMR spectrum.
"""

# ============================================================================

__author__ = "David Broadway"

# ============================================================================
# IMPORTS 
import scipy
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================

# import qdmpy.shared.fourier
# import qdmpy.shared.itool

# ============================================================================

class ODMR:
    def __init__(self, path, label=None):
        self.path = path
        self.label = label
        self.freqs, self.sig, self.ref = self.read_odmr_spectrum(path, label)
        self.norm_vals = (self.sig / self.ref) * 100
        self.norm_vals = self.norm_vals - np.min(self.norm_vals)
        self.fit_result = None
        # self.norm_vals = self.norm_vals / np.max(self.norm_vals)

    def read_odmr_spectrum(self, path, label=None):
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
        data = np.genfromtxt(path, skip_header=1, usecols=(0, 1, 2))
        # Define the start and end of the data
        START = 0
        END = None
        freqs = data[START:END, 0]
        sig = data[START:END, 1]
        ref = data[START:END, 2]
        return freqs, sig, ref

    ########################################
    #           FITTING FUNCTIONS          #
    ########################################

    def define_fit_model(self, model_name):
        if model_name == 'gaussian':
            self.model = self.gaussian
        elif model_name == 'lorentzian':
            self.model = self.lorentzian
        else:
            print('Model not implemented yet.')
        return

    # function that contains a guassian model for fitting
    def gaussian(self, x, amp, cen, wid):
        return amp * np.exp(-(x - cen)**2 / wid)
    
    # function that contains a lorentzian model for fitting
    def lorentzian(self, x, amp, cen, wid):
        return amp * wid**2 / ((x - cen)**2 + wid**2)

    def fit_model(self, ax=None, plot=False, **kwargs):
        freqs = self.freqs
        norm_vals = self.norm_vals
        # Fit the data using a Gaussian
        # Initial guess for the fit
        amp_guess = np.max(norm_vals)
        cen_guess = freqs[np.argmax(norm_vals)]
        wid_guess = 1
        p0 = [amp_guess, cen_guess, wid_guess]
        # Do the fit
        popt, pcov = scipy.optimize.curve_fit(self.model, freqs, norm_vals, p0=p0)
        # Plot the fit
        if plot:
            self.plot_fit(freqs, norm_vals, self.model(freqs, *popt), ax=ax, **kwargs)
        # write the fit values to the object
        self.fit_result = self.model(freqs, *popt)
        self.fit_params = popt
        self.fit_cov = pcov
        return

    ########################################
    #           PLOTTING FUNCTIONS         #
    ########################################

    def plot(self, ax=None, plot_fit=False, **kwargs):
        if ax is None:
            ax = plt.gca()
        if plot_fit and self.fit_result is not None:
            ax.plot(self.freqs, self.norm_vals, '.', **kwargs)
            ax.plot(self.freqs, self.fit_result, '-', color=plt.gca().lines[-1].get_color(), label=self.label, **kwargs)
        else:
            ax.plot(self.freqs, self.norm_vals, label=self.label, **kwargs)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Norm. Contrast (%)')
        ax.tick_params(direction="in")
        ax.legend()
    
    def plot_fit(self, freqs, norm_vals, fit_values, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(freqs, norm_vals, '.', label='data', color = 'tab:grey')
        ax.plot(freqs, fit_values,label = 'fit', color = 'tab:blue')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Norm. Contrast (%)')
        ax.tick_params(direction="in")
        plt.legend()

    # function to plot multiple ODMR spectra contained in different objects
    def plot_odmr_spectra(self, odmr_list, plot_fit=False, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        for odmr in odmr_list:
            odmr.plot(ax=ax, plot_fit=plot_fit, **kwargs)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Norm. Contrast (%)')
        ax.tick_params(direction="in")
        plt.tight_layout()
        ax.legend()
        

# Define the PL Spectrum class and additional functions
class PL:
    def __init__(self, path, label=None):
        self.path = path
        self.label = label
        self.wavelength, self.intensity = self.read_spectrum(path, label)
        self.norm_intensity = (self.intensity / np.sum(self.intensity)) 
        self.fit_result = None
        # self.norm_vals = self.norm_vals / np.max(self.norm_vals)

    def read_spectrum(self, path, label=None):
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
        data = np.genfromtxt(path, skip_header=1, usecols=(0, 1))
        # Define the start and end of the data
        START = 0
        END = None
        wavelength = data[START:END, 0]
        intensity = data[START:END, 1]
        return wavelength, intensity

    ########################################
    #           PLOTTING FUNCTIONS         #
    ########################################

    def plot(self, ax=None, plot_fit=False, normalise=False, xlim=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if normalise:
            intensity = self.intensity/np.max(self.intensity)
        else:
            intensity = self.intensity

        if plot_fit and self.fit_result is not None:
            ax.plot(self.wavelength, intensity, '.', **kwargs)
            if normalise:
                ax.plot(self.wavelength, fit_result/np.max(self.intensity), '-', color=plt.gca().lines[-1].get_color(), label=self.label, **kwargs)
            else:
                ax.plot(self.wavelength, fit_result, '-', color=plt.gca().lines[-1].get_color(), label=self.label, **kwargs)
        else:
            ax.plot(self.wavelength, intensity, label=self.label, **kwargs)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity (cps)')
        if xlim is not None:
            plt.xlim((xlim[0],xlim[1]))
        ax.tick_params(direction="in")
        ax.legend()
    

    # function to plot multiple spectra contained in different objects
    def plot_spectra(self, spectra, plot_fit=False, normalise=False, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        for spectrum in spectra:
            spectrum.plot(ax=ax, plot_fit=plot_fit, normalise=normalise, **kwargs)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Counts (a.u.)')
        ax.tick_params(direction="in")
        plt.tight_layout()
        ax.legend()
        
    def plot_fit(self,fit_values, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.wavelength, self.intensity, '.', label='data', color = 'tab:grey')
        ax.plot(self.wavelength, fit_values,label = 'fit', color = 'tab:blue')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Counts (a.u.)')
        ax.tick_params(direction="in")
        plt.legend()

    ########################################
    #           FITTING FUNCTIONS          #
    ########################################

    def define_fit_model(self, model_name):
        if model_name == 'gaussian':
            self.model = self.gaussian
        elif model_name == 'lorentzian':
            self.model = self.lorentzian
        else:
            print('Model not implemented yet.')
        return

    # function that contains n guassian models for fitting 
    def gaussian(self, x, *args):
        x = np.array(x).reshape(-1, 1)
        amp = np.array(args[0::3]).reshape(1, -1)
        cen = np.array(args[1::3]).reshape(1, -1)
        wid = np.array(args[2::3]).reshape(1, -1)
        # print(x.shape)
        # print(amp.shape)
        return np.sum(amp * np.exp(-(x - cen)**2 / wid), axis=1)
    
        # function that contains a lorentzian model for fitting
    def lorentzian(self, x, *args):
        x = np.array(x).reshape(-1, 1)
        amp = np.array(args[0::3]).reshape(1, -1)
        cen = np.array(args[1::3]).reshape(1, -1)
        wid = np.array(args[2::3]).reshape(1, -1)
        return np.sum(amp * wid**2 / ((x - cen)**2 + wid**2), axis=1)
    

    def fit_model(self, ax=None, plot=False, n_peaks = 1, **kwargs):
        wavelength = self.wavelength
        intensity = self.intensity
        # Fit the data using a Gaussian
        # Initial guess for the fit
        amp_guess = np.max(intensity)
        cen_guess = wavelength[np.argmax(intensity)]
        wid_guess = 1
        p0 = [amp_guess, cen_guess, wid_guess] * n_peaks
        # Do the fit
        popt, pcov = scipy.optimize.curve_fit(self.model, wavelength, intensity, p0=p0)
        # Plot the fit
        if plot:
            self.plot_fit(self.model(wavelength, *popt), ax=ax, **kwargs)
        # write the fit values to the object
        self.fit_result = self.model(wavelength, *popt)
        self.fit_params = popt
        self.fit_cov = pcov
        print('Fit parameters: ')
        print('Amplitudes: ' + str(popt[0::3]))
        print('Centers: ' + str(popt[1::3]))
        print('Widths: ' + str(popt[2::3]))
        return    