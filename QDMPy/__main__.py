# -*- coding: utf-8 -*-

__author__ = "Sam Scholten"


# import QDMPy.fitting as fitting
# import QDMPy.data_loading as data_loading
# import QDMPy.fit_plots as fit_plots

# import matplotlib.pyplot as plt

# # FIXME this is *very* out of date.
# def main(__spec__=None):
#     options = data_loading.load_options(
#         path="QDMPy/options/fit_options.json", check_for_prev_result=True
#     )
#     fit_plots.set_mpl_rcparams(options)

#     raw_data, prelim_sweep_list = data_loading.load_image_and_sweep(options)

#     PL_image, PL_image_ROI, sig, ref, sig_norm, sweep_list = data_loading.reshape_dataset(
#         options, raw_data, prelim_sweep_list
#     )
#     fit_model = fitting.define_fit_model(options)

#     fig1 = fit_plots.plot_ROI_PL_image(options, PL_image)

#     AOIs = data_loading.define_AOIs(options)

#     fig2 = fit_plots.plot_AOI_PL_images(options, PL_image_ROI, AOIs)

#     aoi_spectra_plot = fit_plots.plot_AOI_spectra(options, AOIs, sig, ref, sweep_list)

#     fit_model = fitting.define_fit_model(options)

#     roi_avg_fit_result = fitting.fit_ROI_avg(options, sig_norm, sweep_list, fit_model)

#     fig3 = fit_plots.plot_ROI_avg_fit(options, roi_avg_fit_result)

#     AOI_avg_best_fit_results_lst = fitting.fit_AOIs(
#         options, sig_norm, sweep_list, fit_model, AOIs, roi_avg_fit_result
#     )

#     AOI_spectra_fit_fig = fit_plots.plot_AOI_spectra_fit(
#         options,
#         sig,
#         ref,
#         sweep_list,
#         AOIs,
#         AOI_avg_best_fit_results_lst,
#         roi_avg_fit_result,
#         fit_model,
#     )

#     if (options["force_fit"] or not options["found_prev_result"]) and options["fit_pixels"]:
#         pixel_fit_params = fitting.fit_pixels(
#             options, sig_norm, sweep_list, fit_model, roi_avg_fit_result
#         )
#     else:
#         pixel_fit_params = fitting.load_prev_fit_results(options)

#     fit_plots.plot_param_images(options, fit_model, pixel_fit_params, "c")

#     fit_plots.plot_param_images(options, fit_model, pixel_fit_params, "m")

#     fit_plots.plot_param_images(options, fit_model, pixel_fit_params, "pos")

#     fit_plots.plot_param_images(options, fit_model, pixel_fit_params, "fwhm")

#     fit_plots.plot_param_images(options, fit_model, pixel_fit_params, "amp")

#     plt.show()


# if __name__ == "__main__":
#     main()
