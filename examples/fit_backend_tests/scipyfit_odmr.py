import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import qdmpy
    import qdmpy.pl
    import qdmpy.plot
    import qdmpy.field
    import qdmpy.source

    import matplotlib
    import numpy as np

    exec(open("../TEST_DATA_PATH.py").read())

    options_dict = {
        "base_dir": TEST_DATA_PATH,  # var is read from TEST_DATA_PATH.py
        "filepath": "j_test/ODMR - Pulsed_42",
        "custom_output_dir_suffix": "_{fit_backend}_bin_{total_bin}",
        "additional_bins": 12,
        "objective_mag": 40,
        "system_name": "Zyla",
        "ROI": "Full",
        "ROI_start": [10, 30],
        "ROI_end": [20, 40],
        "AOI_1_start": [30, 40],
        "AOI_1_end": [40, 45],
        "AOI_2_start": [20, 20],
        "AOI_2_end": [24, 24],
        "AOI_3_start": [30, 20],
        "AOI_3_end": [32, 24],
        "single_pixel_check": [90, 150],
        "fit_backend": "scipyfit",
        "fit_backend_comparison": ["scipyfit"],
        "fit_pl_pixels": True,
        "force_fit": True,
        "use_ROI_avg_fit_res_for_all_pixels": True,
        "scipyfit_sub_threads": 2,
        "fit_functions": {"linear": 1, "lorentzian": 8},
        # "pos_guess": [2635, 2724, 2785, 2865, 2958, 3020, 3071, 3130],
        "pos_guess": [2630, 2724, 2785, 2864, 2953, 3025, 3067, 3129],
        "pos_range": 25,
        "amp_guess": -0.015,
        "amp_bounds": [-0.0300, -0.0003],
        "save_fig_type": "pdf",
        "field_method": "auto_dc",
        "freqs_to_use": [1, 1, 1, 1, 1, 1, 1, 1],
        "diamond_ori": "<100>_<110>",  # CVD
        "bias_mag": 100,
        "bias_theta": 67,
        "bias_phi": -120,
        "calc_field_pixels": True,
        "force_field_calc": True,
        "bfield_bground_method": "poly",
        "bfield_bground_params": {"order": 1},
        "bnv_bground_method": "poly",
        "bnv_bground_params": {"order": 1},
        "source_bground_method": None,  # "poly",
        "source_bground_params": {"order": 1},
        "colormap_range_dicts": {
            "current_vector_images": {"type": "strict_range", "values": [-50, 50]},
            "current_norm_images": {"type": "strict_range", "values": [0, 50]},
            "current_div_images": {"type": "percentile", "values": [5, 95]},
            "bfield_images": {"type": "percentile", "values": [2, 98]},
        },
        "NVs_above_sample": True,
        "Bx_range": 100,
        "By_range": 100,
        "Bz_range": 100,
        "recon_methods": ["from_bxy", "from_bz", "from_bnv", "without_ft"],
    }
    # if you want to use a reference experiment {ensure you run this even if 'None' as it sets up output dirs etc.}
    ref_options_dir = None

    options, ref_options = qdmpy.initialize(
        options_dict=options_dict,
        ref_options_dir=ref_options_dir,
        set_mpl_rcparams=True,
    )

    image, prelim_sweep_list = qdmpy.pl.load_image_and_sweep(options)
    (
        PL_image,
        PL_image_ROI,
        sig,
        ref,
        sig_norm,
        single_pixel_pl,
        sweep_list,
        ROI,
    ) = qdmpy.pl.reshape_dataset(options, image, prelim_sweep_list)

    ref_fit_params, ref_sigmas = qdmpy.pl.load_ref_exp_pl_fit_results(ref_options)

    fit_model = qdmpy.pl.define_fit_model(options)
    backend_ROI_results_lst = qdmpy.pl.fit_roi_avg_pl(
        options, sig, ref, sweep_list, fit_model
    )
    # ROI_fit_fig = qdmpy.plot.roi_avg_fits(options, backend_ROI_results_lst)

    fit_result_collection_lst = qdmpy.pl.fit_aois_pl(
        options,
        sig,
        ref,
        single_pixel_pl,
        sweep_list,
        fit_model,
        backend_ROI_results_lst,
    )
    # AOI_fit_fig = qdmpy.plot.aoi_spectra_fit(
    #     options, sig, ref, sweep_list, fit_result_collection_lst, fit_model
    # )

    wanted_roi_result = next(
        filter(
            lambda result: result.fit_backend == options["fit_backend"],
            backend_ROI_results_lst,
        )
    )  # ROI fit result for chosen fit backend
    pixel_fit_params, sigmas = qdmpy.pl.get_pl_fit_result(
        options, sig_norm, sweep_list, fit_model, wanted_roi_result
    )

    # qdmpy.pl.save_pl_fit_results(options, pixel_fit_params)
    # qdmpy.pl.save_pl_fit_sigmas(options, sigmas)

    # field_res = qdmpy.field.odmr_field_retrieval(options, pixel_fit_params, ref_fit_params)
    # (
    #     (sig_bnvs, ref_bnvs, bnvs),
    #     (sig_dshifts, ref_dshifts),
    #     (sig_params, ref_params, field_params),
    #     (sig_field_sigmas, ref_field_sigmas, field_sigmas),
    # ) = field_res
    # qdmpy.field.save_field_calcs(options, *field_res)

    # bnvs_plot = qdmpy.plot.bnvs_and_dshifts(options, "sig_sub_ref", bnvs, [])

    # qdmpy.save_options(options)
    # import qdmpy.shared.json2dict

    print(options["fit_time_(s)"])
    # print(qdmpy.shared.json2dict.dict_to_json_str(options))

    # import matplotlib.pyplot as plt

    # plt.show()
