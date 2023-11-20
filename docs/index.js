URLS=[
"qdmpy/index.html",
"qdmpy/pl/index.html",
"qdmpy/pl/common.html",
"qdmpy/pl/funcs.html",
"qdmpy/pl/interface.html",
"qdmpy/pl/io.html",
"qdmpy/pl/interface.html",
"qdmpy/pl/model.html",
"qdmpy/pl/scipyfit.html",
"qdmpy/pl/funcs.html",
"qdmpy/pl/common.html",
"qdmpy/plot/index.html",
"qdmpy/plot/field.html",
"qdmpy/plot/pl.html",
"qdmpy/plot/common.html",
"qdmpy/plot/source.html",
"qdmpy/shared/index.html",
"qdmpy/shared/fourier.html",
"qdmpy/shared/geom.html",
"qdmpy/shared/itool.html",
"qdmpy/shared/json2dict.html",
"qdmpy/shared/misc.html",
"qdmpy/shared/polygon.html",
"qdmpy/source/index.html",
"qdmpy/source/io.html",
"qdmpy/source/interface.html",
"qdmpy/source/current.html",
"qdmpy/source/magnetization.html",
"qdmpy/magsim/index.html",
"qdmpy/magsim/interface.html",
"qdmpy/interface.html",
"qdmpy/shared/index.html",
"qdmpy/shared/misc.html",
"qdmpy/shared/widget.html",
"qdmpy/shared/itool.html",
"qdmpy/shared/geom.html",
"qdmpy/shared/polygon.html",
"qdmpy/shared/linecut.html",
"qdmpy/shared/json2dict.html",
"qdmpy/shared/fourier.html",
"qdmpy/field/index.html",
"qdmpy/field/ham_scipyfit.html",
"qdmpy/field/bnv.html",
"qdmpy/field/io.html",
"qdmpy/field/interface.html",
"qdmpy/field/bxyz.html",
"qdmpy/field/hamiltonian.html",
"qdmpy/system/index.html",
"qdmpy/system/systems.html"
];
INDEX=[
{
"ref":"qdmpy",
"url":0,
"doc":"Quantum Diamond MicroscoPy: A module/package for analysing widefield NV microscopy images. 'Super-package' that holds all of the others within.                   = Subpackage dependency graph (DAG)                 = +             + +  -+ |  = | |qdmpy| | Key | +       -+  =+      + |  = | | +  -+ | | | | | | +  + | v v | |name| = Package | +   + + + +  -+ +   + +  + +   + | |  | | |system| |pl| |field| |source| |plot| |magsim| | +  + | |   | | | |  =| |   | |  | |   | | | + + -+ +- + + + + + -+ +-+ + + + -+ | name = Module | | | | | | | |   | | | | | | | | | | | | | | | | | | v v v | | |  -> = Dependency | | +            -+ | shared | | +             + |    |<    -+ | | | itool | | geom   - | |   | | CANNOT IMPORT FROM HIGHER IN HEIRARCHY | v | | misc polygon | |      - | | | | | fourier v | |    - json2dict | |     - | +            -+   -  qdmpy.field - Field sub-package. Contains functions to convert bnvs/resonances to fields (e.g. magnetic, electric,  .) through hamiltonian fits/otherwise. -  qdmpy.magsim - Tooling that simulates magnetic field produced by magnetised flakes (static only). -  qdmpy.pl - Sub-package for dealing with pl data. Contains procedures for fitting raw photoliminescence, outputting results etc. -  qdmpy.plot - This sub-package contains all of the plotting functions (matplotlib based). -  qdmpy.source - Contains tools for reconstructing source fields (e.g. current densities or intrinsic magnetization) from the measured magnetic field calculated in qdmpy.field. -  qdmpy.shared - Contains procedures shared between the other higher level modules. Cannot import from the other modules or you'll get circular import errors. Specific tooling here includes those to help with fourier transforms, NV geometry, image tooling such as filtering and background subtraction, as well as json io and polygon selection. -  qdmpy.system - This sub-package contains the tooling for defining institution specific settings for example for loading raw datafiles etc. These settings can be implemented down to the specific experimental 'system' to define pixel sizes etc.  qdmpy itself also exposes some functions from qdmpy.interface"
},
{
"ref":"qdmpy.pl",
"url":1,
"doc":"Sub-package for fitting widefield NV microscopy (photoluminescence) data. This (sub-) package exposes all of the concents of  qdmpy.pl.interface,  qdmpy.pl.model (FitModel class etc.) and  qdmpy.pl.io "
},
{
"ref":"qdmpy.pl.io",
"url":2,
"doc":"This module holds tools for calculating the bnv from ODMR datasets (after they've been fit with the  qdmpy.pl.interface tooling). Functions     - -  qdmpy.field.bnv.get_bnvs_and_dshifts -  qdmpy.field.bnv.get_bnv_sd -  qdmpy.field.bnv.check_exp_bnv_compatibility -  qdmpy.field.bnv.bnv_refsub -  qdmpy.field.bnv.sub_bground_bnvs "
},
{
"ref":"qdmpy.pl.io.load_image_and_sweep",
"url":2,
"doc":"Reads raw image data and sweep_list (affine parameters) using system methods Arguments     - options : dict Generic options dict holding all the user options. Returns    - image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. sweep_list : list List of sweep parameter values",
"func":1
},
{
"ref":"qdmpy.pl.io.reshape_dataset",
"url":2,
"doc":"Reshapes and re-bins raw data into more useful format. Cuts down to ROI and removes unwanted sweeps. Arguments     - options : dict Generic options dict holding all the user options. image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. sweep_list : list List of sweep parameter values Returns    - pl_image : np array, 2D. Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and cut down to ROI. pl_image_ROI : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and cut down to ROI. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] ref : np array, 3D Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] sig_norm : np array, 3D Signal normalised by reference (via subtraction or normalisation, chosen in options), reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] single_pixel_pl : np array, 1D Normalised measurement array, for chosen single pixel check. sweep_list : list List of sweep parameter values (with removed unwanted sweeps at start/end) roi : length 2 list of np meshgrids Defines an ROI that can be applied to the 3D image through direct indexing. E.g. sig_ROI = sig[:, roi[0], roi[1 (post rebinning)",
"func":1
},
{
"ref":"qdmpy.pl.io.save_pl_data",
"url":2,
"doc":"pixel_fit_params -> bnvs, dshifts (both lists of np arrays, 2D) Arguments     - fit_result_dict : OrderedDict Dictionary, key: param_keys, val: image (2D) of param values across FOV. Ordered by the order of functions in options[\"fit_functions\"]. If None, returns ([], []) bias_field_spherical_deg : tuple Bias field in spherical polar degrees (and gauss).  Returns    - bnvs : list List of np arrays (2D) giving B_NV for each NV family/orientation. If num_peaks is odd, the bnv is given as the shift of that peak, and the dshifts is left as np.nans. dshifts : list List of np arrays (2D) giving the D (~DFS) of each NV family/orientation. If num_peaks is odd, the bnv is given as the shift of that peak, and the dshifts is left as np.nans.",
"func":1
},
{
"ref":"qdmpy.pl.io._rebin_image",
"url":2,
"doc":"Reshapes raw data into more useful shape, according to image size in metadata. Arguments     - options : dict Generic options dict holding all the user options. image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Returns    - image_rebinned : np array, 3D Format: [sweep values, y, x]. Same as image, but now rebinned (x size and y size have changed). Not cut down to ROI. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet. Format: [sweep_vals, y, x]. Not cut down to ROI. ref : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet. Not cut down to ROI. Format: [sweep_vals, y, x]. sig_norm : np array, 3D Signal normalised by reference (via subtraction or normalisation, chosen in options), reshaped and rebinned. Unwanted sweeps not removed yet. Not cut down to ROI. Format: [sweep_vals, y, x].",
"func":1
},
{
"ref":"qdmpy.pl.io._remove_unwanted_data",
"url":2,
"doc":"Removes unwanted sweep values (i.e. freq values or tau values) for all of the data arrays. Also cuts data down to ROI. Arguments     - options : dict Generic options dict holding all the user options. image_rebinned : np array, 3D Format: [sweep values, y, x]. Same as image, but rebinned (x size and y size have changed). sweep_list : list List of sweep parameter values sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet. Format: [sweep_vals, y, x] ref : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet. Format: [sweep_vals, y, x] sig_norm : np array, 3D Signal normalised by reference (via subtraction or normalisation, chosen in options), reshaped and rebinned. Unwanted sweeps not removed yet. Format: [sweep_vals, y, x] Returns    - pl_image : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and cut down to ROI pl_image_ROI : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped and rebinned as well as cut down to ROI. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed. Format: [sweep_vals, y, x] ref : np array, 3D Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed. Format: [sweep_vals, y, x] sig_norm : np array, 3D Signal normalised by reference (via subtraction or normalisation, chosen in options), reshaped and rebinned. Unwanted sweeps removed. Format: [sweep_vals, y, x] sweep_list : list List of sweep parameter values (with removed unwanted sweeps at start/end) roi : length 2 list of np meshgrids Defines an ROI that can be applied to the 3D image through direct indexing. E.g. sig_ROI = sig[:, roi[0], roi[1 ",
"func":1
},
{
"ref":"qdmpy.pl.io._check_start_end_rectangle",
"url":2,
"doc":"Checks that 'name' rectange (defined by top left corner 'start_x', 'start_y' and bottom right corner 'end_x', 'end_y') fits within a larger rectangle of size 'full_size_w', 'full_size_h'. If there are no good, they're fixed with a warning. Arguments     - name : str The name of the rectangle we're checking (e.g. \"ROI\", \"AOI\"). start_x : int x coordinate (relative to origin) of rectangle's top left corner. start_y : int y coordinate (relative to origin) of rectangle's top left corner. end_x : int x coordinate (relative to origin) of rectangle's bottom right corner. end_y : int y coordinate (relative to origin) of rectangle's bottom right corner. full_size_w : int Full width of image (or image region, e.g. ROI). full_size_h : int Full height of image (or image region, e.g. ROI). Returns    - start_coords : tuple 'fixed' start coords: (start_x, start_y) end_coords : tuple 'fixed' end coords: (end_x, end_y)",
"func":1
},
{
"ref":"qdmpy.pl.io.load_prev_pl_fit_results",
"url":2,
"doc":"Load (all) parameter pl fits from previous processing.",
"func":1
},
{
"ref":"qdmpy.pl.io.load_prev_pl_fit_sigmas",
"url":2,
"doc":"Load (all) parameter pl fits from previous processing.",
"func":1
},
{
"ref":"qdmpy.pl.io.load_pl_fit_sigma",
"url":2,
"doc":"Load a previous fit sigma result, of name 'param_key'",
"func":1
},
{
"ref":"qdmpy.pl.io.load_fit_param",
"url":2,
"doc":"Load a previously fit param, of name 'param_key'.",
"func":1
},
{
"ref":"qdmpy.pl.io.save_pl_fit_sigmas",
"url":2,
"doc":"Saves pixel fit sigmas to disk. Arguments     - options : dict Generic options dict holding all the user options. sigmas : OrderedDict Dictionary, key: param_keys, val: image (2D) of param sigmas across FOV.",
"func":1
},
{
"ref":"qdmpy.pl.io.save_pl_fit_results",
"url":2,
"doc":"Saves pl fit results to disk. Arguments     - options : dict Generic options dict holding all the user options. pixel_fit_params : OrderedDict Dictionary, key: param_keys, val: image (2D) of param values across FOV.",
"func":1
},
{
"ref":"qdmpy.pl.io.load_ref_exp_pl_fit_results",
"url":2,
"doc":"ref_options dict -> pixel_fit_params dict. Provide one of ref_options and ref_options_dir. If both are None, returns None (with a warning). If both are supplied, ref_options takes precedence. Arguments     - ref_options : dict, default=None Generic options dict holding all the user options (for the reference experiment). Returns    - fit_result_dict : OrderedDict Dictionary, key: param_keys, val: image (2D) of (fit) param values across FOV. If no reference experiment is given (i.e. ref_options and ref_options_dir are None) then returns None fit_result_sigmas : OrderedDict Dict as above, but fit sigmas",
"func":1
},
{
"ref":"qdmpy.pl.io.check_if_already_fit",
"url":2,
"doc":"Looks for previous fit result. If previous fit result exists, checks for compatibility between option choices. loading_ref (bool): skip checks for force_fit etc. and just see if prev pixel results exist. Returns nothing.",
"func":1
},
{
"ref":"qdmpy.pl.io.prev_options_exist",
"url":2,
"doc":"Checks if options file from previous result can be found in default location, returns Bool.",
"func":1
},
{
"ref":"qdmpy.pl.io.options_compatible",
"url":2,
"doc":"Checks if option choices are compatible with previously fit options Arguments     - options : dict Generic options dict holding all the user options. prev_options : dict Generic options dict from previous fit result. Returns    - options_compatible : bool Whether or not options are compatible. reason : str Reason for the above",
"func":1
},
{
"ref":"qdmpy.pl.io._prev_pl_fits_exist",
"url":2,
"doc":"Check if the actual fit result files exists. Arguments     - options : dict Generic options dict from (either prev. or current, should be the equiv.) fit result. Returns    - pixels_results_exist : bool Whether or not previous pixel result files exist. reason : str Reason for the above",
"func":1
},
{
"ref":"qdmpy.pl.io._prev_pl_sigmas_exist",
"url":2,
"doc":"as  qdmpy.pl.io._prev_pl_fits_exist but for sigmas",
"func":1
},
{
"ref":"qdmpy.pl.io.get_prev_options",
"url":2,
"doc":"Reads options file from previous fit result (.json), returns a dictionary.",
"func":1
},
{
"ref":"qdmpy.pl.interface",
"url":3,
"doc":"This module holds the general interface tools for fitting data, independent of fit backend (e.g. scipy/gpufit etc.). All of these functions are automatically loaded into the namespace when the fit sub-package is imported. (e.g. import qdmpy.fit). Functions     - -  qdmpy.pl.interface.define_fit_model -  qdmpy.pl.interface.fit_roi_avg_pl -  qdmpy.pl.interface.fit_aois_pl -  qdmpy.pl.interface.fit_all_pixels_pl -  qdmpy.pl.interface._prep_fit_backends -  qdmpy.pl.interface.get_pl_fit_result "
},
{
"ref":"qdmpy.pl.interface.define_fit_model",
"url":3,
"doc":"Define (and return) fit_model object, from options dictionary. Arguments     - options : dict Generic options dict holding all the user options. Returns    - fit_model :  qdmpy.pl.model.FitModel FitModel used to fit to data.",
"func":1
},
{
"ref":"qdmpy.pl.interface._prep_fit_backends",
"url":3,
"doc":"Prepare all possible fit backends, checking that everything will work. Also attempts to import relevant modules into global scope. This is a wrapper around specific functions for each backend. All possible fit backends are loaded - these are decided in the config file for this system, i.e. system.option_choices(\"fit_backend\") Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel FitModel used to fit to data.",
"func":1
},
{
"ref":"qdmpy.pl.interface.fit_roi_avg_pl",
"url":3,
"doc":"Fit the average of the measurement over the region of interest specified, with backend chosen via options[\"fit_backend_comparison\"]. Arguments     - options : dict Generic options dict holding all the user options. sig : np array, 3D Sig measurement array, unnormalised, shape: [sweep_list, y, x]. ref : np array, 3D Ref measurement array, unnormalised, shape: [sweep_list, y, x]. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. Returns    - backend_roi_results_lst : list List of  qdmpy.pl.common.ROIAvgFitResult objects containing the fit result (see class specifics) for each fit backend selected for comparison.",
"func":1
},
{
"ref":"qdmpy.pl.interface.fit_aois_pl",
"url":3,
"doc":"Bxyz measured -> Bxyz_recon via fourier methods. Arguments     - bfield : list List of magnetic field components, e.g [bx_image, by_image, bz_image] pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. nvs_above_sample : bool True if NV layer is above (higher in z) than sample being imaged. Returns    - bx_recon, by_recon, bz_recon : np arrays (2D) The reconstructed bfield maps. For a proper explanation of methodology, see [CURR_RECON]_.  \\nabla \\times {\\bf B} = 0  to get Bx_recon and By_recon from Bz (in a source-free region), and  \\nabla \\cdot {\\bf B} = 0  to get Bz_recon from Bx and By Start with e.g.:  \\frac{\\partial B_x^{\\rm recon {\\partial z} = \\frac{\\partial B_z}{\\partial x}  with the definitions  \\hat{B}:= \\hat{\\mathcal{F _{xy} \\big\\{ B \\big\\}  and  k = \\sqrt{k_x^2 + k_y^2}  we have:  \\frac{\\partial }{\\partial z} \\hat{B}_x^{\\rm recon}(x,y,z=z_{\\rm NV}) = i k_x \\hat{B}_z(x,y, z=z_{\\rm NV})  . Now using upward continuation [CURR_RECON]_ to evaluate the z partial:  -k \\hat{B}_x^{\\rm recon}(x,y,z=z_{\\rm NV}) = i k_x \\hat{B}_z(k_x, k_y, z_{\\rm NV})  such that for  k \\neq 0  we have  (\\hat{B}_x^{\\rm recon}(x,y,z=z_{\\rm NV}), \\hat{B}_y^{\\rm recon}(x,y,z=z_{\\rm NV} = \\frac{-i}{k} (k_x, k_y) \\hat{B}_z(x,y z=z_{\\rm NV})  Utilising the zero-divergence property of the magnetic field, it can also be shown:  \\hat{B}_z^{\\rm recon}(x,y,z=z_{\\rm NV}) = \\frac{i}{k} \\left( k_x \\hat{B}_x(x,y,z=z_{\\rm NV}) + k_y \\hat{B}_y(x,y,z=z_{\\rm NV}) \\right)  References       [CURR_RECON] E. A. Lima and B. P. Weiss, Obtaining Vector Magnetic Field Maps from Single-Component Measurements of Geological Samples, Journal of Geophysical Research: Solid Earth 114, (2009). https: doi.org/10.1029/2008JB006006",
"func":1
},
{
"ref":"qdmpy.field.hamiltonian",
"url":4,
"doc":"This module is for . Functions     - -  qdmpy.field.hamiltonian.define_hamiltonian -  qdmpy.field.hamiltonian._prep_fit_backends -  qdmpy.field.hamiltonian.fit_hamiltonian_pixels -  qdmpy.field.hamiltonian.ham_gen_init_guesses -  qdmpy.field.hamiltonian.ham_bounds_from_range -  qdmpy.field.hamiltonian.ham_pixel_generator -  qdmpy.field.hamiltonian.ham_shuffle_pixels -  qdmpy.field.hamiltonian.ham_unshuffle_pixels -  qdmpy.field.hamiltonian.ham_unshuffle_fit_results -  qdmpy.field.hamiltonian.ham_get_pixel_fitting_results Classes    - -  qdmpy.field.hamiltonian.Chooser -  qdmpy.field.hamiltonian.Hamiltonian -  qdmpy.field.hamiltonian.ApproxBxyz -  qdmpy.field.hamiltonian.Bxyz "
},
{
"ref":"qdmpy.field.hamiltonian.S_MAT_X",
"url":4,
"doc":"Spin-1 operator: S{\\rm X}"
},
{
"ref":"qdmpy.pl.model",
"url":4,
"doc":"Spin-1 operator: S{\\rm Y}"
},
{
"ref":"qdmpy.pl.model.FitModel",
"url":4,
"doc":"Spin-1 operator: S{\\rm Z}"
},
{
"ref":"qdmpy.pl.model.FitModel.residuals_scipyfit",
"url":4,
"doc":"Evaluates residual: fit model (affine params/sweep_vec) - pl values",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.jacobian_scipyfit",
"url":4,
"doc":"Evaluates (analytic) jacobian of fitmodel in format expected by scipy least_squares",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.jacobian_defined",
"url":4,
"doc":"Check if analytic jacobian is defined for this fit model.",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.get_param_defn",
"url":4,
"doc":"Returns list of parameters in fit_model, note there will be duplicates, and they do not have numbers e.g. 'pos_0'. Use  qdmpy.pl.model.FitModel.get_param_odict for that purpose. Returns    - param_defn_ar : list List of parameter names (param_defn) in fit model.",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.get_param_odict",
"url":4,
"doc":"get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model Returns    - param_dict : dict Dictionary containing key: params, values: units.",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.get_param_unit",
"url":4,
"doc":"Get unit for a given param_key (given by param_name + \"_\" + param_number) Arguments     - param_name : str Name of parameter, e.g. 'pos' param_number : float or int Which parameter to use, e.g. 0 for 'pos_0' Returns    - unit : str Unit for that parameter, e.g. \"constant\" -> \"Amplitude (a.u.) ",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit",
"url":5,
"doc":"This module holds tools for fitting raw data via scipy. (scipy backend) Functions     - -  qdmpy.pl.scipyfit.prep_scipyfit_options -  qdmpy.pl.scipyfit.gen_scipyfit_init_guesses -  qdmpy.pl.scipyfit.fit_roi_avg_pl_scipyfit -  qdmpy.pl.scipyfit.fit_single_pixel_pl_scipyfit -  qdmpy.pl.scipyfit.fit_aois_pl_scipyfit -  qdmpy.pl.scipyfit.limit_cpu -  qdmpy.pl.scipyfit.to_squares_wrapper -  qdmpy.pl.scipyfit.fit_all_pixels_pl_scipyfit "
},
{
"ref":"qdmpy.pl.scipyfit.prep_scipyfit_options",
"url":5,
"doc":"General options dict -> scipyfit_options in format that scipy least_squares expects. Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel Fit model object. Returns    - scipy_fit_options : dict Dictionary with options that scipy.optimize.least_squares expects, specific to fitting.",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.gen_scipyfit_init_guesses",
"url":5,
"doc":"Generate arrays of initial fit guesses and bounds in correct form for scipy least_squares. init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays, that scipy will recognise. In particular, we specificy that each of the 'num' of each 'fn_type' have independent parameters, so must have independent init_guesses and init_bounds when plugged into scipy. Arguments     - options : dict Generic options dict holding all the user options. init_guesses : dict Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent version of that fn_type. init_bounds : dict Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent version of that fn_type. Returns    - fit_param_ar : np array, shape: num_params The initial fit parameter guesses. fit_param_bound_ar : np array, shape: (num_params, 2) Fit parameter bounds.",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.fit_roi_avg_pl_scipyfit",
"url":5,
"doc":"Fit the average of the measurement over the region of interest specified, with scipy. Arguments     - options : dict Generic options dict holding all the user options. sig : np array, 3D Sig measurement array, unnormalised, shape: [sweep_list, y, x]. ref : np array, 3D Ref measurement array, unnormalised, shape: [sweep_list, y, x]. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel The fit model object. Returns    -  qdmpy.pl.common.ROIAvgFitResult object containing the fit result (see class specifics)",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.fit_single_pixel_pl_scipyfit",
"url":5,
"doc":"Fit Single pixel and return best_fit_result.x (i.e. the optimal fit parameters) Arguments     - options : dict Generic options dict holding all the user options. pixel_pl_ar : np array, 1D Normalised pl as function of sweep_list for a single pixel. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel The fit model. roi_avg_fit_result :  qdmpy.pl.common.ROIAvgFitResult  qdmpy.pl.common.ROIAvgFitResult object, to pull fit_options from. Returns    - pixel_parameters : np array, 1D Best fit parameters, as determined by scipy.",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.fit_aois_pl_scipyfit",
"url":5,
"doc":"Fit AOIs and single pixel and return list of (list of) fit results (optimal fit parameters), using scipy. Arguments     - options : dict Generic options dict holding all the user options. sig : np array, 3D Sig measurement array, unnormalised, shape: [sweep_list, y, x]. ref : np array, 3D Ref measurement array, unnormalised, shape: [sweep_list, y, x]. single_pixel_pl : np array, 1D Normalised measurement array, for chosen single pixel check. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq). fit_model :  qdmpy.pl.model.FitModel The model we're fitting to. aois : list List of AOI specifications - each a length-2 iterable that can be used to directly index into sig_norm to return that AOI region, e.g. sig_norm[:, AOI[0], AOI[1 . roi_avg_fit_result :  qdmpy.pl.common.ROIAvgFitResult  qdmpy.pl.common.ROIAvgFitResult object, to pull  qdmpy.pl.common.ROIAvgFitResult.fit_options from. Returns    - fit_result_collection :  qdmpy.pl.common.FitResultCollection Collection of ROI/AOI fit results for this fit backend.",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.limit_cpu",
"url":5,
"doc":"Called at every process start, to reduce the priority of this process",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.to_squares_wrapper",
"url":5,
"doc":"Simple wrapper of scipy.optimize.least_squares to allow us to keep track of which solution is which (or where). Arguments     - fun : function Function object acting as residual (fit model minus pl value) p0 : np array Initial guess: array of parameters sweep_vec : np array Array (or I guess single value, anything iterable) of affine parameter (tau/freq) shaped_data : list (3 elements) array returned by  qdmpy.pl.common.pixel_generator : [y, x, sig_norm[:, y, x fit_optns : dict Other options (dict) passed to least_squares Returns    - wrapped_squares : tuple (y, x), least_squares( .).x, leas_squares( .).jac I.e. the position of the fit result, the fit result parameters array, jacobian at solution",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.fit_all_pixels_pl_scipyfit",
"url":5,
"doc":"Fits each pixel and returns dictionary of param_name -> param_image. Arguments     - options : dict Generic options dict holding all the user options. sig_norm : np array, 3D Normalised measurement array, shape: [sweep_list, y, x]. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel The model we're fitting to. roi_avg_fit_result :  qdmpy.pl.common.ROIAvgFitResult  qdmpy.pl.common.ROIAvgFitResult object, to pull fit_options from. Returns    - fit_image_results : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. Also has 'residual' as a key.",
"func":1
},
{
"ref":"qdmpy.pl.funcs",
"url":6,
"doc":"This module defines the fit functions used in the  qdmpy.pl.model.FitModel . We grab/use this regardless of fitting on cpu (scipy) or gpu etc. Ensure any fit functions you define are added to the AVAILABLE_FNS module variable at bottom. Try not to have overlapping parameter names in the same fit. For ODMR peaks, ensure the frequency position of the peak is named something prefixed by 'pos'. (see  qdmpy.field.bnv.get_bnvs_and_dshifts for the reasoning). Classes    - -  qdmpy.pl.funcs.FitFunc -  qdmpy.pl.funcs.Constant -  qdmpy.pl.funcs.Linear -  qdmpy.pl.funcs.Circular -  qdmpy.pl.funcs.Gaussian -  qdmpy.pl.funcs.GaussianHyperfine14 -  qdmpy.pl.funcs.GaussianHyperfine15 -  qdmpy.pl.funcs.Lorentzian -  qdmpy.pl.funcs.LorentzianHyperfine14 -  qdmpy.pl.funcs.LorentzianHyperfine15 -  qdmpy.pl.funcs.StretchedExponential -  qdmpy.pl.funcs.DampedRabi -  qdmpy.pl.funcs.WalshT1 "
},
{
"ref":"qdmpy.pl.funcs.FitFunc",
"url":6,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.FitFunc.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.FitFunc.grad_fn",
"url":6,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Constant",
"url":6,
"doc":"Constant Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Constant.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Constant.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Constant.eval",
"url":6,
"doc":"speed tested multiple methods, this was the fastest",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Constant.grad_fn",
"url":6,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 1)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Linear",
"url":6,
"doc":"Linear function, y=mx+c Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Linear.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Linear.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Linear.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Linear.grad_fn",
"url":6,
"doc":"Compute the grad of the residual, excluding pl as a param {output shape: (len(x), 2)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Circular",
"url":6,
"doc":"Circular function (sine) Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Circular.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Circular.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Circular.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Circular.grad_fn",
"url":6,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 3)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Gaussian",
"url":6,
"doc":"Gaussian function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Gaussian.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Gaussian.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Gaussian.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Gaussian.grad_fn",
"url":6,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14",
"url":6,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14.grad_fn",
"url":6,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15",
"url":6,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15.grad_fn",
"url":6,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Lorentzian",
"url":6,
"doc":"Lorentzian function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Lorentzian.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Lorentzian.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Lorentzian.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Lorentzian.grad_fn",
"url":6,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 3)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14",
"url":6,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14.grad_fn",
"url":6,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15",
"url":6,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15.grad_fn",
"url":6,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianhBN",
"url":6,
"doc":"Bxyz measured -> Bxyz_recon via fourier methods. Adds Bx_recon etc. to field_params. Arguments     - options : dict Generic options dict holding all the user options (for the main/signal experiment). field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. Returns    - nothing (operates in place on field_params) For a proper explanation of methodology, see [CURR_RECON]_.  \\nabla \\times {\\bf B} = 0  to get Bx_recon and By_recon from Bz (in a source-free region), and  \\nabla \\cdot {\\bf B} = 0  to get Bz_recon from Bx and By Start with e.g.:  \\frac{\\partial B_x^{\\rm recon {\\partial z} = \\frac{\\partial B_z}{\\partial x}  with the definitions  \\hat{B}:= \\hat{\\mathcal{F _{xy} \\big\\{ B \\big\\}  and  k = \\sqrt{k_x^2 + k_y^2}  we have:  \\frac{\\partial }{\\partial z} \\hat{B}_x^{\\rm recon}(x,y,z=z_{\\rm NV}) = i k_x \\hat{B}_z(x,y, z=z_{\\rm NV})  . Now using upward continuation [CURR_RECON]_ to evaluate the z partial:  -k \\hat{B}_x^{\\rm recon}(x,y,z=z_{\\rm NV}) = i k_x \\hat{B}_z(k_x, k_y, z_{\\rm NV})  such that for  k \\neq 0  we have (analogously for y)  (\\hat{B}_x^{\\rm recon}(x,y,z=z_{\\rm NV}), \\hat{B}_y^{\\rm recon}(x,y,z=z_{\\rm NV} = \\frac{-i}{k} (k_x, k_y) \\hat{B}_z(x,y z=z_{\\rm NV})  Utilising the zero-divergence property of the magnetic field, it can also be shown:  \\hat{B}_z^{\\rm recon}(x,y,z=z_{\\rm NV}) = \\frac{i}{k} \\left( k_x \\hat{B}_x(x,y,z=z_{\\rm NV}) + k_y \\hat{B}_y(x,y,z=z_{\\rm NV}) \\right)  References       [CURR_RECON] E. A. Lima and B. P. Weiss, Obtaining Vector Magnetic Field Maps from Single-Component Measurements of Geological Samples, Journal of Geophysical Research: Solid Earth 114, (2009). https: doi.org/10.1029/2008JB006006",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianhBN.grad_fn",
"url":6,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential",
"url":6,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential.grad_fn",
"url":6,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 3)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.DampedRabi",
"url":6,
"doc":"Damped oscillation Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.DampedRabi.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.DampedRabi.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.DampedRabi.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.DampedRabi.grad_fn",
"url":6,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 4)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.WalshT1",
"url":6,
"doc":"Walsh/Hall T1 model Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.WalshT1.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.WalshT1.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.WalshT1.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.WalshT1.grad_fn",
"url":6,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 3)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.HallT1",
"url":6,
"doc":"Walsh/Hall T1 model Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.HallT1.param_defn",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.HallT1.param_units",
"url":6,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.HallT1.eval",
"url":6,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.HallT1.grad_fn",
"url":6,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 3)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.AVAILABLE_FNS",
"url":6,
"doc":"Dictionary that defines fit functions available for use. Add any functions you define here so you can use them. Aviod overlapping function parameter names."
},
{
"ref":"qdmpy.pl.common",
"url":7,
"doc":"This module holds some functions and classes that are shared between different fitting backends, but are not a part of the user-facing interface. Classes    - -  qdmpy.pl.common.FitResultCollection -  qdmpy.pl.common.ROIAvgFitResult Functions     - -  qdmpy.pl.common.shuffle_pixels -  qdmpy.pl.common.unshuffle_pixels -  qdmpy.pl.common.unshuffle_fit_results -  qdmpy.pl.common.pixel_generator -  qdmpy.pl.common.gen_init_guesses -  qdmpy.pl.common.bounds_from_range -  qdmpy.pl.common.get_pixel_fitting_results "
},
{
"ref":"qdmpy.pl.common.FitResultCollection",
"url":7,
"doc":"Object to hold AOI average fit results, and a place to define their names. Arguments     - fit_backend : str Name of the fit backend (e.g. scipy, gpufit, etc.) used. roi_avg_fit_result  qdmpy.pl.common.ROIAvgFitResult object. single_pixel_result Best (optimal) fit/model parameters for single pixel check. aoi_fit_results_lst : list of lists List of (list of) best (optimal) parameters, for each AOI region (avg)."
},
{
"ref":"qdmpy.pl.common.ROIAvgFitResult",
"url":7,
"doc":"Object to hold ROI average fit result, and a place to define result names. Arguments     - fit_backend : string Name of the fit backend (e.g. scipy, gpufit, etc.) fit_options : dict Options dictionary for this fit method, as will be passed to fitting function. E.g. scipy least_squares is handed various options as a dictionary. fit_model :  qdmpy.pl.model.FitModel Fit model used. Can construct fit via res.fit_model(res.best_params, res.sweep_list). pl_roi : np array, 1D pl data summed over FOV, as fn of sweep_vec. sweep_list : np array, 1D Affine parameter i.e. tau or frequency. best_params : np array, 1D Solution fit parameters array."
},
{
"ref":"qdmpy.pl.common.ROIAvgFitResult.savejson",
"url":7,
"doc":"Save all attributes as a json file in dir/filename, via  qdmpy.shared.json2dict.dict_to_json ",
"func":1
},
{
"ref":"qdmpy.pl.common.ROIAvgFitResult.fit_backend",
"url":7,
"doc":"fit_backend : str Name of the fit method (e.g. scipy, gpufit, etc.)"
},
{
"ref":"qdmpy.pl.common.ROIAvgFitResult.fit_options",
"url":7,
"doc":"fit_options : dict Options dictionary for this fit method, as will be passed to fitting function. E.g. scipy least_squares is handed various options as a dictionary."
},
{
"ref":"qdmpy.pl.common.shuffle_pixels",
"url":7,
"doc":"Simple shuffler Arguments     - data_3d : np array, 3D i.e. sig_norm data, [affine param, y, x]. Returns    - shuffled_in_yx : np array, 3D data_3d shuffled in 2nd, 3rd axis. unshuffler : (y_unshuf, x_unshuf) Both np array. Can be used to unshuffle shuffled_in_yx, i.e. through  qdmpy.pl.common.unshuffle_pixels .",
"func":1
},
{
"ref":"qdmpy.pl.common.unshuffle_pixels",
"url":7,
"doc":"Simple shuffler Arguments     - data_2d : np array, 2D i.e. 'image' of a single fit parameter, all shuffled up! unshuffler : (y_unshuf, x_unshuf) Two arrays returned by  qdmpy.pl.common.shuffle_pixels that allow unshuffling of data_2d. Returns    - unshuffled_in_yx: np array, 2D data_2d but the inverse operation of  qdmpy.pl.common.shuffle_pixels has been applied",
"func":1
},
{
"ref":"qdmpy.pl.common.unshuffle_fit_results",
"url":7,
"doc":"Simple shuffler Arguments     - fit_result_dict : dict Dictionary, key: param_names, val: image (2D) of param values across FOV. Each image requires reshuffling (which this function achieves). Also has 'residual' as a key. unshuffler : (y_unshuf, x_unshuf) Two arrays returned by  qdmpy.pl.common.shuffle_pixels that allow unshuffling of data_2d. Returns    - fit_result_dict : dict Same as input, but each fit parameter has been unshuffled.",
"func":1
},
{
"ref":"qdmpy.pl.common.pixel_generator",
"url":7,
"doc":"Simple generator to shape data as expected by to_squares_wrapper in scipy concurrent method. Also allows us to track  where (i.e. which pixel location) each result corresponds to. See also:  qdmpy.pl.scipyfit.to_squares_wrapper , and corresponding gpufit method. Arguments     - our_array : np array, 3D Shape: [sweep_list, y, x] Returns    - generator : list [y, x, our_array[:, y, x generator (yielded)",
"func":1
},
{
"ref":"qdmpy.pl.common.gen_init_guesses",
"url":7,
"doc":"Generate initial guesses (and bounds) in fit parameters from options dictionary. Both are returned as dictionaries, you need to use 'gen_{scipy/gpufit/ .}_init_guesses' to convert to the correct (array) format for each specific fitting backend. Arguments     - options : dict Generic options dict holding all the user options. Returns    - init_guesses : dict Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent version of that fn_type. init_bounds : dict Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent version of that fn_type.",
"func":1
},
{
"ref":"qdmpy.interface",
"url":8,
"doc":"This module defines some ease-of-use methods for the qdmpy package. Functions     - -  qdmpy.interface.initialize -  qdmpy.interface.load_options -  qdmpy.interface.save_options -  qdmpy.interface._add_bias_field -  qdmpy.interface._get_bias_field -  qdmpy.interface._spherical_deg_to_cart -  qdmpy.interface._define_output_dir -  qdmpy.interface._interpolate_option_str -  qdmpy.interface.load_polygons -  qdmpy.interface.check_option -  qdmpy.interface.check_options -  qdmpy.interface.clean_options Classes    - -  qdmpy.interface.OptionsError "
},
{
"ref":"qdmpy.interface.initialize",
"url":8,
"doc":"Helped function to initialise analysis program. Arguments     - options_dict : dict, default=None Generic options dict holding all the user options (for the main/signal experiment). options_path : str or path object, default=None Direct path to options json, i.e. will run something like 'read(options_path)'. ref_options_dict : dict, default=None Generic options dict holding all the user options (for the reference experiment). ref_options_dir : str or path object, default=None Path to read reference options from, i.e. will run something like 'read('ref_options_dir / saved_options.json')'. Returns    - options_dict : dict (Processed) generic options dict holding all user options. ref_options_dict : dict As options_dict, but for reference experiment (assuming pl already fit).",
"func":1
},
{
"ref":"qdmpy.pl.common.get_pixel_fitting_results",
"url":7,
"doc":"Take the fit result data from scipyfit/gpufit and back it down to a dictionary of arrays. Each array is 2D, representing the values for each parameter (specified by the dict key). Arguments     - fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. fit_results : list of [(y, x), result, jac] objects (see  qdmpy.pl.scipyfit.to_squares_wrapper , or corresponding gpufit method) A list of each pixel's parameter array, as well as position in image denoted by (y, x). pixel_data : np array, 3D Normalised measurement array, shape: [sweep_list, y, x]. i.e. sig_norm. May or may not already be shuffled (i.e. matches fit_results). sweep_list : np array, 1D Affine parameter list (e.g. tau or freq). Returns    - fit_image_results : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. Also has 'residual' as a key. sigmas : dict Dictionary, key: param_keys, val: image (2D) of param uncertainties across FOV.",
"func":1
},
{
"ref":"qdmpy.plot",
"url":8,
"doc":"Sub-package for plotting results of qdmpy. This (sub-) package exposes all of the public members of the following modules: -  qdmpy.plot.common -  qdmpy.plot.field -  qdmpy.plot.pl -  qdmpy.plot.source "
},
{
"ref":"qdmpy.plot.field",
"url":9,
"doc":"This module holds functions for plotting fields. Functions     - -  qdmpy.plot.field.bnvs_and_dshifts -  qdmpy.plot.field.bfield -  qdmpy.plot.field.dshift_fit -  qdmpy.plot.field.field_residual -  qdmpy.plot.field.field_param -  qdmpy.plot.field.field_param_flattened -  qdmpy.plot.field.bfield_consistency -  qdmpy.plot.field.bfield_theta_phi "
},
{
"ref":"qdmpy.plot.field.bnvs_and_dshifts",
"url":9,
"doc":"Plots bnv arrays above dshift arrays. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames bnvs : list List of np arrays (2D) giving B_NV for each NV family/orientation. If num_peaks is odd, the bnv is given as the shift of that peak, and the dshifts is left as np.nans. dshifts : list List of np arrays (2D) giving the D (~DFS) of each NV family/orientation. If num_peaks is odd, the bnv is given as the shift of that peak, and the dshifts is left as np.nans. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.bfield",
"url":9,
"doc":"Plot Bxyz if available as keys in 'field_params'. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.interface._spherical_deg_to_cart",
"url":8,
"doc":"Field vector in spherical polar degrees -> cartesian (gauss) Parameters      b_ag_gauss, b_theta_deg, b_phi_deg : float Field components in spherical polar (degrees) Returns    - b_x, b_y, b_z : tuple All floats in gauss, cartesian field components.",
"func":1
},
{
"ref":"qdmpy.plot.field.field_residual",
"url":9,
"doc":"Plot residual if available as keys in 'field_params' (as 'residual_field'). Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.field_param",
"url":9,
"doc":"plot a given field param. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames. param_name : str Name of specific param to plot (e.g. \"Bx\" etc.). field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. c_map : str, default: None colormap to use overrides c_map_type and c_range_type. c_map_type : str, default: \"param_images\" colormap type to search options (options[\"colormaps\"][c_map_type]) for. c_map_range : str, default: \"percentile\" colormap range option (see  qdmpy.plot.common.get_colormap_range ) to use. c_range_vals : number or list, default: (5, 95) passed with c_map_range to get_colormap_range cbar_label : str, default: label to chuck on ye olde colorbar (z-axis label). Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.field_param_flattened",
"url":9,
"doc":"plot a field param flattened vs pixel number. Sigmas are optionally utilised as errobars. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames. param_name : str Name of key in field_params (and sigmas) to plot. field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. sigmas : dict, default: None Dictionary, key: param_keys, val: image (2D) of (field) sigma values across FOV. plot_sigmas : bool, default: True If true & sigmas is not None, sigmas are used as errorbars. plot_bounds : bool, default: True If True, (field) fit bound is annotated as horizontal dashed lines. plot_guess: bool, default: True If True, (field) fit guess is annotates as horizontal dashed line. y_label : str, default:  Label to chuck on y axis. errorevery : int, default: 1 Plot errorbar every 'errorevery' data point (so it doesn't get too messy). Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.strain",
"url":9,
"doc":"",
"func":1
},
{
"ref":"qdmpy.plot.field.efield",
"url":9,
"doc":"",
"func":1
},
{
"ref":"qdmpy.plot.field.bfield_consistency",
"url":9,
"doc":"plot bfield vs bfield_recon Parameters      options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames. field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.bfield_theta_phi",
"url":9,
"doc":"Plots B_theta_phi if found in field_params (the vector field projected onto some direction). Parameters      options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames. field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. c_map : str, default: None colormap to use overrides c_map_type and c_range_type. c_map_type : str, default: \"bfield_images\" colormap type to search options (options[\"colormaps\"][c_map_type]) for. c_map_range : str, default: \"percentile\" colormap range option (see  qdmpy.plot.common.get_colormap_range ) to use. c_range_vals : number or list, default: (5, 95) passed with c_map_range to get_colormap_range. c_bar_label : str, default: label to chuck on ye olde colorbar (z-axis label). Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl",
"url":10,
"doc":"Interface to mag simulations.  FIXME needs better documentation eh! Functions     - -  qdmpy.magsim.interface._plot_image_on_ax -  qdmpy.magsim.interface._add_cbar Classes    - -  qdmpy.magsim.interface.MagSim -  qdmpy.magsim.interface.SandboxMagSim -  qdmpy.magsim.interface.ComparisonMagSim "
},
{
"ref":"qdmpy.plot.pl.roi_pl_image",
"url":10,
"doc":"Plots full pl image with ROI region annotated. Arguments     - options : dict Generic options dict holding all the user options. pl_image : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned but not cut down to ROI. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.aoi_pl_image",
"url":10,
"doc":"Plots pl image cut down to ROI, with annotated AOI regions. Arguments     - options : dict Generic options dict holding all the user options. pl_image_roi : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and cut down to ROI. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.roi_avg_fits",
"url":10,
"doc":"Plots fit of spectrum averaged across ROI, as well as corresponding residual values. Arguments     - options : dict Generic options dict holding all the user options. backend_roi_results_lst : list of tuples Format: (fit_backend,  qdmpy.pl.common.ROIAvgFitResult objects), for each fit_backend Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.aoi_spectra",
"url":10,
"doc":"Plots spectra from each AOI, as well as subtraction and division norms. Arguments     - options : dict Generic options dict holding all the user options. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] ref : np array, 3D Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] sweep_list : list List of sweep parameter values (with removed unwanted sweeps at start/end) Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.aoi_spectra_fit",
"url":10,
"doc":"Plots sig and ref spectra, sub and div normalisation and fit for the ROI average, a single pixel, and each of the AOIs. All stacked on top of each other for comparison. The ROI average fit is plot against the fit of all of the others for comparison. Note here and elsewhere the single pixel check is the first element of the AOI array. NOTE this could be faster if we passed in sig_norm as well (backwards-compat. issues tho) Arguments     - options : dict Generic options dict holding all the user options. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] ref : np array, 3D Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] sweep_list : list List of sweep parameter values (with removed unwanted sweeps at start/end) fit_result_collection_lst : list List of  qdmpy.pl.common.FitResultCollection objects (one for each fit_backend) holding ROI, AOI fit results fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.pl_param_image",
"url":10,
"doc":"Plots an image corresponding to a single parameter in pixel_fit_params. Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. pixel_fit_params : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. param_name : str Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'. Optional arguments          param_number : int Which version of the parameter you want. I.e. there might be 8 independent parameters in the fit model called 'pos', each labeled 'pos_0', 'pos_1' etc. Default: 0. errorplot : bool Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so ylabel & save names are changed accordingly. Can't be True if param_name='residual'. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.pl_param_images",
"url":10,
"doc":"Plots images for all independent versions of a single parameter type in pixel_fit_params. Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. pixel_fit_params : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. param_name : str Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'. errorplot : bool Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so ylabel & save names are changed accordingly. Can't be True if param_name='residual'. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.pl_params_flattened",
"url":10,
"doc":"Compare pixel fits against flattened pixels: initial guess vs ROI fit vs fit result. Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. pixel_fit_params : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. roi_avg_fit_result  qdmpy.pl.common.ROIAvgFitResult object. param_name : str Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'. plot_bounds : bool Defaults to True, add fit bounds/constraints to plot. (Does nothing for residual plots) plot_sigmas : bool Defaults to True, add error bars (sigma) to plot. (Does nothing for residual plots) Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.other_measurements",
"url":10,
"doc":"polygons is dict (polygons directly) or str (path to)",
"func":1
},
{
"ref":"qdmpy.plot.pl._add_patch_rect",
"url":10,
"doc":"magnetizations: int/float if the same for all polygons, or an iterable of len(polygon_nodes) -> in units of mu_b / nm^2 (or mu_b / PX^2 for SandboxMagSim) unit_vectors: 3-iterable if the same for all polygons (cartesian coords), or an iterable of len(polygon_nodes) each element a 3-iterable",
"func":1
},
{
"ref":"qdmpy.plot.pl._annotate_roi_image",
"url":10,
"doc":"standoff: units of pixels for sandbox, units of m for comparison.",
"func":1
},
{
"ref":"qdmpy.plot.pl._annotate_aoi_image",
"url":10,
"doc":"Annotates AOI onto a given Axis object. Generally used on pl image.",
"func":1
},
{
"ref":"qdmpy.plot.pl.spectra_comparison",
"url":10,
"doc":"FIXME docstring TODO. locs =  y, x], [y, x],  .] . TODO add styles returns fig, dict (data) {as well as writing to output_dir}",
"func":1
},
{
"ref":"qdmpy.plot.common",
"url":11,
"doc":"This module holds tools for loading raw data etc. and reshaping to a usable format. Also contained here are functions for defining regions/areas of interest within the larger image dataset, that are then used in consequent functions, as well as the general options dictionary. Functions     - -  qdmpy.plot.common.set_mpl_rcparams -  qdmpy.plot.common.plot_image -  qdmpy.plot.common.plot_image_on_ax -  qdmpy.plot.common._add_colorbar -  qdmpy.plot.common.get_colormap_range -  qdmpy.plot.common._min_max -  qdmpy.plot.common._strict_range -  qdmpy.plot.common._min_max_sym_mean -  qdmpy.plot.common._min_max_sym_zero -  qdmpy.plot.common._deviation_from_mean -  qdmpy.plot.common._percentile -  qdmpy.plot.common._percentile_sym_zero "
},
{
"ref":"qdmpy.plot.common.set_mpl_rcparams",
"url":11,
"doc":"Reads matplotlib-relevant parameters in options and used to define matplotlib rcParams",
"func":1
},
{
"ref":"qdmpy.plot.common.plot_image",
"url":11,
"doc":"Plots an image given by image_data. Saves image_data as txt file as well as the figure. Arguments     - options : dict Generic options dict holding all the user options. image_data : np array, 3D Data that is plot. title : str Title of figure, as well as name for save files c_map : str Colormap object used to map image_data values to a color. c_range : str Range of values in image_data to map to colors c_label : str Label for colormap axis Returns    - fig : matplotlib Figure object ax : matplotlib Axis object",
"func":1
},
{
"ref":"qdmpy.magsim.interface.MagSim.plot_magsim_bfield_at_nvs",
"url":10,
"doc":"",
"func":1
},
{
"ref":"qdmpy.plot.common._add_colorbar",
"url":11,
"doc":"Adds a colorbar to matplotlib axis Arguments     - im : image as returned by ax.imshow fig : matplotlib Figure object ax : matplotlib Axis object Returns    - cbar : matplotlib colorbar object Optional Arguments          aspect : int Reciprocal of aspect ratio passed to new colorbar axis width. Default: 20. pad_fraction : int Fraction of new colorbar axis width to pad from image. Default: 1.  kwargs : other keyword arguments Passed to fig.colorbar.",
"func":1
},
{
"ref":"qdmpy.magsim.interface.SandboxMagSim.add_template_polygons",
"url":10,
"doc":"polygons takes precedence.",
"func":1
},
{
"ref":"qdmpy.plot.common._min_max",
"url":11,
"doc":"Map between minimum and maximum values in image Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.plot.common._strict_range",
"url":11,
"doc":"Map between c_range_values Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.magsim.interface.SandboxMagSim.add_polygons",
"url":10,
"doc":"polygons is dict (polygons directly) or str (path to)",
"func":1
},
{
"ref":"qdmpy.plot.common._min_max_sym_zero",
"url":11,
"doc":"Map symmetrically about zero, capturing all values in image. Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.magsim.interface.SandboxMagSim.run",
"url":10,
"doc":"standoff: units of pixels for sandbox, units of m for comparison.",
"func":1
},
{
"ref":"qdmpy.magsim.interface.ComparisonMagSim",
"url":10,
"doc":""
},
{
"ref":"qdmpy.magsim.interface.ComparisonMagSim.unscaled_polygon_nodes",
"url":10,
"doc":""
},
{
"ref":"qdmpy.magsim.interface.ComparisonMagSim.rescale",
"url":10,
"doc":"",
"func":1
},
{
"ref":"qdmpy.magsim.interface.ComparisonMagSim.select_polygons",
"url":10,
"doc":"",
"func":1
},
{
"ref":"qdmpy.plot.source.current_quiver",
"url":12,
"doc":"",
"func":1
},
{
"ref":"qdmpy.magsim.interface.ComparisonMagSim.add_polygons",
"url":10,
"doc":"polygons is dict (polygons directly) or str (path to)",
"func":1
},
{
"ref":"qdmpy.plot.source.current_stream",
"url":12,
"doc":"Plot current density as a streamplot. Arguments     - options : dict Generic options dict holding all the user options. source_params : dict Dictionary, key: param_keys, val: image (2D) of (source field) param values across FOV. background_image : numpy array or None, default: None If not None, must be pl image of ROI (or other background image), to plot behind streams Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.magsim.interface.ComparisonMagSim.run",
"url":10,
"doc":"standoff: units of pixels for sandbox, units of m for comparison.",
"func":1
},
{
"ref":"qdmpy.plot.source.divperp_j",
"url":12,
"doc":"plot perpindicular divergence of J, i.e. in-plane divergence (dJ/dx + dJ/dy). Parameters      options : dict Generic options dict holding all the user options. source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV. sigma : int Gaussian smoothing width. Ignored if less than or equal to 1. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.source",
"url":13,
"doc":"Sub-package for inverting fields to calculate their source (field)."
},
{
"ref":"qdmpy.source.io",
"url":14,
"doc":"This module holds the tools for loading/saving source results. Functions     - -  qdmpy.source.io.prep_output_directories -  qdmpy.source.io.save_source_params "
},
{
"ref":"qdmpy.source.io.prep_output_directories",
"url":14,
"doc":"",
"func":1
},
{
"ref":"qdmpy.source.io.save_source_params",
"url":14,
"doc":"",
"func":1
},
{
"ref":"qdmpy.source.interface",
"url":15,
"doc":"Interface for source sub-module. Functions     - -  qdmpy.source.interface.odmr_source_retrieval -  qdmpy.source.interface.get_current_density -  qdmpy.source.interface.get_magnetization -  qdmpy.source.interface.add_divperp_j -  qdmpy.source.interface.in_plane_mag_normalise "
},
{
"ref":"qdmpy.source.interface.odmr_source_retrieval",
"url":15,
"doc":"Calculates source field that created field measured by bnvs and field params. Arguments     - options : dict Generic options dict holding all the user options. bnvs : list List of bnv results (each a 2D image). sig_fit_params : dict Dictionary, key: param_keys, val: image (2D) of field values across FOV. Returns    - source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV. For methodology see D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.interface.get_current_density",
"url":15,
"doc":"Gets current density from bnvs and field_params according to options in options. Returned as a dict similar to fit_params/field_params etc. Arguments     - options : dict Generic options dict holding all the user options. bnvs : list List of bnv results (each a 2D image). sig_fit_params : dict Dictionary, key: param_keys, val: image (2D) of field values across FOV. Returns    - source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.interface.get_magnetization",
"url":15,
"doc":"Gets magnetization from bnvs and field_params according to options in options. Returned as a dict similar to fit_params/field_params etc. Arguments     - options : dict Generic options dict holding all the user options. bnvs : list List of bnv results (each a 2D image). sig_fit_params : dict Dictionary, key: param_keys, val: image (2D) of field values across FOV. Returns    - source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.interface.add_divperp_j",
"url":15,
"doc":"jxy -> Divperp J Divperp = divergence in x and y only (perpendicular to surface normal) Arguments     - options : dict Generic options dict holding all the user options (for the main/signal experiment). source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV. Returns    - nothing (operates in place on field_params)  \\nabla \\times {\\bf J} = \\frac{\\partial {\\bf J} }{\\partial x} + \\frac{\\partial {\\bf J {\\partial y} + \\frac{\\partial {\\bf J {\\partial z}   \\nabla_{\\perp} \\times {\\bf J} = \\frac{\\partial {\\bf J} }{\\partial x} + \\frac{\\partial {\\bf J {\\partial y}  ",
"func":1
},
{
"ref":"qdmpy.source.interface.in_plane_mag_normalise",
"url":15,
"doc":"Normalise in-plane magnetization by taking average of mag near edge of image per line @ psi. The jist of this function was copied from D. Broadway's previous version of the code. Parameters      mag_image : np array 2D magnetization array as directly calculated. psi : float Assumed in-plane magnetization angle (deg) edge_pixels_used : int Number of pixels to use at edge of image to calculate average to subtract. Returns    - mag_image : np array in-plane magnetization image with line artifacts substracted.",
"func":1
},
{
"ref":"qdmpy.source.current",
"url":16,
"doc":"Implement inversion of magnetic field to current density. Functions     - -  qdmpy.source.current.get_divperp_j -  qdmpy.source.current.get_j_from_bxy -  qdmpy.source.current.get_j_from_bz -  qdmpy.source.current.get_j_from_bnv -  qdmpy.source.current.get_j_without_ft "
},
{
"ref":"qdmpy.source.current.get_divperp_j",
"url":16,
"doc":"jxy calculated -> perpindicular (in-plane) divergence of j Arguments     - jvec : list List of current components, e.g [jx_image, jy_image] pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. Returns    - jx_recon, jy_recon : np arrays (2D) The reconstructed j (source) field maps.  \\nabla \\times {\\bf J} = \\frac{\\partial {\\bf J} }{\\partial x} + \\frac{\\partial {\\bf J {\\partial y} + \\frac{\\partial {\\bf J {\\partial z}   \\nabla_{\\perp} \\times {\\bf J} = \\frac{\\partial {\\bf J} }{\\partial x} + \\frac{\\partial {\\bf J {\\partial y}  ",
"func":1
},
{
"ref":"qdmpy.source.current.get_j_from_bxy",
"url":16,
"doc":"Bxy measured -> Jxy via fourier methods. Arguments     - bfield : list List of magnetic field components, e.g [bx_image, by_image, bz_image] pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. do_hanning_filter : bool Do a hanning filter? hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. standoff : float Distance NV layer  Sample (in metres) nv_layer_thickness : float Thickness of NV layer (in metres) nvs_above_sample : bool True if NVs exist at higher z (in lab frame) than sample. Returns    - jx, jy : np arrays (2D) The calculated current density images, in A/m. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.current.get_j_from_bz",
"url":16,
"doc":"Bz measured -> Jxy via fourier methods. Arguments     - bfield : list List of magnetic field components, e.g [bx_image, by_image, bz_image] pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. do_hanning_filter : bool Do a hanning filter? hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. standoff : float Distance NV layer  Sample (in metres) nv_layer_thickness : float Thickness of NV layer (in metres) Returns    - jx, jy : np arrays (2D) The calculated current density images, in A/m. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.current.get_j_without_ft",
"url":16,
"doc":"Bxy measured -> (pseudo-)Jxy  without fourier methods. Arguments     - bfield : list List of magnetic field components, e.g [bx_image, by_image, bz_image] Returns    - jx, jy : np arrays (2D) The calculated current density images, in A/m. Get Jx, Jy approximation, without any fourier propogation. Simply rescale to currect units (Teslas  2 / mu_0)",
"func":1
},
{
"ref":"qdmpy.source.current.get_j_from_bnv",
"url":16,
"doc":"(Single) Bnv measured -> Jxy via fourier methods. Arguments     - single_bnv : np array Single bnv map (np 2D array). unv : array-like, 1D Shape: 3, the uNV_Z corresponding to the above bnv map. pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. do_hanning_filter : bool Do a hanning filter? hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. standoff : float Distance NV layer  Sample (in metres) nv_layer_thickness : float Thickness of NV layer (in metres) nvs_above_sample : bool True if NVs exist at higher z (in lab frame) than sample. Returns    - jx, jy : np arrays (2D) The calculated current density images, in A/m. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.pl.funcs",
"url":13,
"doc":"This module defines the fit functions used in the  qdmpy.pl.model.FitModel . We grab/use this regardless of fitting on cpu (scipy) or gpu etc. Ensure any fit functions you define are added to the AVAILABLE_FNS module variable at bottom. Try not to have overlapping parameter names in the same fit. For ODMR peaks, ensure the frequency position of the peak is named something prefixed by 'pos'. (see  qdmpy.field.bnv.get_bnvs_and_dshifts for the reasoning). Classes    - -  qdmpy.pl.funcs.FitFunc -  qdmpy.pl.funcs.Constant -  qdmpy.pl.funcs.Linear -  qdmpy.pl.funcs.Circular -  qdmpy.pl.funcs.Gaussian -  qdmpy.pl.funcs.GaussianHyperfine14 -  qdmpy.pl.funcs.GaussianHyperfine15 -  qdmpy.pl.funcs.Lorentzian -  qdmpy.pl.funcs.LorentzianHyperfine14 -  qdmpy.pl.funcs.LorentzianHyperfine15 -  qdmpy.pl.funcs.StretchedExponential -  qdmpy.pl.funcs.DampedRabi "
},
{
"ref":"qdmpy.pl.funcs.FitFunc",
"url":13,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.FitFunc.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.FitFunc.grad_fn",
"url":13,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Constant",
"url":13,
"doc":"Constant Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Constant.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Constant.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Constant.eval",
"url":13,
"doc":"speed tested multiple methods, this was the fastest",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Constant.grad_fn",
"url":13,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 1)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Linear",
"url":13,
"doc":"Linear function, y=mx+c Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Linear.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Linear.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Linear.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Linear.grad_fn",
"url":13,
"doc":"Compute the grad of the residual, excluding pl as a param {output shape: (len(x), 2)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Circular",
"url":13,
"doc":"Circular function (sine) Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Circular.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Circular.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Circular.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Circular.grad_fn",
"url":13,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 3)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Gaussian",
"url":13,
"doc":"Gaussian function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Gaussian.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Gaussian.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Gaussian.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Gaussian.grad_fn",
"url":13,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14",
"url":13,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine14.grad_fn",
"url":13,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15",
"url":13,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.GaussianHyperfine15.grad_fn",
"url":13,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Lorentzian",
"url":13,
"doc":"Lorentzian function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.Lorentzian.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Lorentzian.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.Lorentzian.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.Lorentzian.grad_fn",
"url":13,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 3)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14",
"url":13,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine14.grad_fn",
"url":13,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15",
"url":13,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.LorentzianHyperfine15.grad_fn",
"url":13,
"doc":"if you want to use a grad_fn override this in the subclass",
"func":1
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential",
"url":13,
"doc":"Singular fit function Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.StretchedExponential.grad_fn",
"url":13,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 3)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.DampedRabi",
"url":13,
"doc":"Damped oscillation Arguments     - param_indices : np array Where the parameters for this fitfunc are located within broader fitmodel param array."
},
{
"ref":"qdmpy.pl.funcs.DampedRabi.param_defn",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.DampedRabi.param_units",
"url":13,
"doc":""
},
{
"ref":"qdmpy.pl.funcs.DampedRabi.eval",
"url":13,
"doc":"",
"func":1
},
{
"ref":"qdmpy.pl.funcs.DampedRabi.grad_fn",
"url":13,
"doc":"Compute the grad of the residue, excluding pl as a param {output shape: (len(x), 4)}",
"func":1
},
{
"ref":"qdmpy.pl.funcs.AVAILABLE_FNS",
"url":13,
"doc":"Dictionary that defines fit functions available for use. Add any functions you define here so you can use them. Aviod overlapping function parameter names."
},
{
"ref":"qdmpy.pl.interface",
"url":14,
"doc":"This module holds the general interface tools for fitting data, independent of fit backend (e.g. scipy/gpufit etc.). All of these functions are automatically loaded into the namespace when the fit sub-package is imported. (e.g. import qdmpy.fit). Functions     - -  qdmpy.pl.interface.define_fit_model -  qdmpy.pl.interface.fit_roi_avg_pl -  qdmpy.pl.interface.fit_aois_pl -  qdmpy.pl.interface.fit_all_pixels_pl -  qdmpy.pl.interface._prep_fit_backends -  qdmpy.pl.interface.get_pl_fit_result "
},
{
"ref":"qdmpy.pl.interface.define_fit_model",
"url":14,
"doc":"Define (and return) fit_model object, from options dictionary. Arguments     - options : dict Generic options dict holding all the user options. Returns    - fit_model :  qdmpy.pl.model.FitModel FitModel used to fit to data.",
"func":1
},
{
"ref":"qdmpy.pl.interface._prep_fit_backends",
"url":14,
"doc":"Prepare all possible fit backends, checking that everything will work. Also attempts to import relevant modules into global scope. This is a wrapper around specific functions for each backend. All possible fit backends are loaded - these are decided in the config file for this system, i.e. system.option_choices(\"fit_backend\") Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel FitModel used to fit to data.",
"func":1
},
{
"ref":"qdmpy.pl.interface.fit_roi_avg_pl",
"url":14,
"doc":"Fit the average of the measurement over the region of interest specified, with backend chosen via options[\"fit_backend_comparison\"]. Arguments     - options : dict Generic options dict holding all the user options. sig_norm : np array, 3D Normalised measurement array, shape: [sweep_list, y, x]. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. Returns    - backend_roi_results_lst : list List of  qdmpy.pl.common.ROIAvgFitResult objects containing the fit result (see class specifics) for each fit backend selected for comparison.",
"func":1
},
{
"ref":"qdmpy.pl.interface.fit_aois_pl",
"url":14,
"doc":"Fit AOIs and single pixel with chosen backends and return fit_result_collection_lst Arguments     - options : dict Generic options dict holding all the user options. sig_norm : np array, 3D Normalised measurement array, shape: [sweep_list, y, x]. single_pixel_pl : np array, 1D Normalised measurement array, for chosen single pixel check. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq). fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. backend_roi_results_lst : list List of  qdmpy.pl.common.ROIAvgFitResult objects, to pull fit_options from. Returns    - fit_result_collection :  qdmpy.pl.common.FitResultCollection  qdmpy.pl.common.FitResultCollection object.",
"func":1
},
{
"ref":"qdmpy.pl.interface.fit_all_pixels_pl",
"url":14,
"doc":"Fit all pixels in image with chosen fit backend. Arguments     - options : dict Generic options dict holding all the user options. sig_norm : np array, 3D Normalised measurement array, shape: [sweep_list, y, x]. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. roi_avg_fit_result :  qdmpy.pl.common.ROIAvgFitResult  qdmpy.pl.common.ROIAvgFitResult object, to pull fit_options from. Returns    - fit_image_results : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. Also has 'residual' as a key. sigmas : dict As above, but standard deviation for each param",
"func":1
},
{
"ref":"qdmpy.pl.interface.get_pl_fit_result",
"url":14,
"doc":"Fit all pixels in image with chosen fit backend (or loads previous fit result) Wrapper for  qdmpy.pl.interface.fit_all_pixels_pl with some options logic. Arguments     - options : dict Generic options dict holding all the user options. sig_norm : np array, 3D Normalised measurement array, shape: [sweep_list, y, x]. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. wanted_roi_result :  qdmpy.pl.common.ROIAvgFitResult  qdmpy.pl.common.ROIAvgFitResult object, to pull fit_options from. Returns    - fit_image_results : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. Also has 'residual' as a key. (or None if not fitting pixels, this stops plotting, e.g. via  qdmpy.plot.pl.pl_param_images from erroring) sigmas : dict As above, but standard deviation for each param (or None if not fitting pixels)",
"func":1
},
{
"ref":"qdmpy.pl.io",
"url":15,
"doc":"This module holds tools for pl input/output Functions     - -  qdmpy.pl.io.load_image_and_sweep -  qdmpy.pl.io.reshape_dataset -  qdmpy.pl.io.save_pl_data -  qdmpy.pl.io._rebin_image -  qdmpy.pl.io._remove_unwanted_data -  qdmpy.pl.io._check_start_end_rectangle -  qdmpy.pl.io.load_prev_pl_fit_results -  qdmpy.pl.io.load_prev_pl_fit_sigmas -  qdmpy.pl.io.load_pl_fit_sigma -  qdmpy.pl.io.load_fit_param -  qdmpy.pl.io.save_pl_fit_sigmas -  qdmpy.pl.io.save_pl_fit_results -  qdmpy.pl.io.load_ref_exp_pl_fit_results -  qdmpy.pl.io.check_if_already_fit -  qdmpy.pl.io.prev_options_exist -  qdmpy.pl.io.options_compatible -  qdmpy.pl.io._prev_pl_fits_exist -  qdmpy.pl.io._prev_pl_sigmas_exist -  qdmpy.pl.io.get_prev_options "
},
{
"ref":"qdmpy.pl.io.load_image_and_sweep",
"url":15,
"doc":"Reads raw image data and sweep_list (affine parameters) using system methods Arguments     - options : dict Generic options dict holding all the user options. Returns    - image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. sweep_list : list List of sweep parameter values",
"func":1
},
{
"ref":"qdmpy.pl.io.reshape_dataset",
"url":15,
"doc":"Reshapes and re-bins raw data into more useful format. Cuts down to ROI and removes unwanted sweeps. Arguments     - options : dict Generic options dict holding all the user options. image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. sweep_list : list List of sweep parameter values Returns    - pl_image : np array, 2D. Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and cut down to ROI. pl_image_ROI : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and cut down to ROI. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] ref : np array, 3D Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] sig_norm : np array, 3D Signal normalised by reference (via subtraction or normalisation, chosen in options), reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] single_pixel_pl : np array, 1D Normalised measurement array, for chosen single pixel check. sweep_list : list List of sweep parameter values (with removed unwanted sweeps at start/end) roi : length 2 list of np meshgrids Defines an ROI that can be applied to the 3D image through direct indexing. E.g. sig_ROI = sig[:, roi[0], roi[1 (post rebinning)",
"func":1
},
{
"ref":"qdmpy.pl.io.save_pl_data",
"url":15,
"doc":"Saves pl_image and pl_image_ROI to disk",
"func":1
},
{
"ref":"qdmpy.pl.io._rebin_image",
"url":15,
"doc":"Reshapes raw data into more useful shape, according to image size in metadata. Arguments     - options : dict Generic options dict holding all the user options. image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Returns    - image_rebinned : np array, 3D Format: [sweep values, y, x]. Same as image, but now rebinned (x size and y size have changed). Not cut down to ROI. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet. Format: [sweep_vals, y, x]. Not cut down to ROI. ref : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet. Not cut down to ROI. Format: [sweep_vals, y, x]. sig_norm : np array, 3D Signal normalised by reference (via subtraction or normalisation, chosen in options), reshaped and rebinned. Unwanted sweeps not removed yet. Not cut down to ROI. Format: [sweep_vals, y, x].",
"func":1
},
{
"ref":"qdmpy.pl.io._remove_unwanted_data",
"url":15,
"doc":"Removes unwanted sweep values (i.e. freq values or tau values) for all of the data arrays. Also cuts data down to ROI. Arguments     - options : dict Generic options dict holding all the user options. image_rebinned : np array, 3D Format: [sweep values, y, x]. Same as image, but rebinned (x size and y size have changed). sweep_list : list List of sweep parameter values sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet. Format: [sweep_vals, y, x] ref : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps not removed yet. Format: [sweep_vals, y, x] sig_norm : np array, 3D Signal normalised by reference (via subtraction or normalisation, chosen in options), reshaped and rebinned. Unwanted sweeps not removed yet. Format: [sweep_vals, y, x] Returns    - pl_image : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and cut down to ROI pl_image_ROI : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped and rebinned as well as cut down to ROI. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed. Format: [sweep_vals, y, x] ref : np array, 3D Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed. Format: [sweep_vals, y, x] sig_norm : np array, 3D Signal normalised by reference (via subtraction or normalisation, chosen in options), reshaped and rebinned. Unwanted sweeps removed. Format: [sweep_vals, y, x] sweep_list : list List of sweep parameter values (with removed unwanted sweeps at start/end) roi : length 2 list of np meshgrids Defines an ROI that can be applied to the 3D image through direct indexing. E.g. sig_ROI = sig[:, roi[0], roi[1 ",
"func":1
},
{
"ref":"qdmpy.pl.io._check_start_end_rectangle",
"url":15,
"doc":"Checks that 'name' rectange (defined by top left corner 'start_x', 'start_y' and bottom right corner 'end_x', 'end_y') fits within a larger rectangle of size 'full_size_w', 'full_size_h'. If there are no good, they're fixed with a warning. Arguments     - name : str The name of the rectangle we're checking (e.g. \"ROI\", \"AOI\"). start_x : int x coordinate (relative to origin) of rectangle's top left corner. start_y : int y coordinate (relative to origin) of rectangle's top left corner. end_x : int x coordinate (relative to origin) of rectangle's bottom right corner. end_y : int y coordinate (relative to origin) of rectangle's bottom right corner. full_size_w : int Full width of image (or image region, e.g. ROI). full_size_h : int Full height of image (or image region, e.g. ROI). Returns    - start_coords : tuple 'fixed' start coords: (start_x, start_y) end_coords : tuple 'fixed' end coords: (end_x, end_y)",
"func":1
},
{
"ref":"qdmpy.pl.io.load_prev_pl_fit_results",
"url":15,
"doc":"Load (all) parameter pl fits from previous processing.",
"func":1
},
{
"ref":"qdmpy.pl.io.load_prev_pl_fit_sigmas",
"url":15,
"doc":"Load (all) parameter pl fits from previous processing.",
"func":1
},
{
"ref":"qdmpy.pl.io.load_pl_fit_sigma",
"url":15,
"doc":"Load a previous fit sigma result, of name 'param_key'",
"func":1
},
{
"ref":"qdmpy.pl.io.load_fit_param",
"url":15,
"doc":"Load a previously fit param, of name 'param_key'.",
"func":1
},
{
"ref":"qdmpy.pl.io.save_pl_fit_sigmas",
"url":15,
"doc":"Saves pixel fit sigmas to disk. Arguments     - options : dict Generic options dict holding all the user options. sigmas : OrderedDict Dictionary, key: param_keys, val: image (2D) of param sigmas across FOV.",
"func":1
},
{
"ref":"qdmpy.pl.io.save_pl_fit_results",
"url":15,
"doc":"Saves pl fit results to disk. Arguments     - options : dict Generic options dict holding all the user options. pixel_fit_params : OrderedDict Dictionary, key: param_keys, val: image (2D) of param values across FOV.",
"func":1
},
{
"ref":"qdmpy.pl.io.load_ref_exp_pl_fit_results",
"url":15,
"doc":"ref_options dict -> pixel_fit_params dict. Provide one of ref_options and ref_options_dir. If both are None, returns None (with a warning). If both are supplied, ref_options takes precedence. Arguments     - ref_options : dict, default=None Generic options dict holding all the user options (for the reference experiment). Returns    - fit_result_dict : OrderedDict Dictionary, key: param_keys, val: image (2D) of (fit) param values across FOV. If no reference experiment is given (i.e. ref_options and ref_options_dir are None) then returns None fit_result_sigmas : OrderedDict Dict as above, but fit sigmas",
"func":1
},
{
"ref":"qdmpy.pl.io.check_if_already_fit",
"url":15,
"doc":"Looks for previous fit result. If previous fit result exists, checks for compatibility between option choices. loading_ref (bool): skip checks for force_fit etc. and just see if prev pixel results exist. Returns nothing.",
"func":1
},
{
"ref":"qdmpy.pl.io.prev_options_exist",
"url":15,
"doc":"Checks if options file from previous result can be found in default location, returns Bool.",
"func":1
},
{
"ref":"qdmpy.pl.io.options_compatible",
"url":15,
"doc":"Checks if option choices are compatible with previously fit options Arguments     - options : dict Generic options dict holding all the user options. prev_options : dict Generic options dict from previous fit result. Returns    - options_compatible : bool Whether or not options are compatible. reason : str Reason for the above",
"func":1
},
{
"ref":"qdmpy.pl.io._prev_pl_fits_exist",
"url":15,
"doc":"Check if the actual fit result files exists. Arguments     - options : dict Generic options dict from (either prev. or current, should be the equiv.) fit result. Returns    - pixels_results_exist : bool Whether or not previous pixel result files exist. reason : str Reason for the above",
"func":1
},
{
"ref":"qdmpy.pl.io._prev_pl_sigmas_exist",
"url":15,
"doc":"as  qdmpy.pl.io._prev_pl_fits_exist but for sigmas",
"func":1
},
{
"ref":"qdmpy.pl.io.get_prev_options",
"url":15,
"doc":"Reads options file from previous fit result (.json), returns a dictionary.",
"func":1
},
{
"ref":"qdmpy.pl.model",
"url":16,
"doc":"This module defines fit models used to fit QDM photoluminescence data. We grab/use this regardless of fitting on cpu (scipy) or gpu etc. Ensure any fit functions you define are added to the AVAILABLE_FNS module variable. Try not to have overlapping parameter names in the same fit. For ODMR peaks, ensure the frequency position of the peak is named something prefixed by 'pos'. (see  qdmpy.field.bnv.get_bnvs_and_dshifts for the reasoning). Classes    - -  qdmpy.pl.model.FitModel "
},
{
"ref":"qdmpy.pl.model.FitModel",
"url":16,
"doc":"FitModel used to fit to data. Arguments     - fit_functions: dict Dict of functions to makeup the fit model, key: fitfunc name, val: number of independent copies of that fitfunc. format: {\"linear\": 1, \"lorentzian\": 8} etc., i.e. options[\"fit_functions\"]"
},
{
"ref":"qdmpy.pl.model.FitModel.residuals_scipyfit",
"url":16,
"doc":"Evaluates residual: fit model (affine params/sweep_vec) - pl values",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.jacobian_scipyfit",
"url":16,
"doc":"Evaluates (analytic) jacobian of fitmodel in format expected by scipy least_squares",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.jacobian_defined",
"url":16,
"doc":"Check if analytic jacobian is defined for this fit model.",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.get_param_defn",
"url":16,
"doc":"Returns list of parameters in fit_model, note there will be duplicates, and they do not have numbers e.g. 'pos_0'. Use  qdmpy.pl.model.FitModel.get_param_odict for that purpose. Returns    - param_defn_ar : list List of parameter names (param_defn) in fit model.",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.get_param_odict",
"url":16,
"doc":"get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model Returns    - param_dict : dict Dictionary containing key: params, values: units.",
"func":1
},
{
"ref":"qdmpy.pl.model.FitModel.get_param_unit",
"url":16,
"doc":"Get unit for a given param_key (given by param_name + \"_\" + param_number) Arguments     - param_name : str Name of parameter, e.g. 'pos' param_number : float or int Which parameter to use, e.g. 0 for 'pos_0' Returns    - unit : str Unit for that parameter, e.g. \"constant\" -> \"Amplitude (a.u.) ",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit",
"url":17,
"doc":"This module holds tools for fitting raw data via scipy. (scipy backend) Functions     - -  qdmpy.pl.scipyfit.prep_scipyfit_options -  qdmpy.pl.scipyfit.gen_scipyfit_init_guesses -  qdmpy.pl.scipyfit.fit_roi_avg_pl_scipyfit -  qdmpy.pl.scipyfit.fit_single_pixel_pl_scipyfit -  qdmpy.pl.scipyfit.fit_aois_pl_scipyfit -  qdmpy.pl.scipyfit.limit_cpu -  qdmpy.pl.scipyfit.to_squares_wrapper -  qdmpy.pl.scipyfit.fit_all_pixels_pl_scipyfit "
},
{
"ref":"qdmpy.pl.scipyfit.prep_scipyfit_options",
"url":17,
"doc":"General options dict -> scipyfit_options in format that scipy least_squares expects. Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel Fit model object. Returns    - scipy_fit_options : dict Dictionary with options that scipy.optimize.least_squares expects, specific to fitting.",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.gen_scipyfit_init_guesses",
"url":17,
"doc":"Generate arrays of initial fit guesses and bounds in correct form for scipy least_squares. init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays, that scipy will recognise. In particular, we specificy that each of the 'num' of each 'fn_type' have independent parameters, so must have independent init_guesses and init_bounds when plugged into scipy. Arguments     - options : dict Generic options dict holding all the user options. init_guesses : dict Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent version of that fn_type. init_bounds : dict Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent version of that fn_type. Returns    - fit_param_ar : np array, shape: num_params The initial fit parameter guesses. fit_param_bound_ar : np array, shape: (num_params, 2) Fit parameter bounds.",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.fit_roi_avg_pl_scipyfit",
"url":17,
"doc":"Fit the average of the measurement over the region of interest specified, with scipy. Arguments     - options : dict Generic options dict holding all the user options. sig_norm : np array, 3D Normalised measurement array, shape: [sweep_list, y, x]. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel The fit model object. Returns    -  qdmpy.pl.common.ROIAvgFitResult object containing the fit result (see class specifics)",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.fit_single_pixel_pl_scipyfit",
"url":17,
"doc":"Fit Single pixel and return best_fit_result.x (i.e. the optimal fit parameters) Arguments     - options : dict Generic options dict holding all the user options. pixel_pl_ar : np array, 1D Normalised pl as function of sweep_list for a single pixel. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel The fit model. roi_avg_fit_result :  qdmpy.pl.common.ROIAvgFitResult  qdmpy.pl.common.ROIAvgFitResult object, to pull fit_options from. Returns    - pixel_parameters : np array, 1D Best fit parameters, as determined by scipy.",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.fit_aois_pl_scipyfit",
"url":17,
"doc":"Fit AOIs and single pixel and return list of (list of) fit results (optimal fit parameters), using scipy. Arguments     - options : dict Generic options dict holding all the user options. sig_norm : np array, 3D Normalised measurement array, shape: [sweep_list, y, x]. single_pixel_pl : np array, 1D Normalised measurement array, for chosen single pixel check. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq). fit_model :  qdmpy.pl.model.FitModel The model we're fitting to. aois : list List of AOI specifications - each a length-2 iterable that can be used to directly index into sig_norm to return that AOI region, e.g. sig_norm[:, AOI[0], AOI[1 . roi_avg_fit_result :  qdmpy.pl.common.ROIAvgFitResult  qdmpy.pl.common.ROIAvgFitResult object, to pull  qdmpy.pl.common.ROIAvgFitResult.fit_options from. Returns    - fit_result_collection :  qdmpy.pl.common.FitResultCollection Collection of ROI/AOI fit results for this fit backend.",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.limit_cpu",
"url":17,
"doc":"Called at every process start, to reduce the priority of this process",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.to_squares_wrapper",
"url":17,
"doc":"Simple wrapper of scipy.optimize.least_squares to allow us to keep track of which solution is which (or where). Arguments     - fun : function Function object acting as residual (fit model minus pl value) p0 : np array Initial guess: array of parameters sweep_vec : np array Array (or I guess single value, anything iterable) of affine parameter (tau/freq) shaped_data : list (3 elements) array returned by  qdmpy.pl.common.pixel_generator : [y, x, sig_norm[:, y, x fit_optns : dict Other options (dict) passed to least_squares Returns    - wrapped_squares : tuple (y, x), least_squares( .).x, leas_squares( .).jac I.e. the position of the fit result, the fit result parameters array, jacobian at solution",
"func":1
},
{
"ref":"qdmpy.pl.scipyfit.fit_all_pixels_pl_scipyfit",
"url":17,
"doc":"Fits each pixel and returns dictionary of param_name -> param_image. Arguments     - options : dict Generic options dict holding all the user options. sig_norm : np array, 3D Normalised measurement array, shape: [sweep_list, y, x]. sweep_list : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  qdmpy.pl.model.FitModel The model we're fitting to. roi_avg_fit_result :  qdmpy.pl.common.ROIAvgFitResult  qdmpy.pl.common.ROIAvgFitResult object, to pull fit_options from. Returns    - fit_image_results : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. Also has 'residual' as a key.",
"func":1
},
{
"ref":"qdmpy.plot",
"url":18,
"doc":"Sub-package for plotting results of qdmpy. This (sub-) package exposes all of the public members of the following modules: -  qdmpy.plot.common -  qdmpy.plot.field -  qdmpy.plot.pl -  qdmpy.plot.source "
},
{
"ref":"qdmpy.plot.common",
"url":19,
"doc":"This module holds tools for loading raw data etc. and reshaping to a usable format. Also contained here are functions for defining regions/areas of interest within the larger image dataset, that are then used in consequent functions, as well as the general options dictionary. Functions     - -  qdmpy.plot.common.set_mpl_rcparams -  qdmpy.plot.common.plot_image -  qdmpy.plot.common.plot_image_on_ax -  qdmpy.plot.common._add_colorbar -  qdmpy.plot.common.get_colormap_range -  qdmpy.plot.common._min_max -  qdmpy.plot.common._strict_range -  qdmpy.plot.common._min_max_sym_mean -  qdmpy.plot.common._min_max_sym_zero -  qdmpy.plot.common._deviation_from_mean -  qdmpy.plot.common._percentile -  qdmpy.plot.common._percentile_sym_zero "
},
{
"ref":"qdmpy.plot.common.set_mpl_rcparams",
"url":19,
"doc":"Reads matplotlib-relevant parameters in options and used to define matplotlib rcParams",
"func":1
},
{
"ref":"qdmpy.plot.common.plot_image",
"url":19,
"doc":"Plots an image given by image_data. Saves image_data as txt file as well as the figure. Arguments     - options : dict Generic options dict holding all the user options. image_data : np array, 3D Data that is plot. title : str Title of figure, as well as name for save files c_map : str Colormap object used to map image_data values to a color. c_range : str Range of values in image_data to map to colors c_label : str Label for colormap axis Returns    - fig : matplotlib Figure object ax : matplotlib Axis object",
"func":1
},
{
"ref":"qdmpy.plot.common.plot_image_on_ax",
"url":19,
"doc":"Plots an image given by image_data onto given figure and ax. Does not save any data. Arguments     - fig : matplotlib Figure object ax : matplotlib Axis object options : dict Generic options dict holding all the user options. image_data : np array, 3D Data that is plot. title : str Title of figure, as well as name for save files c_map : str Colormap object used to map image_data values to a color. c_range : str Range of values in image_data to map to colors c_label : str Label for colormap axis Returns    - fig : matplotlib Figure object ax : matplotlib Axis object",
"func":1
},
{
"ref":"qdmpy.plot.common._add_colorbar",
"url":19,
"doc":"Adds a colorbar to matplotlib axis Arguments     - im : image as returned by ax.imshow fig : matplotlib Figure object ax : matplotlib Axis object Returns    - cbar : matplotlib colorbar object Optional Arguments          aspect : int Reciprocal of aspect ratio passed to new colorbar axis width. Default: 20. pad_fraction : int Fraction of new colorbar axis width to pad from image. Default: 1.  kwargs : other keyword arguments Passed to fig.colorbar.",
"func":1
},
{
"ref":"qdmpy.plot.common.get_colormap_range",
"url":19,
"doc":"Produce a colormap range to plot image from, using the options in c_range_dict. Arguments     - c_range_dict : dict Dictionary with key 'values', used to accompany some of the options below, as well as a 'type', one of : - \"min_max\" : map between minimum and maximum values in image. - \"deviation_from_mean\" : requires c_range_dict[\"values\"] be a float between 0 and 1 'dev'. Maps between (1 - dev)  mean and (1 + dev)  mean. - \"min_max_symmetric_about_mean\" : map symmetrically about mean, capturing all values in image (default). - \"min_max_symmetric_about_zero\" : map symmetrically about zero, capturing all values in image. - \"percentile\" : requires c_range_dict[\"values\"] be a list of two numbers between 0 and 100. Maps the range between those percentiles of the data. - \"percentile_symmetric_about_zero\" : requires c_range_dict[\"values\"] be a list of two numbers between 0 and 100. Maps symmetrically about zero, capturing all values between those percentiles in the data (plus perhaps a bit more to ensure symmety) - \"strict_range\" : requires c_range_dict[\"values\"] be list length 2 of floats or ints. Maps colors between the values given. - \"mean_plus_minus\" : mean plus or minus this value. c_range_dict[\"values\"] must be an int or float. as well as accompanying 'values' key, used for some of the options below image : array-like 2D array (image) that fn returns colormap range for. Returns    - c_range : list length 2 i.e. [min value to map to a color, max value to map to a color]",
"func":1
},
{
"ref":"qdmpy.plot.common._min_max",
"url":19,
"doc":"Map between minimum and maximum values in image Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.plot.common._strict_range",
"url":19,
"doc":"Map between c_range_values Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.plot.common._min_max_sym_mean",
"url":19,
"doc":"Map symmetrically about mean, capturing all values in image. Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.plot.common._min_max_sym_zero",
"url":19,
"doc":"Map symmetrically about zero, capturing all values in image. Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.plot.common._deviation_from_mean",
"url":19,
"doc":"Map a (decimal) deviation from mean, i.e. between (1 - dev)  mean and (1 + dev)  mean Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.plot.common._percentile",
"url":19,
"doc":"Maps the range between two percentiles of the data. Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.plot.common._percentile_sym_zero",
"url":19,
"doc":"Maps the range between two percentiles of the data, but ensuring symmetry about zero Arguments     - image : np array, 3D image data being shown as ax.imshow c_range_values : unknown (depends on user settings) See  qdmpy.plot.common.get_colormap_range ",
"func":1
},
{
"ref":"qdmpy.plot.field",
"url":20,
"doc":"This module holds functions for plotting fields. Functions     - -  qdmpy.plot.field.bnvs_and_dshifts -  qdmpy.plot.field.bfield -  qdmpy.plot.field.dshift_fit -  qdmpy.plot.field.field_residual -  qdmpy.plot.field.field_param -  qdmpy.plot.field.field_param_flattened -  qdmpy.plot.field.bfield_consistency -  qdmpy.plot.field.bfield_theta_phi "
},
{
"ref":"qdmpy.plot.field.bnvs_and_dshifts",
"url":20,
"doc":"Plots bnv arrays above dshift arrays. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames bnvs : list List of np arrays (2D) giving B_NV for each NV family/orientation. If num_peaks is odd, the bnv is given as the shift of that peak, and the dshifts is left as np.nans. dshifts : list List of np arrays (2D) giving the D (~DFS) of each NV family/orientation. If num_peaks is odd, the bnv is given as the shift of that peak, and the dshifts is left as np.nans. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.bfield",
"url":20,
"doc":"Plot Bxyz if available as keys in 'field_params'. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.dshift_fit",
"url":20,
"doc":"Plot dshift (fit) if available as keys in 'field_params'. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.field_residual",
"url":20,
"doc":"Plot residual if available as keys in 'field_params' (as 'residual_field'). Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.field_param",
"url":20,
"doc":"plot a given field param. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames. param_name : str Name of specific param to plot (e.g. \"Bx\" etc.). field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. c_map : str, default: None colormap to use overrides c_map_type and c_range_type. c_map_type : str, default: \"param_images\" colormap type to search options (options[\"colormaps\"][c_map_type]) for. c_map_range : str, default: \"percentile\" colormap range option (see  qdmpy.plot.common.get_colormap_range ) to use. c_range_vals : number or list, default: (5, 95) passed with c_map_range to get_colormap_range cbar_label : str, default: label to chuck on ye olde colorbar (z-axis label). Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.field_param_flattened",
"url":20,
"doc":"plot a field param flattened vs pixel number. Sigmas are optionally utilised as errobars. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames. param_name : str Name of key in field_params (and sigmas) to plot. field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. sigmas : dict, default: None Dictionary, key: param_keys, val: image (2D) of (field) sigma values across FOV. plot_sigmas : bool, default: True If true & sigmas is not None, sigmas are used as errorbars. plot_bounds : bool, default: True If True, (field) fit bound is annotated as horizontal dashed lines. plot_guess: bool, default: True If True, (field) fit guess is annotates as horizontal dashed line. y_label : str, default:  Label to chuck on y axis. errorevery : int, default: 1 Plot errorbar every 'errorevery' data point (so it doesn't get too messy). Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.strain",
"url":20,
"doc":"",
"func":1
},
{
"ref":"qdmpy.plot.field.efield",
"url":20,
"doc":"",
"func":1
},
{
"ref":"qdmpy.plot.field.bfield_consistency",
"url":20,
"doc":"plot bfield vs bfield_recon Parameters      options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames. field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.field.bfield_theta_phi",
"url":20,
"doc":"Plots B_theta_phi if found in field_params (the vector field projected onto some direction). Parameters      options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames. field_params : dict Dictionary, key: param_keys, val: image (2D) of (field) param values across FOV. c_map : str, default: None colormap to use overrides c_map_type and c_range_type. c_map_type : str, default: \"bfield_images\" colormap type to search options (options[\"colormaps\"][c_map_type]) for. c_map_range : str, default: \"percentile\" colormap range option (see  qdmpy.plot.common.get_colormap_range ) to use. c_range_vals : number or list, default: (5, 95) passed with c_map_range to get_colormap_range. c_bar_label : str, default: label to chuck on ye olde colorbar (z-axis label). Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl",
"url":21,
"doc":"This module holds functions for plotting initial processing images and fit results. Functions     - -  qdmpy.plot.pl.roi_pl_image -  qdmpy.plot.pl.aoi_pl_image -  qdmpy.plot.pl.roi_avg_fits -  qdmpy.plot.pl.aoi_spectra -  qdmpy.plot.pl.aoi_spectra_fit -  qdmpy.plot.pl.pl_param_image -  qdmpy.plot.pl.pl_param_images -  qdmpy.plot.pl.pl_params_flattened -  qdmpy.plot.pl.other_measurements -  qdmpy.plot.pl._add_patch_rect -  qdmpy.plot.pl._annotate_roi_image -  qdmpy.plot.pl._annotate_aoi_image "
},
{
"ref":"qdmpy.plot.pl.roi_pl_image",
"url":21,
"doc":"Plots full pl image with ROI region annotated. Arguments     - options : dict Generic options dict holding all the user options. pl_image : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned but not cut down to ROI. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.aoi_pl_image",
"url":21,
"doc":"Plots pl image cut down to ROI, with annotated AOI regions. Arguments     - options : dict Generic options dict holding all the user options. pl_image_roi : np array, 2D Summed counts across sweep_value (affine) axis (i.e. 0th axis). Reshaped, rebinned and cut down to ROI. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.roi_avg_fits",
"url":21,
"doc":"Plots fit of spectrum averaged across ROI, as well as corresponding residual values. Arguments     - options : dict Generic options dict holding all the user options. backend_roi_results_lst : list of tuples Format: (fit_backend,  qdmpy.pl.common.ROIAvgFitResult objects), for each fit_backend Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.aoi_spectra",
"url":21,
"doc":"Plots spectra from each AOI, as well as subtraction and division norms. Arguments     - options : dict Generic options dict holding all the user options. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] ref : np array, 3D Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] sweep_list : list List of sweep parameter values (with removed unwanted sweeps at start/end) Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.aoi_spectra_fit",
"url":21,
"doc":"Plots sig and ref spectra, sub and div normalisation and fit for the ROI average, a single pixel, and each of the AOIs. All stacked on top of each other for comparison. The ROI average fit is plot against the fit of all of the others for comparison. Note here and elsewhere the single pixel check is the first element of the AOI array. NOTE this could be faster if we passed in sig_norm as well (backwards-compat. issues tho) Arguments     - options : dict Generic options dict holding all the user options. sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] ref : np array, 3D Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] sweep_list : list List of sweep parameter values (with removed unwanted sweeps at start/end) fit_result_collection_lst : list List of  qdmpy.pl.common.FitResultCollection objects (one for each fit_backend) holding ROI, AOI fit results fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.pl_param_image",
"url":21,
"doc":"Plots an image corresponding to a single parameter in pixel_fit_params. Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. pixel_fit_params : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. param_name : str Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'. Optional arguments          param_number : int Which version of the parameter you want. I.e. there might be 8 independent parameters in the fit model called 'pos', each labeled 'pos_0', 'pos_1' etc. Default: 0. errorplot : bool Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so ylabel & save names are changed accordingly. Can't be True if param_name='residual'. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.pl_param_images",
"url":21,
"doc":"Plots images for all independent versions of a single parameter type in pixel_fit_params. Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. pixel_fit_params : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. param_name : str Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'. errorplot : bool Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so ylabel & save names are changed accordingly. Can't be True if param_name='residual'. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.pl_params_flattened",
"url":21,
"doc":"Compare pixel fits against flattened pixels: initial guess vs ROI fit vs fit result. Arguments     - options : dict Generic options dict holding all the user options. fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. pixel_fit_params : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. roi_avg_fit_result  qdmpy.pl.common.ROIAvgFitResult object. param_name : str Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'. plot_bounds : bool Defaults to True, add fit bounds/constraints to plot. (Does nothing for residual plots) plot_sigmas : bool Defaults to True, add error bars (sigma) to plot. (Does nothing for residual plots) Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.pl.other_measurements",
"url":21,
"doc":"Plot any other tsv/csv datasets (at base_path + s for s in options[\"other_measurement_suffixes\"]). Assumes first column is some form of ind. dataset",
"func":1
},
{
"ref":"qdmpy.plot.pl._add_patch_rect",
"url":21,
"doc":"Adds a rectangular annotation onto ax. Arguments     - ax : matplotlib Axis object rect_corner_x : int Location of top left corner of area you want to annotate, x component. rect_corner_y : int Location of top left corner of area you want to annotate, y component. size_x : int Size of area along x (horizontal axis) you want to annotate. size_y : int Size of area along y (vertical) axis you want to annotate. Optional arguments          label : str Text to label annotated square with. Color is defined by edgecolor. Default: None. edgecolor : str Color of label and edge of annotation. Default: \"b\".",
"func":1
},
{
"ref":"qdmpy.plot.pl._annotate_roi_image",
"url":21,
"doc":"Annotates ROI onto a given Axis object. Generally used on a pl image.",
"func":1
},
{
"ref":"qdmpy.plot.pl._annotate_aoi_image",
"url":21,
"doc":"Annotates AOI onto a given Axis object. Generally used on pl image.",
"func":1
},
{
"ref":"qdmpy.plot.source",
"url":22,
"doc":"This module holds functions for plotting source (fields). Functions     - -  qdmpy.plot.source.source_param -  qdmpy.plot.source.current -  qdmpy.plot.source.current_stream -  qdmpy.plot.source.magnetization -  qdmpy.plot.source.divperp_j "
},
{
"ref":"qdmpy.plot.source.source_param",
"url":22,
"doc":"plot a given source param. Arguments     - options : dict Generic options dict holding all the user options. name : str Name of these results (e.g. sig, ref, sig sub ref etc.) for titles and filenames source_params : dict Dictionary, key: param_keys, val: image (2D) of (source field) param values across FOV. c_map : str, default: None colormap to use overrides c_map_type and c_range_type c_map_type : str, default: \"source_images\" colormap type to search options (options[\"colormaps\"][c_map_type]) for c_map_range : str, default: \"percentile\" colormap range option (see  qdmpy.plot.common.get_colormap_range ) to use c_range_vals : number or list, default: (1, 99) passed with c_map_range to get_colormap_range c_bar_label : str, default: label to chuck on ye olde colorbar (z-axis label). Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.source.current",
"url":22,
"doc":"Plots current (Jx, Jy, Jnorm). Optionally plot background subtracted. Arguments     - options : dict Generic options dict holding all the user options. source_params : dict Dictionary, key: param_keys, val: image (2D) of (source field) param values across FOV. plot_bgrounds : {bool}, default: True Plot background images Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.source.current_stream",
"url":22,
"doc":"Plot current density as a streamplot. Arguments     - options : dict Generic options dict holding all the user options. source_params : dict Dictionary, key: param_keys, val: image (2D) of (source field) param values across FOV. background_image : numpy array or None, default: None If not None, must be pl image of ROI (or other background image), to plot behind streams Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.source.magnetization",
"url":22,
"doc":"Plots magnetization. Optionally plot background subtracted. Arguments     - options : dict Generic options dict holding all the user options. source_params : dict Dictionary, key: param_keys, val: image (2D) of (source field) param values across FOV. plot_bgrounds : {bool}, default: True Plot background images Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.plot.source.divperp_j",
"url":22,
"doc":"plot perpindicular divergence of J, i.e. in-plane divergence (dJ/dx + dJ/dy). Parameters      options : dict Generic options dict holding all the user options. source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"qdmpy.shared",
"url":23,
"doc":"Sub-package containing shared modules for the rest of qdmpy. Modules    - -  qdmpy.shared.fourier -  qdmpy.shared.geom -  qdmpy.shared.itool -  qdmpy.shared.json2dict -  qdmpy.shared.misc -  qdmpy.shared.polygon "
},
{
"ref":"qdmpy.shared.fourier",
"url":24,
"doc":"Shared FFTW tooling. Functions     - -  qdmpy.shared.fourier.unpad_image -  qdmpy.shared.fourier.pad_image -  qdmpy.shared.fourier.define_k_vectors -  qdmpy.shared.fourier.set_naninf_to_zero -  qdmpy.shared.fourier.hanning_filter_kspace Constants     - -  qdmpy.shared.fourier.MAG_UNIT_CONV -  qdmpy.shared.fourier.MU_0 "
},
{
"ref":"qdmpy.shared.fourier.MAG_UNIT_CONV",
"url":24,
"doc":"Convert unit for magnetization to something more helpful. SI unit measured: Amps: A [for 2D magnetization, A/m for 3D] More useful: Bohr magnetons per nanometre squared: mu_B nm^-2 mu_B -> 9.274 010 e-24 A m^+2 or J/T m^2 -> 1e+18 nm^2 Measure x amps = x A def mu_B = 9.2_ in units of A m^2 => x A = x (1 / 9.2_) in units of mu_B/m^2 => x A = x (1e-18/9.2_) in units of mu_B/nm^2"
},
{
"ref":"qdmpy.shared.fourier.MU_0",
"url":24,
"doc":"Vacuum permeability"
},
{
"ref":"qdmpy.shared.fourier.unpad_image",
"url":24,
"doc":"undo a padding defined by  QDMPy.fourier._shared.pad_image (it returns the padder list)",
"func":1
},
{
"ref":"qdmpy.shared.fourier.pad_image",
"url":24,
"doc":"pad_mode -> see np.pad pad_factor -> either side of image",
"func":1
},
{
"ref":"qdmpy.shared.fourier.define_k_vectors",
"url":24,
"doc":"Get scaled k vectors (as meshgrid) for fft. Arguments      shape : list Shape of fft array to get k vectors for. pixel_size : float Pixel size, e.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. Returns    - ky, kx, k : np array Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )",
"func":1
},
{
"ref":"qdmpy.shared.fourier.set_naninf_to_zero",
"url":24,
"doc":"replaces NaNs and infs with zero",
"func":1
},
{
"ref":"qdmpy.shared.fourier.hanning_filter_kspace",
"url":24,
"doc":"Computes a hanning image filter with both low and high pass filters. Arguments     - k : np array Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 ) do_filt : bool Do a hanning filter? hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. standoff : float Distance NV layer  Sample. Returns    - img_filter : (2d array, float) bandpass filter to remove artifacts in the FFT process.",
"func":1
},
{
"ref":"qdmpy.shared.fourier.define_magnetization_transformation",
"url":24,
"doc":"M => b fourier-space transformation. Parameters      ky, kx, k : np array Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 ) standoff : float Distance NV layer  Sample Returns    - d_matrix : np array Transformation such that B = d_matrix  m. E.g. for z magnetized sample: m_to_bnv = ( unv[0]  d_matrix[2, 0,  ] + unv[1]  d_matrix[2, 1,  ] + unv[2]  d_matrix[2, 2,  ] ) -> First index '2' is for z magnetization (see m_from_bxy for in-plane mag process), the second index is for the (bnv etc.) bfield axis (0:x, 1:y, 2:z), and the last index iterates through the k values/vectors. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.shared.fourier.define_current_transform",
"url":24,
"doc":"b => J fourier-space transformation. Arguments     - u_proj : array-like Shape: 3, the direction the magnetic field was measured in (projected onto). ky, kx, k : np arrays Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 ) standoff : float or None, default : None Distance NV layer  sample Returns    - b_to_jx, b_to_jy : np arrays (2D) See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.shared.geom",
"url":27,
"doc":"This module holds tools for determining the geometry of the NV-bias field system etc., required for retrieving/reconstructing vector fields. Functions     - -  qdmpy.shared.geom.get_unvs -  qdmpy.shared.geom.get_unv_frames Constants     - -  qdmpy.shared.geom.NV_AXES_100_110 -  qdmpy.shared.geom.NV_AXES_100_100 -  qdmpy.shared.geom.NV_AXES_111 "
},
{
"ref":"qdmpy.shared.geom.NV_AXES_100_110",
"url":27,
"doc":" top face oriented,  edge face oriented diamond (CVD). NV orientations (unit vectors) relative to lab frame. Assuming diamond is square to lab frame: first 3 numbers: orientation of top face of diamond, e.g.  second 3 numbers: orientation of edges of diamond, e.g.  CVD Diamonds are usually  ,  . HPHT usually  ,  . ![](https: i.imgur.com/Rudnzyo.png) Purple plane corresponds to top (or bottom) face of diamond, orange planes correspond to edge faces."
},
{
"ref":"qdmpy.shared.geom.NV_AXES_100_100",
"url":27,
"doc":" top face oriented,  edge face oriented diamond (HPHT). NV orientations (unit vectors) relative to lab frame. Assuming diamond is square to lab frame: first 3 numbers: orientation of top face of diamond, e.g.  second 3 numbers: orientation of edges of diamond, e.g.  CVD Diamonds are usually  ,  . HPHT usually  ,  . ![](https: i.imgur.com/cpErjAH.png) Purple plane: top face of diamond, orange plane: edge faces."
},
{
"ref":"qdmpy.shared.geom.NV_AXES_111",
"url":27,
"doc":" top face oriented. NV orientations (unit vectors) relative to lab frame. Only the first nv can be oriented in general. This constant is defined as a convenience for single-bnv  measurements.  diamonds have an NV family oriented in z, i.e. perpindicular to the diamond surface."
},
{
"ref":"qdmpy.shared.geom.get_unvs",
"url":27,
"doc":"Returns orientation (relative to lab frame) of NVs. Shape: (4,3) regardless of sample. Arguments     - options : dict Generic options dict holding all the user options. Returns    - unvs : np array Shape: (4,3). Equivalent to uNV_Z for each NV. (Sorted largest to smallest Bnv)",
"func":1
},
{
"ref":"qdmpy.shared.geom.get_unv_frames",
"url":27,
"doc":"Returns array representing each NV reference frame. I.e. each index is a 2D array: [uNV_X, uNV_Y, uNV_Z] representing the unit vectors for that NV reference frame, in the lab frame. Arguments     - options : dict Generic options dict holding all the user options. Returns    - unv_frames : np array [ [uNV1_X, uNV1_Y, uNV1_Z], [uNV2_X, uNV2_Y, uNV2_Z],  .]",
"func":1
},
{
"ref":"qdmpy.shared.itool",
"url":26,
"doc":"This module holds misc image tooling. Functions     - -  qdmpy.shared.itool.mask_polygons -  qdmpy.shared.itool.get_im_filtered -  qdmpy.shared.itool._get_im_filtered_gaussian -  qdmpy.shared.itool._zero_background -  qdmpy.shared.itool._equation_plane -  qdmpy.shared.itool._points_to_params -  qdmpy.shared.itool._three_point_background -  qdmpy.shared.itool._mean_background -  qdmpy.shared.itool._residual_poly -  qdmpy.shared.itool._poly_background -  qdmpy.shared.itool._gaussian -  qdmpy.shared.itool._moments -  qdmpy.shared.itool._residual_gaussian -  qdmpy.shared.itool._gaussian_background -  qdmpy.shared.itool._interpolated_background -  qdmpy.shared.itool._filtered_background -  qdmpy.shared.itool.get_background -  qdmpy.shared.itool.mu_sigma_inside_polygons "
},
{
"ref":"qdmpy.shared.itool.mask_polygons",
"url":26,
"doc":"Mask image for the given polygon regions. Arguments     - image : 2D array-like Image array to mask. polygons : list, optional List of  qdmpy.shared.polygon.Polygon objects. (the default is None, where image is returned with no mask) invert_mask : bool, optional Invert mask such that background is masked, not polygons (i.e. polygons will be operated on if array is passed to np.mean instead of background). (the default is False) Returns    - masked_im : np.ma.MaskedArray image, now masked",
"func":1
},
{
"ref":"qdmpy.shared.itool.get_background",
"url":26,
"doc":"Returns a background for given image, via chosen method. Methods available: - \"fix_zero\" - Fix background to be a constant offset (z value) - params required in method_params_dict: \"zero\" an int/float, defining the constant offset of the background - \"three_point\" - Calculate plane background with linear algebra from three [x,y] lateral positions given - params required in method_params_dict: - \"points\" a len-3 iterable containing [x, y] points - \"mean\" - background calculated from mean of image - no params required - \"poly\" - background calculated from polynomial fit to image. - params required in method_params_dict: - \"order\": an int, the 'order' polynomial to fit. (e.g. 1 = plane). - \"gaussian\" - background calculated from _gaussian fit to image. - no params required - \"interpolate\" - Background defined by the dataset smoothed via a sigma-_gaussian filtering, and method-interpolation over masked (polygon) regions. - params required in method_params_dict: - \"interp_method\": nearest, linear, cubic. - \"sigma\": sigma passed to _gaussian filter (see scipy.ndimage._gaussian_filter) which is utilized on the background before interpolating - \"gaussian_filter\" - background calculated from image filtered with a _gaussian filter. - params required in method_params_dict: - \"sigma\": sigma passed to _gaussian filter (see scipy.ndimage._gaussian_filter) - \"gaussian_then_poly\" - runs gaussian then poly subtraction - params required in method_params_dict: - \"order\": an int, the 'order' polynomial to fit. (e.g. 1 = plane). polygon utilization: - if method is not interpolate, the image is masked where the polygons are and the background is calculated without these regions - if the method is interpolate, these regions are interpolated over (and the rest of the image, _gaussian smoothed, is 'background'). Arguments     - image : 2D array-like image to get backgrond of method : str Method to use, available options above  method_params_dict : dict Key-value pairs passed onto each background backend. Required params given above. polygons : list, optional list of  qdmpy.shared.polygon.Polygon objects. (the default is None, in which case the polygon feature is not used) Returns    - im_bground : ndarray 2D numpy array, representing the 'background' of image.",
"func":1
},
{
"ref":"qdmpy.shared.itool.mu_sigma_inside_polygons",
"url":26,
"doc":"returns (mean, standard_deviation) for image, only _within_ polygon areas.",
"func":1
},
{
"ref":"qdmpy.shared.itool.get_im_filtered",
"url":26,
"doc":"Wrapped over other filters defined in  qdmpy.shared.polygon.Polygon_filter'. Current filters defined: - filter_type = gaussian,  qdmpy.shared.itool._get_im_filtered_gaussian ",
"func":1
},
{
"ref":"qdmpy.shared.itool._get_im_filtered_gaussian",
"url":26,
"doc":"Returns image filtered through scipy.ndimage.gaussian_filter with parameter 'sigma'.",
"func":1
},
{
"ref":"qdmpy.shared.itool._zero_background",
"url":26,
"doc":"Background defined by z level of 'zero'",
"func":1
},
{
"ref":"qdmpy.shared.itool._equation_plane",
"url":26,
"doc":"params: [a, b, c, d] s.t. d = a y + b x + c z so z = (1/c)  (d - (ay + bx -> return this.",
"func":1
},
{
"ref":"qdmpy.shared.itool._points_to_params",
"url":26,
"doc":"http: pi.math.cornell.edu/~froh/231f08e1a.pdf points: iterable of 3 iterables: [x, y, z] returns a,b,c,d parameters (see _equation_plane)",
"func":1
},
{
"ref":"qdmpy.shared.itool._three_point_background",
"url":26,
"doc":"points: len 3 iterable of len 2 iterables:  x1, y1], [x2, y2], [x3, y3 https: stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points https: www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/",
"func":1
},
{
"ref":"qdmpy.shared.itool._mean_background",
"url":26,
"doc":"Background defined by mean of image.",
"func":1
},
{
"ref":"qdmpy.shared.itool._residual_poly",
"url":26,
"doc":"z = image data, order = highest polynomial order to go to y, x: index meshgrids",
"func":1
},
{
"ref":"qdmpy.shared.itool._poly_background",
"url":26,
"doc":"Background defined by a polynomial fit up to order 'order'.",
"func":1
},
{
"ref":"qdmpy.shared.itool._gaussian",
"url":26,
"doc":"Simple Gaussian function, height, center_y, center_x, width_y, width_x = p .",
"func":1
},
{
"ref":"qdmpy.shared.itool._moments",
"url":26,
"doc":"Calculate _moments of image (get initial guesses on _gaussian function)",
"func":1
},
{
"ref":"qdmpy.shared.itool._residual_gaussian",
"url":26,
"doc":"Residual of data with a _gaussian model.",
"func":1
},
{
"ref":"qdmpy.shared.itool._gaussian_background",
"url":26,
"doc":"Background defined by a Gaussian function.",
"func":1
},
{
"ref":"qdmpy.shared.itool._interpolated_background",
"url":26,
"doc":"Background defined by the dataset smoothed via a sigma-_gaussian filtering, and method-interpolation over masked (polygon) regions. method available: nearest, linear, cubic.",
"func":1
},
{
"ref":"qdmpy.shared.itool._filtered_background",
"url":26,
"doc":"Background defined by a filter_type-filtering of the image. Passed to  qdmpy.shared.itool.get_background",
"func":1
},
{
"ref":"qdmpy.shared.json2dict",
"url":27,
"doc":"json2dict; functions for loading json files to dicts and the inverse. Functions     - -  qdmpy.shared.json2dict.json_to_dict -  qdmpy.shared.json2dict.dict_to_json -  qdmpy.shared.json2dict.dict_to_json_str -  qdmpy.shared.json2dict._prettyjson -  qdmpy.shared.json2dict._getsubitems -  qdmpy.shared.json2dict._basictype2str -  qdmpy.shared.json2dict._indentitems -  qdmpy.shared.json2dict._json_remove_comments -  qdmpy.shared.json2dict.failfloat -  qdmpy.shared.json2dict._defaultdict_from_d -  qdmpy.shared.json2dict.recursive_dict_update "
},
{
"ref":"qdmpy.shared.json2dict.json_to_dict",
"url":27,
"doc":"read the json file at filepath into a dict",
"func":1
},
{
"ref":"qdmpy.shared.json2dict.dict_to_json",
"url":27,
"doc":"save the dict as a json in a pretty way",
"func":1
},
{
"ref":"qdmpy.shared.json2dict.dict_to_json_str",
"url":27,
"doc":"Copy of  qdmpy.shared.json2dict._prettyjson ",
"func":1
},
{
"ref":"qdmpy.shared.json2dict._prettyjson",
"url":27,
"doc":"Renders JSON content with indentation and line splits/concatenations to fit maxlinelength. Only dicts, lists and basic types are supported. Now supports np.ndarrays.  ",
"func":1
},
{
"ref":"qdmpy.shared.json2dict._getsubitems",
"url":27,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.json2dict._basictype2str",
"url":27,
"doc":"This is a filter on objects that get sent to the json. Some types can't be stored literally in json files, so we can adjust for that here.",
"func":1
},
{
"ref":"qdmpy.shared.json2dict._indentitems",
"url":27,
"doc":"Recursively traverses the list of json lines, adds indentation based on the current depth",
"func":1
},
{
"ref":"qdmpy.shared.json2dict._json_remove_comments",
"url":27,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.json2dict.failfloat",
"url":27,
"doc":"Used in particular for reading the metadata to convert all numbers into floats and leave strings as strings.",
"func":1
},
{
"ref":"qdmpy.shared.json2dict._defaultdict_from_d",
"url":27,
"doc":"converts d to a defaultdict, with default value of None for all keys",
"func":1
},
{
"ref":"qdmpy.shared.json2dict.recursive_dict_update",
"url":27,
"doc":"Recursively updates to_be_updated_dict with values from updating_dict (to all dict depths).",
"func":1
},
{
"ref":"qdmpy.shared.linecut",
"url":30,
"doc":"This module contains . FIXME TODO docs, once code settles"
},
{
"ref":"qdmpy.shared.linecut.bulk_vert_linecut_vs_position",
"url":30,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.linecut.vert_linecut_vs_position",
"url":30,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.linecut.BulkLinecutWidget",
"url":30,
"doc":"How to use      import matplotlib.pyplot as plt import numpy as np from qdmpy.shared.linecut import BulkLinecutWidget path = \" \" times = [0.325, 1, 5, 10, 20, 21, 22, 25, 30, 40] paths = [f\"{path}/{t}.txt\" for t in times] images = [np.loadtxt(p) for p in paths] selector_image = images[4] fig, axs = plt.subplots(ncols=3, figsize=(12, 6 axs[0].imshow(selector_image)  (data can be nans if you want an empty selector) selector = BulkLinecutWidget( axs, images, times) plt.show() selector.disconnect(path=\"/home/samsc/share/result.json\")"
},
{
"ref":"qdmpy.shared.linecut.BulkLinecutWidget.ondraw",
"url":30,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.linecut.BulkLinecutWidget.onselect",
"url":30,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.linecut.BulkLinecutWidget.disconnect",
"url":30,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.linecut.LinecutSelectionWidget",
"url":30,
"doc":"How to Use      fig, axs = plt.subplots(ncols=2) axs[0].imshow(data)  (data may be nans if you want empty selector) selector = LinecutSelectionWidget(axs[0], axs[1],  .) plt.show() selector.disconnect()"
},
{
"ref":"qdmpy.shared.linecut.LinecutSelectionWidget.ondraw",
"url":30,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.linecut.LinecutSelectionWidget.onselect",
"url":30,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.linecut.LinecutSelectionWidget.disconnect",
"url":30,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.misc",
"url":28,
"doc":"Shared methods that I couldn't find a nice place to put"
},
{
"ref":"qdmpy.shared.misc.define_aois",
"url":28,
"doc":"Defines areas of interest (aois). Returns list of aois that can be used to directly index into image array, e.g.: sig_aoi = sig[:, aoi[0], aoi[1 . Arguments     - options : dict Generic options dict holding all the user options. Returns    - aois : list List of aoi regions. Much like roi object, these are a length-2 list of np meshgrids that can be used to directly index into image to provide a view into just the aoi part of the image. E.g. sig_aoi = sig[:, aoi[0], aoi[1 . Returns a list as in general we have more than one area of interest. I.e. sig_aoi_1 = sig[:, aois[1][0], aois[1][1 ",
"func":1
},
{
"ref":"qdmpy.shared.misc.define_roi",
"url":28,
"doc":"Defines meshgrids that can be used to slice image into smaller region of interest (roi). Arguments     - options : dict Generic options dict holding all the user options. full_size_w : int Width of image (after rebin, before roi cut). full_size_h : int Height of image (after rebin, before roi cut). Returns    - roi : length 2 list of np meshgrids Defines an roi that can be applied to the 3D image through direct indexing. E.g. sig_roi = sig[:, roi[0], roi[1 ",
"func":1
},
{
"ref":"qdmpy.shared.polygon",
"url":29,
"doc":"This module holds the Polygon class: a class to compute if a point lies inside/outside/on-side of a polygon. Also defined is a function (polygon_gui) that can be called to select a polygon region on an image. Polygon-GUI      - Function to select polygons on an image. Ensure you have the required gui backends for matplotlib. Best ran seperately/not within jupyter. E.g. open python REpl (python at cmd), 'import qdmpy.itool', then run qdmpy.shared.polygon.Polygonpolygon_gui() & follow the prompts. An optional array (i.e. the image used to define regions) can be passed to polygon_gui. The output json path can then be specified in the usual way (there's an option called 'polygon_nodes_path') to utilize these regions in the main processing code. Polygon    - This is a Python 3 implementation of the Sloan's improved version of the Nordbeck and Rystedt algorithm, published in the paper: SLOAN, S.W. (1985): A point-in-polygon program. Adv. Eng. Software, Vol 7, No. 1, pp 45-47. This class has 1 method (is_inside) that returns the minimum distance to the nearest point of the polygon: If is_inside  0 then point is inside the polygon. Sam Scholten copied from: http: code.activestate.com/recipes/578381-a-point-in-polygon-program-sw-sloan-algorithm/ -> swapped x & y args order (etc.) for image use. Classes    - -  qdmpy.shared.polygon.Polygon Functions     - -  qdmpy.shared.polygon.polygon_selector -  qdmpy.shared.polygon.polygon_gui -  qdmpy.shared.polygon.Polygon -  qdmpy.shared.polygon._tri_2area_det "
},
{
"ref":"qdmpy.shared.polygon._tri_2area_det",
"url":29,
"doc":"Compute twice the area of the triangle defined by points with using determinant formula. Arguments     - yvert : array-like A vector of nodal x-coords (all unique). xvert : array-like A vector of nodal y-coords (all unique). Returns    - Twice the area of the triangle defined by the points. Notes:asfarray _tri_2area_det is positive if asfarraypoints define polygon in anticlockwise order. _tri_2area_det is negative if asfarraypoints define polygon in clockwise order. _tri_2area_det is zero if at least two of the points are coincident or if all points are collinear.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.Polygon",
"url":29,
"doc":"Polygon object. Arguments     - y : array-like A sequence of nodal y-coords (all unique). x : array-like A sequence of nodal x-coords (all unique)."
},
{
"ref":"qdmpy.shared.polygon.Polygon.get_nodes",
"url":29,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.polygon.Polygon.is_inside",
"url":29,
"doc":"Check if point is inside a general polygon. Arguments     - ypoint : array-like The y-coord of the point to be tested. xpoint : array-like The x-coords of the point to be tested. smalld : float A small float number. ypoint and xpoint could be scalars or array-like sequences. Returns    - mindst : scalar The distance from the point to the nearest point of the polygon. If mindst  0 then point is inside the polygon. Notes   - An improved version of the algorithm of Nordbeck and Rydstedt. REF: SLOAN, S.W. (1985): A point-in-polygon program. Adv. Eng. Software, Vol 7, No. 1, pp 45-47.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.polygon_selector",
"url":29,
"doc":"Generates mpl (qt) gui for selecting a polygon. Arguments     - numpy_txt_file_path : path Path to (numpy) .txt file to load as image. json_output_path : str or path-like, default=\"~/poly.json\" Path to put output json, defaults to home/poly.json. json_input_path : str or path-like, default=None Loads previous polygons at this path for editing. mean_plus_minus : float, default=None Plot image with color scaled to mean +- this number. strict_range: length 2 list, default=None Plot image with color scaled between these values. Precedence over mean_plus_minus. print_help : bool, default=False View this message.  kwargs : dict Other keyword arguments to pass to plotters. Currently implemented: cmap : string Passed to imshow. lineprops : dict Passed to PolygonSelectionWidget. markerprops : dict Passed to PolygonSelectionWidget. GUI help     In the mpl gui, select points to draw polygons. Press 'enter' to continue in the program. Press the 'esc' key to reset the current polygon Hold 'shift' to move all of the vertices (from all polygons) 'ctrl' to move a single vertex in the current polygon 'alt' to start a new polygon (and finalise the current one) 'del' to clear all lines from the graphic (thus deleting all polygons).",
"func":1
},
{
"ref":"qdmpy.shared.polygon.polygon_gui",
"url":29,
"doc":"Load gui to select polygon regions. Follow the prompts. Arguments     - image : 2D array-like, optional image to use for select polygons on. (the default is None, forces user to specify path in gui) Returns    - polygon_lst : list list of polygons (each defined by a list of nodes [y, x]) Also saves polygon_lst into a json at a given path (in the gui).",
"func":1
},
{
"ref":"qdmpy.shared.polygon.Widget",
"url":29,
"doc":"Abstract base class for GUI neutral widgets"
},
{
"ref":"qdmpy.shared.polygon.Widget.drawon",
"url":29,
"doc":""
},
{
"ref":"qdmpy.shared.polygon.Widget.eventson",
"url":29,
"doc":""
},
{
"ref":"qdmpy.shared.polygon.Widget.set_active",
"url":29,
"doc":"Set whether the widget is active.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.Widget.get_active",
"url":29,
"doc":"Get whether the widget is active.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.Widget.active",
"url":29,
"doc":"Is the widget active?"
},
{
"ref":"qdmpy.shared.polygon.Widget.ignore",
"url":29,
"doc":"Return True if event should be ignored. This method (or a version of it) should be called at the beginning of any event callback.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.AxesWidget",
"url":29,
"doc":"Widget that is connected to a single :class: ~matplotlib.axes.Axes . To guarantee that the widget remains responsive and not garbage-collected, a reference to the object should be maintained by the user. This is necessary because the callback registry maintains only weak-refs to the functions, which are member functions of the widget. If there are no references to the widget object it may be garbage collected which will disconnect the callbacks. Attributes:  ax : :class: ~matplotlib.axes.Axes The parent axes for the widget  canvas : :class: ~matplotlib.backend_bases.FigureCanvasBase subclass The parent figure canvas for the widget.  active : bool If False, the widget does not respond to events."
},
{
"ref":"qdmpy.shared.polygon.AxesWidget.connect_event",
"url":29,
"doc":"Connect callback with an event. This should be used in lieu of  figure.canvas.mpl_connect since this function stores callback ids for later clean up.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.AxesWidget.disconnect_events",
"url":29,
"doc":"Disconnect all events created by this widget.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.AxesWidget.set_active",
"url":29,
"doc":"Set whether the widget is active.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.AxesWidget.get_active",
"url":29,
"doc":"Get whether the widget is active.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.AxesWidget.active",
"url":29,
"doc":"Is the widget active?"
},
{
"ref":"qdmpy.shared.polygon.AxesWidget.ignore",
"url":29,
"doc":"Return True if event should be ignored. This method (or a version of it) should be called at the beginning of any event callback.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.ToolHandles",
"url":29,
"doc":"Control handles for canvas tools. Arguments     - ax : :class: matplotlib.axes.Axes Matplotlib axes where tool handles are displayed. x, y : 1D arrays Coordinates of control handles. marker : str Shape of marker used to display handle. See  matplotlib.pyplot.plot . marker_props : dict Additional marker properties. See :class: matplotlib.lines.Line2D ."
},
{
"ref":"qdmpy.shared.polygon.ToolHandles.x",
"url":29,
"doc":""
},
{
"ref":"qdmpy.shared.polygon.ToolHandles.y",
"url":29,
"doc":""
},
{
"ref":"qdmpy.shared.polygon.ToolHandles.set_data",
"url":29,
"doc":"Set x and y positions of handles",
"func":1
},
{
"ref":"qdmpy.shared.polygon.ToolHandles.set_visible",
"url":29,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.polygon.ToolHandles.set_animated",
"url":29,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.polygon.ToolHandles.closest",
"url":29,
"doc":"Return index and pixel distance to closest index.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector",
"url":29,
"doc":"OLD DOCSTRING Select a polygon region of an axes. Place vertices with each mouse click, and make the selection by completing the polygon (clicking on the first vertex). Hold the  ctrl key and click and drag a vertex to reposition it (the  ctrl key is not necessary if the polygon has already been completed). Hold the  shift key and click and drag anywhere in the axes to move all vertices. Press the  esc key to start a new polygon. For the selector to remain responsive you must keep a reference to it. Arguments     - ax : :class: ~matplotlib.axes.Axes The parent axes for the widget. onselect : function When a polygon is completed or modified after completion, the  onselect function is called and passed a list of the vertices as  (xdata, ydata) tuples. useblit : bool, optional lineprops : dict, optional The line for the sides of the polygon is drawn with the properties given by  lineprops . The default is  dict(color='k', linestyle='-', linewidth=2, alpha=0.5) . markerprops : dict, optional The markers for the vertices of the polygon are drawn with the properties given by  markerprops . The default is  dict(marker='o', markersize=7, mec='k', mfc='k', alpha=0.5) . vertex_select_radius : float, optional A vertex is selected (to complete the polygon or to move a vertex) if the mouse click is within  vertex_select_radius pixels of the vertex. The default radius is 15 pixels. Examples     :doc: /gallery/widgets/polygon_selector_demo "
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.onmove",
"url":29,
"doc":"Cursor move event handler and validator",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.draw_polygon",
"url":29,
"doc":"Redraw the polygon(s) based on the new vertex positions.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.verts",
"url":29,
"doc":"Get the polygon vertices. Returns    - list A list of the vertices of the polygon as  (xdata, ydata) tuples. for each polygon (A, B,  .) selected  [ [(Ax1, Ay1), (Ax2, Ay2)], [(Bx1, By1), (Bx2, By2)] ] "
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.xy_verts",
"url":29,
"doc":"Return list of the vertices for each polygon in the format: [ ( [Ax1, Ax2,  .], [Ay1, Ay2,  .] ), ( [Bx1, Bx2,  .], [By1, By2,  .] ) ]"
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.connect_event",
"url":29,
"doc":"Connect callback with an event. This should be used in lieu of  figure.canvas.mpl_connect since this function stores callback ids for later clean up.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.disconnect_events",
"url":29,
"doc":"Disconnect all events created by this widget.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.set_active",
"url":29,
"doc":"Set whether the widget is active.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.get_active",
"url":29,
"doc":"Get whether the widget is active.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.active",
"url":29,
"doc":"Is the widget active?"
},
{
"ref":"qdmpy.shared.polygon.PolygonSelector.ignore",
"url":29,
"doc":"Return True if event should be ignored. This method (or a version of it) should be called at the beginning of any event callback.",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelectionWidget",
"url":29,
"doc":"How to Use      selector = PolygonSelectionWidget(ax,  .) plt.show() selector.disconnect() polygon_lst = selector.get_polygon_lst()"
},
{
"ref":"qdmpy.shared.polygon.PolygonSelectionWidget.onselect",
"url":29,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelectionWidget.disconnect",
"url":29,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelectionWidget.get_polygons_lst",
"url":29,
"doc":"",
"func":1
},
{
"ref":"qdmpy.shared.polygon.PolygonSelectionWidget.load_nodes",
"url":29,
"doc":"",
"func":1
},
{
"ref":"qdmpy.source",
"url":30,
"doc":"Sub-package for inverting fields to calculate their source (field)."
},
{
"ref":"qdmpy.source.current",
"url":31,
"doc":"Implement inversion of magnetic field to current density. Functions     - -  qdmpy.source.current.get_divperp_j -  qdmpy.source.current.get_j_from_bxy -  qdmpy.source.current.get_j_from_bz -  qdmpy.source.current.get_j_from_bnv "
},
{
"ref":"qdmpy.source.current.get_divperp_j",
"url":31,
"doc":"jxy calculated -> perpindicular (in-plane) divergence of j Arguments     - jvec : list List of magnetic field components, e.g [jx_image, jy_image] pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. Returns    - jx_recon, jy_recon : np arrays (2D) The reconstructed j (source) field maps.  \\nabla \\times {\\bf J} = \\frac{\\partial {\\bf J} }{\\partial x} + \\frac{\\partial {\\bf J {\\partial y} + \\frac{\\partial {\\bf J {\\partial z}   \\nabla_{\\perp} \\times {\\bf J} = \\frac{\\partial {\\bf J} }{\\partial x} + \\frac{\\partial {\\bf J {\\partial y}  ",
"func":1
},
{
"ref":"qdmpy.source.current.get_j_from_bxy",
"url":31,
"doc":"Bxy measured -> Jxy via fourier methods. Arguments     - bfield : list List of magnetic field components, e.g [bx_image, by_image, bz_image] pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. do_hanning_filter : bool Do a hanning filter? hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. standoff : float Distance NV layer  Sample (in metres) nv_layer_thickness : float Thickness of NV layer (in metres) nvs_above_sample : bool True if NVs exist at higher z (in lab frame) than sample. Returns    - jx, jy : np arrays (2D) The calculated current density images, in A/m. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.current.get_j_from_bz",
"url":31,
"doc":"Bz measured -> Jxy via fourier methods. Arguments     - bfield : list List of magnetic field components, e.g [bx_image, by_image, bz_image] pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. do_hanning_filter : bool Do a hanning filter? hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. standoff : float Distance NV layer  Sample (in metres) nv_layer_thickness : float Thickness of NV layer (in metres) Returns    - jx, jy : np arrays (2D) The calculated current density images, in A/m. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.current.get_j_from_bnv",
"url":31,
"doc":"(Single) Bnv measured -> Jxy via fourier methods. Arguments     - single_bnv : np array Single bnv map (np 2D array). unv : array-like, 1D Shape: 3, the uNV_Z corresponding to the above bnv map. pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. do_hanning_filter : bool Do a hanning filter? hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. standoff : float Distance NV layer  Sample (in metres) nv_layer_thickness : float Thickness of NV layer (in metres) nvs_above_sample : bool True if NVs exist at higher z (in lab frame) than sample. Returns    - jx, jy : np arrays (2D) The calculated current density images, in A/m. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.interface",
"url":32,
"doc":"Interface for source sub-module. Functions     - -  qdmpy.source.interface.odmr_source_retrieval -  qdmpy.source.interface.get_current_density -  qdmpy.source.interface.get_magnetization -  qdmpy.source.interface.add_divperp_j -  qdmpy.source.interface.in_plane_mag_normalise "
},
{
"ref":"qdmpy.source.interface.odmr_source_retrieval",
"url":32,
"doc":"Calculates source field that created field measured by bnvs and field params. Arguments     - options : dict Generic options dict holding all the user options. bnvs : list List of bnv results (each a 2D image). sig_fit_params : dict Dictionary, key: param_keys, val: image (2D) of field values across FOV. Returns    - source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV. For methodology see D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.interface.get_current_density",
"url":32,
"doc":"Gets current density from bnvs and field_params according to options in options. Returned as a dict similar to fit_params/field_params etc. Arguments     - options : dict Generic options dict holding all the user options. bnvs : list List of bnv results (each a 2D image). sig_fit_params : dict Dictionary, key: param_keys, val: image (2D) of field values across FOV. Returns    - source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.interface.get_magnetization",
"url":32,
"doc":"Gets magnetization from bnvs and field_params according to options in options. Returned as a dict similar to fit_params/field_params etc. Arguments     - options : dict Generic options dict holding all the user options. bnvs : list List of bnv results (each a 2D image). sig_fit_params : dict Dictionary, key: param_keys, val: image (2D) of field values across FOV. Returns    - source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.interface.add_divperp_j",
"url":32,
"doc":"jxy -> Divperp J Divperp = divergence in x and y only (perpendicular to surface normal) Arguments     - options : dict Generic options dict holding all the user options (for the main/signal experiment). source_params : dict Dictionary, key: param_keys, val: image (2D) of source field values across FOV. Returns    - nothing (operates in place on field_params)  \\nabla \\times {\\bf J} = \\frac{\\partial {\\bf J} }{\\partial x} + \\frac{\\partial {\\bf J {\\partial y} + \\frac{\\partial {\\bf J {\\partial z}   \\nabla_{\\perp} \\times {\\bf J} = \\frac{\\partial {\\bf J} }{\\partial x} + \\frac{\\partial {\\bf J {\\partial y}  ",
"func":1
},
{
"ref":"qdmpy.source.interface.in_plane_mag_normalise",
"url":32,
"doc":"Normalise in-plane magnetization by taking average of mag near edge of image per line @ psi. The jist of this function was copied from D. Broadway's previous version of the code. Parameters      mag_image : np array 2D magnetization array as directly calculated. psi : float Assumed in-plane magnetization angle (deg) edge_pixels_used : int Number of pixels to use at edge of image to calculate average to subtract. Returns    - mag_image : np array in-plane magnetization image with line artifacts substracted.",
"func":1
},
{
"ref":"qdmpy.source.io",
"url":33,
"doc":"This module holds the tools for loading/saving source results. Functions     - -  qdmpy.source.io.prep_output_directories -  qdmpy.source.io.save_source_params "
},
{
"ref":"qdmpy.source.io.prep_output_directories",
"url":33,
"doc":"",
"func":1
},
{
"ref":"qdmpy.source.io.save_source_params",
"url":33,
"doc":"",
"func":1
},
{
"ref":"qdmpy.source.magnetization",
"url":34,
"doc":"Implement inversion of magnetic field to magnetization source Functions     - -  qdmpy.source.magnetization.get_m_from_bxy -  qdmpy.source.magnetization.get_m_from_bz -  qdmpy.source.magnetization.get_m_from_bnv "
},
{
"ref":"qdmpy.source.magnetization.get_m_from_bxy",
"url":34,
"doc":"Bxy measured -> M via fourier methods. Arguments     - bfield : list List of magnetic field components, e.g [bx_image, by_image, bz_image] pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. do_hanning_filter : bool Do a hanning filter? hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. standoff : float Distance NV layer  Sample (in metres) nv_layer_thickness : float Thickness of NV layer (in metres) Returns    - m : np array (2D) The calculated magnetization, in mu_B / nm^2 See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.magnetization.get_m_from_bz",
"url":34,
"doc":"Bz measured -> M via fourier methods. Arguments     - bfield : list List of magnetic field components, e.g [bx_image, by_image, bz_image] pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. do_hanning_filter : bool Do a hanning filter? hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. standoff : float Distance NV layer  Sample (in metres) nv_layer_thickness : float Thickness of NV layer (in metres) Returns    - m : np array (2D) The calculated magnetization, in mu_B / nm^2 See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.source.magnetization.get_m_from_bnv",
"url":34,
"doc":"(Single) Bnv measured -> M via fourier methods. Arguments     - single_bnv : np array Single bnv map (np 2D array). unv : array-like, 1D Shape: 3, the uNV_Z corresponding to the above bnv map. pad_mode : str Mode to use for fourier padding. See np.pad for options. pad_factor : int Factor to pad image on all sides. E.g. pad_factor = 2 => 2  image width on left and right and 2  image height above and below. pixel_size : float Size of pixel in bnv, the rebinned pixel size. E.g. options[\"system\"].get_raw_pixel_size(options)  options[\"total_bin\"]. k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. do_hanning_filter : bool Do a hanning filter? hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. standoff : float Distance NV layer  Sample (in metres) nv_layer_thickness : float Thickness of NV layer (in metres) nvs_above_sample : bool True if NVs exist at higher z (in lab frame) than sample. Returns    - m : np array (2D) The calculated magnetization, in mu_B / nm^2 See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"qdmpy.field.hamiltonian.AVAILABLE_HAMILTONIANS",
"url":36,
"doc":"Dictionary that defines hamiltonians available for use. Add any classes you define here so you can use them. You do not need to avoid overlapping parameter names as hamiltonian classes can not be used in combination."
},
{
"ref":"qdmpy.system",
"url":35,
"doc":"Sub-package for handling different physical QDM system requirements."
},
{
"ref":"qdmpy.system.systems",
"url":36,
"doc":"This sub-package holds classes and functions to define different systems. This sub-package allows users with different data saving/loading procedures to use this package. Also defined are variables such as raw_pixel_size which may be different for different systems at the same institution. Also defined are functions to handle the checking/cleaning of the json options file (and then dict) to ensure it is valid etc. _Make sure_ that any valid systems you define here are placed in the _SYSTEMS dict at the bottom of the file. Classes    - -  qdmpy.system.systems.System -  qdmpy.system.systems.UniMelb -  qdmpy.system.systems.Zyla -  qdmpy.system.systems.CryoWidefield Functions     - -  qdmpy.system.systems.choose_system Module variables         -  qdmpy.system.systems._CONFIG_PATH -  qdmpy.system.systems._SYSTEMS "
},
{
"ref":"qdmpy.system.systems._CONFIG_PATH",
"url":36,
"doc":"Path to the system directory (e.g. /qdmpy/system) Allows access to config json files."
},
{
"ref":"qdmpy.system.systems.System",
"url":36,
"doc":"Abstract class defining what is expected for a system. Initialisation of system. Must set options_dict."
},
{
"ref":"qdmpy.system.systems.System.name",
"url":36,
"doc":"Name of the system."
},
{
"ref":"qdmpy.system.systems.System.options_dict",
"url":36,
"doc":"Dictionary of available options for this system (loaded from config file)"
},
{
"ref":"qdmpy.system.systems.System.filepath_joined",
"url":36,
"doc":"Used to ensure base_dir is not prepended to filepath twice!"
},
{
"ref":"qdmpy.system.systems.System.read_image",
"url":36,
"doc":"Method that must be defined to read raw data in from filepath. Returns    - image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.System.determine_binning",
"url":36,
"doc":"Method that must be defined to define the original_bin and total_bin options (in the options dict). original_bin equiv. to a camera binning, e.g. any binning before qdmpy was run. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.System.read_sweep_list",
"url":36,
"doc":"Method that must be defined to read sweep_list in from filepath.",
"func":1
},
{
"ref":"qdmpy.system.systems.System.get_raw_pixel_size",
"url":36,
"doc":"Method that must be defined to return a raw_pixel_size. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.System.get_default_options",
"url":36,
"doc":"Method that must be defined to return an options dictionary of default values.",
"func":1
},
{
"ref":"qdmpy.system.systems.System.option_choices",
"url":36,
"doc":"Method that must be defined to return the available option choices for a given option_name",
"func":1
},
{
"ref":"qdmpy.system.systems.System.available_options",
"url":36,
"doc":"Method that must be defined to return what options are available for this system.",
"func":1
},
{
"ref":"qdmpy.system.systems.System.get_bias_field",
"url":36,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmed electromagnet. Default: False, None. Arguments     - options : dict Generic options dict holding all the user options. Returns    - bias_on : bool Was programmed bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (gauss), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"qdmpy.system.systems.System.system_specific_option_update",
"url":36,
"doc":"Hardcoded method that allows updating of the options at runtime with a custom script. In particular this defines some things that cannot be stored in a json.",
"func":1
},
{
"ref":"qdmpy.system.systems.System.get_headers_and_read_csv",
"url":36,
"doc":"Harcoded method that allows reading of a csv file with 'other' measurement data (e.g. temperature) from a csv file. Needs to return headers, csv_data (as a list of strings, and a numpy array). The 1st column should be some sort of independ. data e.g. time.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb",
"url":36,
"doc":"University of Melbourne-wide properties of our systems. Inherited by specific systems defined as sub-classes. Initialisation of system. Must set options_dict."
},
{
"ref":"qdmpy.system.systems.UniMelb.name",
"url":36,
"doc":"Name of the system."
},
{
"ref":"qdmpy.system.systems.UniMelb.uni_defaults_path",
"url":36,
"doc":""
},
{
"ref":"qdmpy.system.systems.UniMelb.get_raw_pixel_size",
"url":36,
"doc":"Method that must be defined to return a raw_pixel_size. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.get_default_options",
"url":40,
"doc":"Method that must be defined to return an options dictionary of default values.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.option_choices",
"url":40,
"doc":"Method that must be defined to return the available option choices for a given option_name",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.available_options",
"url":40,
"doc":"Method that must be defined to return what options are available for this system.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.system_specific_option_update",
"url":40,
"doc":"Hardcoded method that allows updating of the options at runtime with a custom script. In particular this defines some things that cannot be stored in a json.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.get_headers_and_read_csv",
"url":40,
"doc":"Harcoded method that allows reading of a csv file with 'other' measurement data (e.g. temperature) from a csv file. Needs to return headers, csv_data (as a list of strings, and a numpy array). The 1st column should be some sort of independ. data e.g. time.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.options_dict",
"url":40,
"doc":"Dictionary of available options for this system (loaded from config file)"
},
{
"ref":"qdmpy.system.systems.UniMelb.filepath_joined",
"url":40,
"doc":"Used to ensure base_dir is not prepended to filepath twice!"
},
{
"ref":"qdmpy.system.systems.UniMelb.read_image",
"url":36,
"doc":"Method that must be defined to read raw data in from filepath. Returns    - image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.determine_binning",
"url":36,
"doc":"Method that must be defined to define the original_bin and total_bin options (in the options dict). original_bin equiv. to a camera binning, e.g. any binning before qdmpy was run. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.read_sweep_list",
"url":36,
"doc":"Method that must be defined to read sweep_list in from filepath.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.get_default_options",
"url":36,
"doc":"Method that must be defined to return an options dictionary of default values.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.option_choices",
"url":36,
"doc":"Method that must be defined to return the available option choices for a given option_name",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.available_options",
"url":36,
"doc":"Method that must be defined to return what options are available for this system.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.get_bias_field",
"url":36,
"doc":"get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad ",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.system_specific_option_update",
"url":36,
"doc":"Hardcoded method that allows updating of the options at runtime with a custom script. In particular this defines some things that cannot be stored in a json.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.get_headers_and_read_csv",
"url":36,
"doc":"Harcoded method that allows reading of a csv file with 'other' measurement data (e.g. temperature) from a csv file. Needs to return headers, csv_data (as a list of strings, and a numpy array). The 1st column should be some sort of independ. data e.g. time.",
"func":1
},
{
"ref":"qdmpy.system.systems.UniMelb.options_dict",
"url":36,
"doc":"Dictionary of available options for this system (loaded from config file)"
},
{
"ref":"qdmpy.system.systems.UniMelb.filepath_joined",
"url":36,
"doc":"Used to ensure base_dir is not prepended to filepath twice!"
},
{
"ref":"qdmpy.system.systems.Zyla",
"url":36,
"doc":"Specific system details for the Zyla QDM. Initialisation of system. Must set options_dict."
},
{
"ref":"qdmpy.system.systems.Zyla.name",
"url":36,
"doc":"Name of the system."
},
{
"ref":"qdmpy.system.systems.Zyla.config_path",
"url":36,
"doc":""
},
{
"ref":"qdmpy.system.systems.Zyla.get_raw_pixel_size",
"url":36,
"doc":"Method that must be defined to return a raw_pixel_size. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.read_image",
"url":36,
"doc":"Method that must be defined to read raw data in from filepath. Returns    - image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.determine_binning",
"url":36,
"doc":"Method that must be defined to define the original_bin and total_bin options (in the options dict). original_bin equiv. to a camera binning, e.g. any binning before qdmpy was run. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.read_sweep_list",
"url":36,
"doc":"Method that must be defined to read sweep_list in from filepath.",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.get_default_options",
"url":36,
"doc":"Method that must be defined to return an options dictionary of default values.",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.option_choices",
"url":36,
"doc":"Method that must be defined to return the available option choices for a given option_name",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.available_options",
"url":36,
"doc":"Method that must be defined to return what options are available for this system.",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.get_bias_field",
"url":36,
"doc":"get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad ",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.system_specific_option_update",
"url":36,
"doc":"Hardcoded method that allows updating of the options at runtime with a custom script. In particular this defines some things that cannot be stored in a json.",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.get_headers_and_read_csv",
"url":36,
"doc":"Harcoded method that allows reading of a csv file with 'other' measurement data (e.g. temperature) from a csv file. Needs to return headers, csv_data (as a list of strings, and a numpy array). The 1st column should be some sort of independ. data e.g. time.",
"func":1
},
{
"ref":"qdmpy.system.systems.Zyla.options_dict",
"url":36,
"doc":"Dictionary of available options for this system (loaded from config file)"
},
{
"ref":"qdmpy.system.systems.Zyla.filepath_joined",
"url":36,
"doc":"Used to ensure base_dir is not prepended to filepath twice!"
},
{
"ref":"qdmpy.system.systems.cQDM",
"url":36,
"doc":"Specific system details for the cQDM QDM. Initialisation of system. Must set options_dict."
},
{
"ref":"qdmpy.system.systems.cQDM.name",
"url":36,
"doc":"Name of the system."
},
{
"ref":"qdmpy.system.systems.cQDM.config_path",
"url":36,
"doc":""
},
{
"ref":"qdmpy.system.systems.cQDM.get_raw_pixel_size",
"url":36,
"doc":"Method that must be defined to return a raw_pixel_size. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.read_image",
"url":36,
"doc":"Method that must be defined to read raw data in from filepath. Returns    - image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.determine_binning",
"url":36,
"doc":"Method that must be defined to define the original_bin and total_bin options (in the options dict). original_bin equiv. to a camera binning, e.g. any binning before qdmpy was run. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.read_sweep_list",
"url":36,
"doc":"Method that must be defined to read sweep_list in from filepath.",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.get_default_options",
"url":36,
"doc":"Method that must be defined to return an options dictionary of default values.",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.option_choices",
"url":36,
"doc":"Method that must be defined to return the available option choices for a given option_name",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.available_options",
"url":36,
"doc":"Method that must be defined to return what options are available for this system.",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.get_bias_field",
"url":36,
"doc":"get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad ",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.system_specific_option_update",
"url":36,
"doc":"Hardcoded method that allows updating of the options at runtime with a custom script. In particular this defines some things that cannot be stored in a json.",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.get_headers_and_read_csv",
"url":36,
"doc":"Harcoded method that allows reading of a csv file with 'other' measurement data (e.g. temperature) from a csv file. Needs to return headers, csv_data (as a list of strings, and a numpy array). The 1st column should be some sort of independ. data e.g. time.",
"func":1
},
{
"ref":"qdmpy.system.systems.cQDM.options_dict",
"url":36,
"doc":"Dictionary of available options for this system (loaded from config file)"
},
{
"ref":"qdmpy.system.systems.cQDM.filepath_joined",
"url":36,
"doc":"Used to ensure base_dir is not prepended to filepath twice!"
},
{
"ref":"qdmpy.system.systems.CryoWidefield",
"url":36,
"doc":"Specific system details for Cryogenic (Attocube) widefield QDM. Initialisation of system. Must set options_dict."
},
{
"ref":"qdmpy.system.systems.CryoWidefield.name",
"url":36,
"doc":"Name of the system."
},
{
"ref":"qdmpy.system.systems.CryoWidefield.config_path",
"url":36,
"doc":""
},
{
"ref":"qdmpy.system.systems.CryoWidefield.get_raw_pixel_size",
"url":36,
"doc":"Method that must be defined to return a raw_pixel_size. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.read_image",
"url":36,
"doc":"Method that must be defined to read raw data in from filepath. Returns    - image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.determine_binning",
"url":36,
"doc":"Method that must be defined to define the original_bin and total_bin options (in the options dict). original_bin equiv. to a camera binning, e.g. any binning before qdmpy was run. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.read_sweep_list",
"url":36,
"doc":"Method that must be defined to read sweep_list in from filepath.",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.get_default_options",
"url":36,
"doc":"Method that must be defined to return an options dictionary of default values.",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.option_choices",
"url":36,
"doc":"Method that must be defined to return the available option choices for a given option_name",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.available_options",
"url":36,
"doc":"Method that must be defined to return what options are available for this system.",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.get_bias_field",
"url":36,
"doc":"get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad ",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.system_specific_option_update",
"url":36,
"doc":"Hardcoded method that allows updating of the options at runtime with a custom script. In particular this defines some things that cannot be stored in a json.",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.get_headers_and_read_csv",
"url":36,
"doc":"Harcoded method that allows reading of a csv file with 'other' measurement data (e.g. temperature) from a csv file. Needs to return headers, csv_data (as a list of strings, and a numpy array). The 1st column should be some sort of independ. data e.g. time.",
"func":1
},
{
"ref":"qdmpy.system.systems.CryoWidefield.options_dict",
"url":36,
"doc":"Dictionary of available options for this system (loaded from config file)"
},
{
"ref":"qdmpy.system.systems.CryoWidefield.filepath_joined",
"url":36,
"doc":"Used to ensure base_dir is not prepended to filepath twice!"
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield",
"url":36,
"doc":"Specific system details for cryogenic (Attocube) widefield QDM. - Legacy binning version Initialisation of system. Must set options_dict."
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.name",
"url":36,
"doc":"Name of the system."
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.config_path",
"url":36,
"doc":""
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.determine_binning",
"url":36,
"doc":"Method that must be defined to define the original_bin and total_bin options (in the options dict). original_bin equiv. to a camera binning, e.g. any binning before qdmpy was run. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.get_raw_pixel_size",
"url":36,
"doc":"Method that must be defined to return a raw_pixel_size. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.read_image",
"url":36,
"doc":"Method that must be defined to read raw data in from filepath. Returns    - image : np array, 3D Format: [sweep values, y, x]. Has not been seperated into sig/ref etc. and has not been rebinned. Unwanted sweep values not removed. options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.read_sweep_list",
"url":36,
"doc":"Method that must be defined to read sweep_list in from filepath.",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.get_bias_field",
"url":40,
"doc":"get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad ",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.get_raw_pixel_size",
"url":40,
"doc":"Method that must be defined to return a raw_pixel_size. Arguments     - options : dict Generic options dict holding all the user options.",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.get_default_options",
"url":36,
"doc":"Method that must be defined to return an options dictionary of default values.",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.option_choices",
"url":36,
"doc":"Method that must be defined to return the available option choices for a given option_name",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.available_options",
"url":36,
"doc":"Method that must be defined to return what options are available for this system.",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.get_bias_field",
"url":36,
"doc":"get bias on (bool) and field as tuple (mag (G), theta (rad), phi (rad ",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.system_specific_option_update",
"url":36,
"doc":"Hardcoded method that allows updating of the options at runtime with a custom script. In particular this defines some things that cannot be stored in a json.",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.get_headers_and_read_csv",
"url":36,
"doc":"Harcoded method that allows reading of a csv file with 'other' measurement data (e.g. temperature) from a csv file. Needs to return headers, csv_data (as a list of strings, and a numpy array). The 1st column should be some sort of independ. data e.g. time.",
"func":1
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.options_dict",
"url":36,
"doc":"Dictionary of available options for this system (loaded from config file)"
},
{
"ref":"qdmpy.system.systems.LegacyCryoWidefield.filepath_joined",
"url":36,
"doc":"Used to ensure base_dir is not prepended to filepath twice!"
},
{
"ref":"qdmpy.system.systems._SYSTEMS",
"url":36,
"doc":"Dictionary that defines systems available for use. Add any systems you define here so you can use them."
},
{
"ref":"qdmpy.system.systems.choose_system",
"url":36,
"doc":"Returns  qdmpy.system.systems.System object called 'name'.",
"func":1
}
]