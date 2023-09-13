Quantum Diamond MicroscoPy
==========================

# Introduction

If you're installing for the first time scroll down to 'Installing with pip on new system'.

# Usage

Best used in a jupyter notebook to avoid producing a million graphs. In future a command-line suitable hook will be defined, hopefully utilizing some sort of template scheme (e.g. `qdmpy --template=mz --fit-backend=gpu ./ODMR_1`)

Usage is best understood through the example notebooks.


# Module Hierarchy

View in text editor if scrambled.

```
      =================================
      Subpackage dependency graph (DAG)
      =================================

                                                           +--------------------------+
                   +-----+                                 |  ===                     |
                   |qdmpy|                                 |  Key                     |
   +---------------+=====+------------+                    |  ===                     |
   |               +-----+            |                    |                          |
   |                                  |                    | +----+                   |
   v                                  v                    | |name|  =  Package       |
+--+---+  +--+  +-----+  +------+  +--+-+  +------+        | |====|                   |
|system|  |pl|  |field|  |source|  |plot|  |magsim|        | +----+                   |
|======|  |==|  |=====|  |======|  |====|  |======|        |                          |
+--+---+  +-++  +--+--+  +--+---+  +-+--+  +--+---+        |  name   =  Module        |
   |        |      |        |        |        |            |  +--+                    |
   |        |      |        |        |        |            |                          |
   |        |      |        |        |        |            |                          |
   |        v      v        v        |        |            |  +-->   =  Dependency    |
   |     +--+------+--------+------+<+        |            |                          |
   +---->+          shared         |          |            +--------------------------+
         |          ======         +<---------+
         | geom                    |
         | +--+    misc    itool   |
         |         +--+    +-+-+   |
         |                   |     |             CANNOT IMPORT FROM HIGHER IN HEIRARCHY
         | fourier           v     |
         | +-----+        polygon  |
         |                +--+--+  |
         | linecut           |     |
         | +-+---+           |     |
         |   |               |     |
         |   +---+--------+--+     |
         |       |        |        |
         |       v        v        |
         |    widget    json2dict  |
         |    +----+    +-------+  |
         +-------------------------+

```

# Order of operations

View in text editor if scrambled.

```
+---------------------------------------------------------------------------------------+                                                                                 
| Methods are qdmpy.{what is listed here}                                               |                                                                                 
|---------------------------------------------------------------------------------------|                                                                                 
| Functions                         Variables: type          Plotting                   |                                                                                 
|---------------------------------------------------------------------------------------|                                                                                 
|                                   options = {                                         |                                                                                 
| initialize                           ...                                              |                                                                                 
|                                   }: dict or json                                     |                                                                                
|                                   (same for ref)                                      |                                                                                 
|                                                                                       |                                                                                 
| pl.load_image_and_sweep           sig_norm: 3D ndarray     plot.roi_pl_image          |                                                                                 
| pl.reshape_dataset                sweep_list: list         plot.aoi_pl_image          |                                                                                 
|                                                            plot.aoi_spectra           |                                                                                 
|                                                                                       |                                                                                 
| pl.save_pl_data                                                                       |                                                                                 
|                                   ref_fit_params: dict                                |                                                                                 
| pl.load_ref_exp_pl_fit_results    ref_sigmas: dict                                    |                                                                                 
|                                                                                       |                                                                                 
| pl.define_fit_model               fit_model: FitModel                                 |                                                                                 
| pl.fit_roi_avg_pl                                          plot.roi_avg_fits          |                                                                                 
|                                                            plot.aoi_spectra_fit       |                                                                                 
|                                                                                       |                                                                                 
| pl.get_pl_fit_result              pixel_fit_params: dict   plot.pl_param_images       |                                                                                 
|                                   sigmas: dict             plot.pl_param_sigmas       |                                                                                 
| pl.save_pl_fit_results                                     plot.pl_params_flattened   |                                                                                 
| pl.save_pl_fit_sigmas                                                                 |                                                                                 
|                                                                                       |                                                                                 
| field.odmr_field_retrieval        field_res: dict                                     |                                                                                 
|                                                            plot.bnvs_and_dshifts      |                                                                                 
| field.add_bfield_theta_phi                                 plot.bfield                |                                                                                 
| field.save_field_calcs                                     plot.dshift_fit            |                                                                                 
|                                                            plot.field_param_flattened |                                                                                 
|                                                            plot.bfield_consistency    |                                                                                 
|                                                            plot.bfield_theta_phi      |                                                                                 
|                                                                                       |                                                                                 
|                                                                                       |                                                                                 
| source.odmr_source_retrieval      source_params: dict                                 |                                                                                 
| source.save_source_params                                  plot.current               |                                                                                 
|                                                            plot.current_stream        |                                                                                 
|                                                            plot.divperp_j             |                                                                                 
|                                                                                       |                                                                                 
| save_options                                               plot.magnetization         |                                                                                 
|                                                                                       |                                                                                 
|                                                            plot.other_measurements    |                                                                                 
+---------------------------------------------------------------------------------------+
```

# Ok but how do I actually use qdmpy Sam?

Look at the example notebooks. Magsim and other parts of the code are a bit more separate and don't use notebooks, but they have their own example scripts - lovely.

How to run an example script (if you're in the example directory in a terminal, which you can navigate to with the commands `cd` and `dir` on windows):

```Bash
python <script_name>.py
```

How to load jupyter notebooks, run this (e.g. from the examples dir):
```Bash
jupyter-lab
```

... then click on the relevant notebook in your browser. I hope I don't need to spell it out much more :)

# Magsim usage

TODO. Code is still pretty rough. My next big TODO is cleaning this up though!

# Install

See INSTALL.md

# Options reference

Check the unimelb_defaults.json file (in the system folder).

# How to implement your own custom System class

Key things to check:

- the '_SYSTEMS' variable in systems.py
- implement your own child of the System class, including all its methods (unless you're inheriting from the unimelb defaults, which would be the case if you're at RMIT/Unimelb).
