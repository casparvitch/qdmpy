"""Quantum Diamond MicroscoPy: A module/package for analysing widefield NV microscopy images.

'Super-package' that holds all of the others within.

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
+------+  +--+  +-----+  +------+  +----+  +------+        | |====|                   |
|system|  |pl|  |field|  |source|  |plot|  |magsim|        | +----+                   |
|======|  |==|  |=====|  |======|  |====|  |======|        |                          |
+--+---+  +-++  +--+--+  +--+---+  +-+--+  +--+---+        |  name   =  Module        |
   |        |      |        |        |        |            |  ----                    |
   |        |      |        |        |        |            |                          |
   |        |      |        |        |        |            |                          |
   |        v      v        v        |        |            |  --->   =  Dependency    |
   |     +-------------------------+<+        |            |                          |
   +---->|          shared         |          |            +--------------------------+
         |          ======         |<---------+
         |                         |
         |                 itool   |
         |       geom      -----   |            +--------------+
         |       ----        |     |            | driftcorrect |
         |                   v     |            |==============|
         | misc           polygon  |            +--------------+
         | ----           -------  |
         |                   |     |                
         |      fourier      v     |                CANNOT IMPORT FROM HIGHER IN HEIRARCHY
         |      -------  json2dict |               
         |               --------- |                
         +-------------------------+
```


- `qdmpy.driftcorrect`
    - Sub-package to correct for (in-plane) drifts before continuing with rest of analysis.
      Not particularly mature.
- `qdmpy.field`
    - Field sub-package. Contains functions to convert bnvs/resonances to fields (e.g.
     magnetic, electric, ...) through hamiltonian fits/otherwise.
- `qdmpy.magsim`
    - Tooling that simulates magnetic field produced by magnetised flakes (static only).
      Not particularly mature.
- `qdmpy.pl`
    - Sub-package for dealing with pl data. Contains procedures for fitting raw photoliminescence,
      outputting results etc.
- `qdmpy.plot`
    - This sub-package contains all of the plotting functions (matplotlib based).
- `qdmpy.source`
    - Contains tools for reconstructing source fields (e.g. current densities or intrinsic 
      magnetization) from the measured magnetic field calculated in qdmpy.field.
- `qdmpy.shared`
    - Contains procedures shared between the other higher level modules. Cannot import from the
      other modules or you'll get circular import errors. Specific tooling here includes those
      to help with fourier transforms, NV geometry, image tooling such as filtering and 
      background subtraction, as well as json io and polygon selection.
- `qdmpy.system`
    - This sub-package contains the tooling for defining institution specific settings for example
     for loading raw datafiles etc. These settings can be implemented down to the specific
     experimental 'system' to define pixel sizes etc.

`qdmpy` itself also exposes some functions from qdmpy.interface
"""

from qdmpy.interface import *  # noqa: F401, F403
