import qdmpy
import qdmpy.plot
import qdmpy.shared.polygon as qp

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl

numpy_txt_file_path = (
    "/home/samsc/ResearchData/test_images/mz_test/ODMR -"
    " Pulsed_10_Full_bin_8/field/sig_sub_ref/sig_sub_ref_bnv_0.txt"
)
json_output_path = "/home/samsc/src/nv/qdmpy_proj/qdmpy_git/examples/polytest.json"
json_input_path = "/home/samsc/src/nv/qdmpy_proj/qdmpy_git/examples/polytest.json"
mean_plus_minus = 0.2


pgon_lst = qp.polygon_selector(
    numpy_txt_file_path,
    json_output_path=json_output_path,
    json_input_path=json_input_path,
    mean_plus_minus=mean_plus_minus,
)
