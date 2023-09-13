import qdmpy
import qdmpy.plot
import qdmpy.shared.polygon as qp

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl

exec(open("../TEST_DATA_PATH.py").read())

numpy_txt_file_path = (
    TEST_DATA_PATH
    + "mz_test/ODMR - Pulsed_10_Rectangle_bin_8/field/sig_sub_ref/sig_sub_ref_bnv_0.txt"
)

json_input_path = TEST_DATA_PATH + "mz_test/polys.json"
json_output_path = TEST_DATA_PATH + "mz_test/polys.json"

mean_plus_minus = 0.5


pgon_lst = qp.polygon_selector(
    numpy_txt_file_path,
    json_output_path=json_output_path,
    json_input_path=json_input_path,
    mean_plus_minus=mean_plus_minus,
)
