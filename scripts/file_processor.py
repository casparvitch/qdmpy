import numpy as np
import os
import re
import matplotlib.pyplot as plt


def failfloat(a):
    """Used in particular for reading the metadata to convert all numbers into
    floats and leave strings as strings.
    """
    try:
        return float(a)
    except ValueError:
        return a


def read_meta_data(filepath):
    # === get metadata
    # first read off the sweep (i.e. freq/tau times) string, then read
    # out each pair of key:value items in the metadata file. If the
    # value is numeric, store it as a float, else keep the string.
    # Store sweep in a list, metadata in a dict
    # ===

    with open(os.path.normpath(filepath + "_metaSpool.txt"), "r") as fid:
        sweep_str = fid.readline().rstrip().split("\t")
        sweep_list = [float(i) for i in sweep_str]

        rest_str = fid.read()
        matches = re.findall(
            r"^([a-zA-Z0-9_ _/+()#-]+):([a-zA-Z0-9_ _/+()#-]+)",
            rest_str,
            re.MULTILINE,
        )
        metadata = {a: failfloat(b) for (a, b) in matches}

        return sweep_list, metadata


def reshape_raw(raw_data, sweep_list, metadata):
    used_ref = False
    try:
        data_points = len(sweep_list)
        image = np.reshape(
            raw_data,
            [
                data_points,
                int(metadata["AOIHeight"]),
                int(metadata["AOIWidth"]),
            ],
        )
    except ValueError:
        # if the ref is used then there's 2* the number of sweeps
        data_points = 2 * len(sweep_list)
        image = np.reshape(
            raw_data,
            [
                data_points,
                int(metadata["AOIHeight"]),
                int(metadata["AOIWidth"]),
            ],
        )
        used_ref = True

    # Transpose the dataset to get the correct x and y orientations
    for idx in range(len(image)):
        image[idx, ::] = image[idx, ::].transpose()

    return image, used_ref


def rebin_image(image, num_bins, used_ref):
    """
    image: numpy array, before rebinning and sig/ref normalisation
    num_bins: int, additional binning to perform
    used_ref: bool, just to check if we have a reference or not
    """
    if num_bins % 2:
        raise RuntimeError(
            "The binning parameter needs to be a multiple of 2."
        )
    if num_bins:
        data_points = image.shape[0]
        height = image.shape[1]
        width = image.shape[2]

        image = (
            np.reshape(
                image,
                [
                    data_points,
                    int(height / num_bins),
                    num_bins,
                    int(width / num_bins),
                    num_bins,
                ],
            )
            .sum(2)
            .sum(3)
        )
    # define sig and ref differently if we're using a ref
    # you probably don't need them but they're here if required

    if used_ref:
        sig = image[::2, :, :]
        ref = image[1::2, :, :]
        sig_norm = sig / ref
    else:
        sig = ref = image
        sig_norm = sig / np.max(sig, 0)

    return sig_norm


def read_data(filepath, num_bins):
    """
    filepath: str,
    num_bins: int, additional binning to perform (resampling)
    """
    filepath = str(filepath)
    num_bins = int(num_bins)

    # === load dataset, resample, create output directory

    with open(os.path.normpath(filepath), "r") as fid:
        raw_data = np.fromfile(fid, dtype=np.float32())[2:]

    sweep_list, metadata = read_meta_data(filepath)

    # === make output_dir for processed data
    bin_conversion = [1, 2, 4, 8, 16, 32]
    original_bin = bin_conversion[int(metadata["Binning"])]
    total_bin = original_bin * num_bins

    # === now transform data set to correct format. Assuming not already processed

    image, used_ref = reshape_raw(raw_data, sweep_list, metadata)
    sig_norm = rebin_image(image, num_bins, used_ref)

    return sig_norm, sweep_list


if __name__ == "__main__":
    # example of usage

    filepath = "/home/samsc/Desktop/ODMR - CW_217"
    num_bins = 16

    # read the data into sig_norm, in the easiest format for use
    sig_norm, sweep_list = read_data(filepath, num_bins)

    # shape of array, in the first case: (51, 512, 512)
    # print(sig_norm.shape)

    # [0, 0] pixel spectrum:
    # print(sig_norm[:, 0, 0])

    # 0th frequency 'frame':
    # print(sig_norm[0, :, :])

    # average over all pixels
    spct = np.mean(sig_norm, (-1, -2))

    # remove first data data_point
    spct = spct[1:]
    sweep_list = np.array(sweep_list[1:])

    # plt.scatter(sweep_list, spct)
    # plt.show()
    output = np.column_stack((sweep_list, spct))
    np.savetxt(
        "/home/samsc/Desktop/spectrum.txt",
        output,
        delimiter="\t",
        header="freq (MHz)\tPL (a.u.)",
    )

    # you can then save/load this (resampled) data, if you like, to avoid the gui:
    # np.save(), np.load()
    # np.savetxt(), np.loadtxt()
    # etc.
