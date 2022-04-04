import numpy as np

"""
Contains preprocessing code for creating additional information based on MRI volumes and true segmentation maps (asegs).
Eg. weight masks for median frequency class weighing, edge weighing etc.
"""

def create_weight_mask(aseg):
    """
    Main function for calculating weight mask of segmentation map for loss function. Currently only Median Frequency
    Weighing is implemented. Other types can be additively added to the 'weights' variable

    Args:
        aseg (numpy.ndarray): Segmentation map with shape l x w x d

    Returns:
         numpy.ndarray: Weight Mask of same shape as aseg
    """
    if len(aseg.shape)==4:
        _, h,w,d = aseg.shape
    elif len(aseg.shape)==3:
        h,w,d = aseg.shape

    weights = np.zeros((h,w,d), dtype=float)    # Container ndarray of zeros for weights

    weights += median_freq_class_weighing(aseg)  # Add median frequency weights

    # Further weights (eg. extra weights for region borders) can be added here
    # Eg. weights += edge_weights(aseg)

    return weights


def median_freq_class_weighing(aseg):
    """
    Median Frequency Weighing. Guarded against class absence of certain classes.

    Args:
        aseg (numpy.ndarray): Segmentation map with shape l x w x d

    Returns:
        numpy.ndarray: Median frequency weighted mask of same shape as aseg
    """

    # Calculates median frequency based weighing for classes
    unique, counts = np.unique(aseg, return_counts=True)
    if len(aseg.shape)==4:
        _, h,w,d = aseg.shape
    elif len(aseg.shape)==3:
        h,w,d = aseg.shape

    class_wise_weights = np.median(counts)/counts
    aseg = aseg.astype(int)

    # Guards against the absence of certain classes in sample
    discon_guard_lut = np.zeros(int(max(unique))+1)-1
    for idx, val in enumerate(unique):
        discon_guard_lut[int(val)] = idx

    discon_guard_lut = discon_guard_lut.astype(int)

    # Assigns weights to w_mask and resets the missing classes
    w_mask = np.reshape(class_wise_weights[discon_guard_lut[aseg.ravel()]], (h, w, d))
    return w_mask


# Label mapping functions (to aparc (eval) and to label (train))
def map_label2aparc_aseg(mapped_aseg):
    """
    Function to perform look-up table mapping from label space to aparc.DKTatlas+aseg space
    :param np.ndarray mapped_aseg: label space segmentation (aparc.DKTatlas + aseg)
    :return:
    """
    aseg = np.zeros_like(mapped_aseg)
    labels = np.array([0, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                       15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                       46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                       77, 1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                       1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                       2002, 2005, 2010, 2012, 2013, 2014, 2016, 2017, 2021, 2022, 2023,
                       2024, 2025, 2028])
    h, w, d = aseg.shape

    aseg = labels[mapped_aseg.ravel()]

    aseg = aseg.reshape((h, w, d))

    return aseg


# if __name__ == "__main__":
#     #a = np.random.randint(0, 5, size=(10,10,10))
#     #b = np.random.randint(5, 10, size=(10000))
#
#     #map_masks_into_5_classes(np.random.randint(0, 250, size=(256, 256, 256)))
#
#     import nibabel as nib
#     from data_utils.process_mgz_into_hdf5 import map_aparc_aseg2label, map_aseg2label
#     path = r"abide_ii/sub-28675/mri/aparc.DKTatlas+aseg.mgz"
#     aseg = nib.load(path).get_data()
#     labels_full, _ = map_aparc_aseg2label(aseg)  # only for 79 classes case
#     # labels_full, _ = map_aseg2label(aseg)        # only for 37 classes case
#     aseg = labels_full
#     # print(aseg.shape)
#     median_freq_class_weighing(aseg)
#     # print(edge_weighing(aseg, 1.5))
