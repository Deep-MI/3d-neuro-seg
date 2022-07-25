import torch
import h5py

from torch.utils.data import Dataset

from .preprocess import create_weight_mask


class HDF5Dataset(Dataset):
    """
    Yields samples from an HDF5 file by index
    """

    def __init__(self, filepath, load_weights=False, ret_index=False):
        """
        Initializes and HDF5 dataset

        Args:
            filepath (string): Absolute path to the HDF5 file
            load_weights (bool): Assign True to return median freq based weight map
            ret_index (bool): Assign True to return indices
        """

        super(HDF5Dataset, self).__init__()
        h5_file = h5py.File(filepath, 'r')
        self.origs = h5_file.get("origs")
        self.asegs = h5_file.get("asegs")
        self.paths = h5_file.get("path_ids")
        self.len = self.origs.shape[0]
        self.load_weights = load_weights
        self.ret_index = ret_index

        print("Data and Target shapes: "+str(self.origs.shape)+" "+str(self.asegs.shape))

    def __getitem__(self, index):
        """
        Returns samples, weights and indices to dataloader by index
        Args:
            index (int): The index of sample to yield

        Returns:
            vol, segmentation, weight (optional), indices (optional)
        """

        if index>=self.len:     # Some bug with h5py and enumerate(..). The StopIteration
            raise StopIteration # never gets raised at the end of the dataset. Not used for dataloader

        # Retrieves the volumes and asegs by index
        orig, aseg = self.origs[index], self.asegs[index]

        # Default weights
        weights = 1.0

        # Create median frequency based weights if flag is set
        if self.load_weights is True:
            weights = create_weight_mask(aseg)
            weights = torch.from_numpy(weights).float()

        orig = torch.from_numpy(orig).float()
        aseg = torch.from_numpy(aseg).float()

        if self.ret_index:
            return index, orig, aseg, weights
        return orig, aseg, weights

    def __len__(self):
        return self.len