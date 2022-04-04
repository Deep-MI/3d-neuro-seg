import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.dataloader import HDF5Dataset
from utils.run_utils import eval_epoch_quadnet
from utils.losses import CombinedLoss

from model.RCVNet import RCVNet

def main():
    """
    Runs the main function which loads the model file and triggers the evaluation and dice calculation. DOES NOT save
    the segmentations and eval_model.py should be used to do so.
    """

    # Intialize combined loss. Can calculate dice scores in eval() mode
    loss = CombinedLoss()

    # The MAIN directory which stores the quadrant model files
    log_dir = os.path.join(params['checkpoint_dir'], params['experiment_id'])

    # Initialize model object
    model = RCVNet(params.copy()).cuda()

    # Run the evaluation #
    print("Experiment ID:{}".format(params['experiment_id']))
    print("Evaluating on Validation Set:")
    dice_valid_classwise, dice_valid = eval_epoch_quadnet(model, eval_loader, loss,
                                                          model_device, loss_device, log_dir,
                                                          base_pretrained_path=params["pretrained_path"],
                                                          p_len=params['patch_size'], p_step=params['patch_step'])

    print("Average Dice Score on Validation Set: %f" % dice_valid)
    print("Classwise Dice on Validation Set: %s\n" % dice_valid_classwise)
    print(np.mean(dice_valid_classwise[1:]), np.std(dice_valid_classwise[1:]), np.median(dice_valid_classwise[1:]),
          np.min(dice_valid_classwise[1:]), np.max(dice_valid_classwise[1:]))
    print("\n\n")


if __name__ == "__main__":

    params = {'in_channels': 1,                 # Input channel of first layer (modified in model for later layers)
              'out_channels': 16,               # Output channel of first layer (modified in model for later layers)
              'create_layer_1': False,          # Internal param to indicate layer 1 is being created
              'create_layer_2': False,          # Internal param to indicate layer 2 is being created
              'kernel_size': (5, 5, 5),         # kernel size
              'input_shape': (256,256,256),     # input volume shape l x b x w
              'patch_size': 128,                # Patch side for overlapping evaluation
              'patch_step': 128,                # Patch stride for overlapping evaluation
              'num_classes': 79,                # Number of classes
              'out': False,                     # Internal param to indicate output layer creation
              'input': True,                    # Internal param to make ip and op channels to 1st layer equal for
                                                # residuals to be used (1 -> 16)
              'epochs': None,                   #
              'lr': None,                       #   NOT USED. FOR COMPATIBILITY (WILL BE REMOVED)
              'milestones': [],                 #
              'experiment_id': 'theid',  # Model ID, also the experiment folder name INSIDE
                                                            # checkpoint folder
              'sub_model_name': 'vnet',
              'pretrained_path': None,          # Path to original base pretrained model
              'gen_random': False,              # For compatibility. Not used
              'gpu_map': {}                     # GPU mapping of individual layers if necessary. Not used
              }

    model_device = torch.device('cuda:0')       # model device
    loss_device = torch.device('cuda:1')        # loss device

    # Load test dataset
    eval_dataset = HDF5Dataset(
        filepath=r"data/test.hdf5",
    )
    # Feed through Dataloader for automatic batch creation
    eval_loader = DataLoader(eval_dataset, batch_size=1)

    # run evaluation
    main()

