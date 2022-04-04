import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.run_utils import eval_epoch, eval_epoch_patched, valid_epoch
from utils.losses import CombinedLoss
from utils.dataloader import HDF5Dataset

def main():
    """
    Runs the main function which loads the model file and triggers the evaluation and dice calculation. DOES NOT save
    the segmentations and eval_model.py should be used to do so.
    """

    # Find the experiment folder, model file and class BY NAME
    package = f"checkpoints.{params['experiment_id']}.{file_string}"
    import_net_class = getattr(__import__(package, fromlist=[file_string]), class_string)
    print(import_net_class)     # print the imported class

    # Intialize the model object by name
    model = import_net_class(params).to(model_device)

    # Intialize combined loss. Can calculate dice scores in eval() mode
    loss = CombinedLoss()

    # Print model for verification. Can use the torchsummary module as an alternative
    print(model)

    # The MAIN directory which stores all other model files
    log_dir = os.path.join(params['checkpoint_dir'], params['experiment_id'])

    # The exact patch to the model
    best_model_save_path = os.path.join(log_dir, params['experiment_id'])
    model.load_state_dict(torch.load(best_model_save_path))

    # Run the evaluation #
    print("Experiment ID:{}".format(params['experiment_id']))
    print("Evaluating on Dataset:")
    dice_valid_classwise, dice_valid = eval_epoch_patched(model, eval_loader, loss, 
                                                          model_device, loss_device,
                                                          p_len=params['patch_size'],
                                                          p_step=params['patch_step'])
    # dice_valid_classwise, dice_valid = eval_epoch(model, test_loader, loss, model_device, loss_device)
    print("Average Dice Score on Dataset: %f" % dice_valid)
    print("Classwise Dice on Dataset: %s\n" % dice_valid_classwise)
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
              'lr': None,                       #   NOT USED. FOR COMPATIBILITY [WILL BE REMOVED]
              'milestones': [],                 #
              'sub_model_name': 'vnet',
              'checkpoint_dir': 'checkpoints',       # Main checkpoint folder
              'experiment_id': 'theid', # Model ID, also the experiment folder name INSIDE checkpoint folder
              'gpu_map':{},                     # layer GPU mapping. Not implemented here. For compatibility
              'F_g': None,                      #
              'F_l': None,                      # For Attention VNet
              'F_int': None,                     #
              'gen_random': None
              }

    # Retrieve model file by name
    file_string = "LResFCNet"                      # Model file name inside model directory
    class_string = "LResFCNetML"                     # Model class name inside model file

    model_device = torch.device('cuda:0')       # model device
    loss_device = torch.device('cuda:1')        # loss device

    # Load test dataset
    eval_dataset = HDF5Dataset(
        filepath=r"data/test.hdf5",
        ret_index=False
    )
    # Feed through Dataloader for automatic batch creation
    eval_loader = DataLoader(eval_dataset, batch_size=1)

    # run evaluation
    main()

