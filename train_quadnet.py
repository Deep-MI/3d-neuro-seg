import os
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataloader import HDF5Dataset
from utils.run_utils import train_epoch_quadnet, valid_epoch_quadnet
from utils.losses import CombinedLoss

from model.QuadNet import populate_random_patch_list, create_quadnet_models_optims
from model.RCVNet import RCVNet

def main():

    # The combined loss function is initialized
    loss = CombinedLoss()

    # The log_dir is the directory where the experiment models is stored. By default contains the 'main' pool
    log_dir = os.path.join(params['checkpoint_dir'], params['experiment_id'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize a directory for the 'current' pool. Referred to as 'tmp' folder throughout the code.
    if not os.path.exists(os.path.join(log_dir, 'tmp')):
        os.makedirs(os.path.join(log_dir, 'tmp'))

    # Copy the training, QuadNet and base model files
    shutil.copy2(__file__, os.path.join(log_dir, "train_quadnet.py"))
    shutil.copy2("./model/QuadNet.py", os.path.join(log_dir, "QuadNet.py"))
    shutil.copy2("./model/RCVNet.py", os.path.join(log_dir, "RCVNet.py"))   # THIS NEEDS TO BE CHANGED WHEN USING A
    shutil.copy2("./model/VNet.py", os.path.join(log_dir, "VNet.py"))       # DIFFERENT BASE MODEL

    # Tensorboard summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # Create and save 8 initial QuadNet models (basically copying the base model 8 times)
    print("Creating Base Models (All same initial weights)..")
    create_quadnet_models_optims(log_dir, params.copy(), params['pretrained_path'])
    print("Done")

    # Calculate an initial validation dice score to serve as the starting 'best_dice'
    print("Calculating initial dice on validation set:")
    model = RCVNet(params.copy()).to(model_device)                          # NEEDS TO BE CHANGED FOR A DIFF BASE MODEL
    dice_valid_classwise, dice_valid = valid_epoch_quadnet(model, valid_loader, loss, model_device,
                                                           loss_device, log_dir)

    del model   # Delete the model from memory after saving

    # Print the initial best overall and classwise dice scores
    print("Average Dice Score on Validation Set: %f" % dice_valid)
    print("Classwise Dice on Validation Set: %s\n" % dice_valid_classwise)

    ## START MAIN TRAINING ##
    print("Starting Training..")
    best_dice = dice_valid      # Set best dice as equal to the dice calculated previously

    for epoch in range(0, params['epochs']):
        print("Experiment ID:{}".format(params['experiment_id']))
        print("\nIteration No: {}".format(epoch + 1))

        # Check if any of the milestones are met. If yes reduce learning rate by half
        if len(params['milestones'])>0:
            if epoch==params['milestones'][0]:
                params['lr'] /= 2
                del params['milestones'][0]

        # For each sample in dataset create a random patch PER SAMPLE, PER MODEL in the respective model quadrant
        net_coords = populate_random_patch_list(params["input_shape"], params["patch_size"], len(train_loader.dataset))

        # Iterate over the keys in net_coords, that is, each quadrant model ID
        for idx, k in enumerate(net_coords.keys()):

            # Extract the patch coordinates of the current model
            quad_data = net_coords[k]
            print(f"Samples for model {idx+1}: {len(quad_data)}")
            coords_dict = dict(quad_data)

            # Create a model object by copying the params file as some values are changed inside
            model = RCVNet(params.copy()).to(model_device)
            # Create an optimizer object
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

            # Load the previous states of the model and the optimizer from the current pool
            part_model_save_path = os.path.join(log_dir, 'tmp', f"quad_model_{idx+1}")
            part_optim_save_path = os.path.join(log_dir, 'tmp', f"quad_optim_{idx+1}")
            model.load_state_dict(torch.load(part_model_save_path))
            optimizer.load_state_dict(torch.load(part_optim_save_path))

            # Train one epoch of the quadnet on the current quad model
            av_loss_epoch = train_epoch_quadnet(model, train_loader, loss, optimizer,
                                                model_device, loss_device, coords_dict)
            print("Average Loss: %f" % av_loss_epoch)

            # Save both the optimizer and the quad model states into the current pool
            torch.save(optimizer.state_dict(), part_optim_save_path)
            torch.save(model.state_dict(), part_model_save_path)
            # scheduler.step()

            del model
            # RUN VALIDATION #
            print("Evaluating on Validation Set:")
            model = RCVNet(params.copy()).to(model_device)              # Create a new model object
            part_model_path_dict = {idx+1: part_model_save_path}        # Create a dictionary to store the index and
                                                                        # path of the quad model in the current pool

            # Run one epoch of validation by temporarily inserting the current quad model into the 'main' pool
            dice_valid_classwise, dice_valid = valid_epoch_quadnet(model, valid_loader, loss, model_device, loss_device,
                                                                   log_dir, part_model_path=part_model_path_dict)
            del model
            print("Average Dice Score on Validation Set: %f" % dice_valid)
            print("Classwise Dice on Validation Set: %s\n" % dice_valid_classwise)

            # If the dice 'after' inserting the 'current' quad model is better. make the change permanent
            if dice_valid > best_dice:
                print("Model Saved. (Dice Valid) {:.4f} > (Best Dice){:.4f}".format(dice_valid, best_dice))
                best_dice = dice_valid
                # Making change permanent by copying the current quad model into the pool
                shutil.copy2(
                        os.path.join(part_model_save_path),
                        os.path.join(log_dir, f"quad_model_{idx+1}")
                    )

        # Write tensorboard log
        writer.add_scalar('Dice Average/valid', dice_valid, epoch)
        writer.add_scalars('Dice Classwise/valid', {f'{i}': dice_valid_classwise[i] for
                                                    i in range(params['num_classes'])}, epoch)

        print("\n\n")

    writer.close()


if __name__ == "__main__":

    params = {'in_channels': 1,                     # Input channel of first layer (modified in model for later layers)
              'out_channels': 16,                   # Output channel of first layer (modified in model for later layers)
              'create_layer_1': False,              # Internal param to indicate layer 1 is being created
              'create_layer_2': False,              # Internal param to indicate layer 2 is being created
              'kernel_size': (5, 5, 5),             # kernel size
              'input_shape': (256,256,256),         # input volume shape l x b x w
              'patch_size': (128,128,128),          # patch size
              'num_classes': 79,                    # number of classes
              'out': False,                         # Internal param to indicate output layer creation
              'input': True,                        # Internal param to make ip and op channels to 1st layer equal for
                                                    # residuals to be used (1 -> 16)
              'epochs': 50,                         # number of training epochs
              'lr': 0.001,                          # Initial learning rate
              'milestones': [20, 30, 40],           # milestones for MANUAL learning rate decay
              'sub_model_name': 'vnet',
              'checkpoint_dir': 'checkpoints', # Main checkpoint director
              'experiment_id': 'QuadNet_128_RCVNet_alter',                          # Experiment ID to store models
              'pretrained_path': 'checkpoints/unensembled/RCVNet_128',
                                                                                    # Path to pretrained model
              'gen_random': False,                                                  # Internal param for random cropping
              'gpu_map': {}                         # Can be used to map individual layers to different GPUs by layer
                                                    # name. Not used here but kept for compatibility
              }

    model_device = torch.device('cuda:0')           # Device to store model on
    loss_device = torch.device('cuda:1')            # Device to store loss on. Can be kept same as model_device


    # Load the training Dataset for HDF5 files. Also load weights for loss and indices for QuadNet
    train_dataset = HDF5Dataset(
        filepath=r"data/train.hdf5",
        load_weights=True,
        ret_index=True
    )
    # Feed it through dataloader for automatic shuffling and batch creation
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Load the validation Dataset for HDF5 files. No weights or indices needed
    valid_dataset = HDF5Dataset(
        filepath=r"data/validation.hdf5"
    )

    # Feed it through dataloader for batch creation
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # Run training
    main()

