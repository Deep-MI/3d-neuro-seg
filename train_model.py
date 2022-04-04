import os
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.MultiResVNet import MultiResVNet, MultiResVNetML
from model.RCVNet import RCVNet, RCVNetAttention
from model.CropNet3D import CropNet3D

from utils.dataloader import HDF5Dataset
from utils.run_utils import train_epoch, train_epoch_multiloss, valid_epoch
from utils.losses import CombinedLoss

def main():

    ## Create model object using params dictionary ##

    # if model is rcvnet
    if params['model_name'] == 'RCVNet':
        if params['sub_model_name'] == 'attention':
            model = RCVNetAttention(params).to(model_device)
        else:
            model = RCVNet(params).to(model_device)

    # If model is cropnet
    elif params['model_name'] == 'CropNet3D':
        model = CropNet3D(params).to(model_device)

    # If model is multiresnet
    elif params['model_name'] == 'MultiResVNet':
        if params['sub_model_name'] == 'multiloss':
            model = MultiResVNetML(params).to(model_device)
        else:
            model = MultiResVNet(params).to(model_device)
    else:
        raise ValueError('model_name must be RCVNet. CropNet3D or MultiResVNet')

    # Map certain layers to different GPUs if specified in params. Only supported by RCVNet
    for l_name, l in model.named_children():
        if l_name in params['gpu_map'].keys():
            l.to(torch.device(params['gpu_map'][l_name]))

    # Initialize Optimizer and Learning rate decay scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['milestones'], gamma=0.5,
                                                     last_epoch=-1)

    # The combined loss function is initialized
    loss = CombinedLoss()

    # Set initial best dice to 0
    best_dice = 0

    # Print model for verification. Summary can be used as an alternative
    print(model)
    # summary(model, input_size=(1,128,128,128))

    # The log_dir is the directory where the experiment model is stored. The experiment id gives the main folder
    log_dir = os.path.join(params['checkpoint_dir'], params['experiment_id'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, 'ckpts'))

    # Copy the training and base model files
    shutil.copy2(__file__, os.path.join(log_dir, "train_model.py"))
    shutil.copy2(f"./model/{params['model_name']}.py", os.path.join(log_dir, f"{params['model_name']}.py"))

    if params['sub_model_name'] == 'vnet' or params['sub_model_name'] == 'multiloss':
        module_name = 'VNet'
    if params['model_name'] == 'RCVNet':
        if params['sub_model_name'] == 'vnet':
            module_name = 'VNet'
        elif params['sub_model_name'] == 'vnet_2d_3d':
            module_name = 'VNet_2D_3D'
        elif params['sub_model_name'] == 'vnet_asym':
            module_name = 'VNetAsym'
        elif params['sub_model_name'] == 'vnet_sym':
            module_name = 'VNetSym'
        elif params['sub_model_name'] == 'vnet_denseadd':
            module_name = 'VNetDenseAdd'
        elif params['sub_model_name'] == 'vnet_exclusion':
            module_name = 'VNetExclusion'
        elif params['sub_model_name'] == 'vnet_se':
            module_name = 'VNetSE'
        elif params['sub_model_name'] == 'attention':
            module_name = 'VNetAttention'
        else:
            raise ValueError(f"{params['sub_model_name']} does not exist.")
    shutil.copy2(f"./model/{module_name}.py", os.path.join(log_dir, f"{module_name}.py"))

    # Tensorboard summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # Best model and optimizer save path
    best_model_save_path = os.path.join(log_dir, params['experiment_id'])
    best_optimizer_save_path = os.path.join(log_dir, params['experiment_id']+'optim_params')

    ## START MAIN TRAINING ##
    for epoch in range(0, params['epochs']):
        print("Experiment ID:{}".format(params['experiment_id']))
        print("\nIteration No: {}".format(epoch + 1))

        # Train one epoch of the model
        if params['model_name'] == 'MultiResVNet':
            if params['sub_model_name'] == 'multiloss':
                av_loss_epoch = train_epoch_multiloss(model, train_loader, loss, optimizer, model_device,
                                                      loss_device, alpha=0.5)
        else:
            av_loss_epoch = train_epoch(model, train_loader, loss, optimizer, model_device, loss_device)
        print("Average Loss: %f" % av_loss_epoch)
        scheduler.step()        # Advance the scheduler

        ## VALIDATION STEP ##
        print("Evaluating on Validation Set:")
        # Performs full volume validation on the validation set
        dice_valid_classwise, dice_valid = valid_epoch(model, valid_loader, loss, model_device, loss_device)
        print("Average Dice Score on Validation Set: %f" % dice_valid)
        print("Classwise Dice on Validation Set: %s\n" % dice_valid_classwise)

        # If the validation dice score is better than the current best, save the model weights
        if dice_valid > best_dice:
            print("Model Saved. (Dice Valid) {:.4f} > (Best Dice){:.4f}".format(dice_valid, best_dice))
            best_dice = dice_valid
            torch.save(model.state_dict(), best_model_save_path)
            torch.save(optimizer.state_dict(), best_optimizer_save_path)

        # Less than 75% of the best_dice is not the right direction on the weight space. Trying again.
        elif os.path.isfile(best_model_save_path) and (dice_valid / (best_dice + 0.0001) < 0.75):
            print("Train Dice < 0.75*Best Dice. Previous Best Model Loaded.")
            model.load_state_dict(torch.load(best_model_save_path))
            try:
                optimizer.load_state_dict(torch.load(best_optimizer_save_path))
            except:
                print('optimizer not found')
                continue

        # Save the model at every 200 epochs or on the last epoch in a ckpts folder.
        if (epoch+1)%200 == 0 or (epoch+1)==params["epochs"]:
            print("Model Saved at epoch {}".format(epoch))
            torch.save(model.state_dict(), os.path.join(log_dir, 'ckpts')+f'/epoch_{epoch+1}')

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
              'epochs': 800,                        # number of training epochs
              'lr': 0.001,                          # Initial learning rate
              'milestones': [320, 640, 720],        # milestones for learning rate decay torch scheduler
              'model_name': 'MultiResVNet',               # Choices between [RCVNet, CropNet3D, MultiResVNet]. Currently name
                                                    # is same as python file for model class
              'sub_model_name': 'multiloss',             # Choices between [vnet, vnet_2d_3d, vnet_asym, vnet_sym,
                                                    # vnet_denseadd, vnet_exclusion, vnet_se, attention, multiloss]
                                                    # multiloss only with MultiResVNet
              'ml_alpha': 0.5,                      # The alpha value for multiloss training of MultiResVNet
              'checkpoint_dir': 'checkpoints',     # Main checkpoint director
              'experiment_id': 'Example',                                      # Experiment ID to store models
              'pretrained': False,
              'gen_random': True,                   # Internal param to make model generate checkpoints
              'gpu_map': {}                         # Maps model blocks by key (python layer var name) to different gpus
              #Eg. 'decoder_block_2': 'cuda:1', 'decoder_block_1': 'cuda:1',
              #'output_block': 'cuda:1'} # MUST ALWAYS map blocks after a mapping sequence ends
              }

    model_device = torch.device('cuda:0')           # Device to store model on
    loss_device = torch.device('cuda:1')            # Device to store loss on. Can be kept same as model_device

    # Load the training Dataset for HDF5 files. Also load weights for loss
    train_dataset = HDF5Dataset(
        filepath=r"data/train.hdf5",
        load_weights=True,
        ret_index=False
    )
    # Feed it through dataloader for automatic shuffling and batch creation
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Load the validation Dataset for HDF5 files. No weights or indices needed
    valid_dataset = HDF5Dataset(
        filepath=r"data/valid.hdf5"
    )
    # Feed it through dataloader for batch creation
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    # Run training
    main()

