import torch
import torch.nn as nn
from model.RCVNet import RCVNet

from numpy.random import randint
import os

from model.RCVNet import RCVNet

# SLightly different from random patches. Trying to generate viable "left corners"
def make_rand_coords(start_coords=(128, 128, 128), patch_size=(64, 64, 64)):
    return [randint(start_coords[0],  start_coords[0] + int(patch_size[0]/2)), \
            randint(start_coords[1],  start_coords[1] + int(patch_size[1]/2)), \
            randint(start_coords[2],  start_coords[2] + int(patch_size[2]/2))]

#
# def get_dims(upper):
#     # Random value in the range [0, upper)
#     return int(upper * random_sample())


def populate_random_patch_list(input_shape, patch_size, num_samples):
    net_coords = {
        (0, 0, 0): [],
        (64, 0, 0): [],
        (0, 64, 0): [],
        (0, 0, 64): [],
        (64, 64, 0): [],
        (64, 0, 64): [],
        (0, 64, 64): [],
        (64, 64, 64): []
    }

    for i in range(num_samples):
        for idx, curr_coords in enumerate(net_coords):
            # if coords[0] in range(curr_coords[0], curr_coords[0] + int(patch_size[0] / 2)) and \
            #         coords[1] in range(curr_coords[1], curr_coords[1] + int(patch_size[1] / 2)) and \
            #         coords[2] in range(curr_coords[2], curr_coords[2] + int(patch_size[2] / 2)):
            coords = make_rand_coords(curr_coords, patch_size)
            net_coords[curr_coords].append((i, coords))

    return net_coords


def assign_nets_to_coords(input_shape, patch_size, coords_list):
    net_coords = {
        (0, 0, 0): [],
        (64, 0, 0): [],
        (0, 64, 0): [],
        (0, 0, 64): [],
        (64, 64, 0): [],
        (64, 0, 64): [],
        (0, 64, 64): [],
        (64, 64, 64): []
    }

    input_shape = input_shape
    patch_size = patch_size

    for idx, coords in enumerate(coords_list):
        # coords = make_rand_coords(input_shape, patch_size)

        for _, curr_coords in enumerate(net_coords):
            if coords[0] in range(curr_coords[0], curr_coords[0] + (int(patch_size[0] / 2) + patch_size[0])) and \
                    coords[1] in range(curr_coords[1], curr_coords[1] + (int(patch_size[0] / 2) + patch_size[1])) and \
                    coords[2] in range(curr_coords[2], curr_coords[2] + (int(patch_size[0] / 2) + patch_size[2])):
                net_coords[curr_coords].append((idx, coords))

    return net_coords


def create_quadnet_models_optims(log_dir, params, pretrained_path=None):

    model = RCVNet(params).cuda()
    if pretrained_path is not None:
        print(f"Loading pretrained RCVNet from {pretrained_path}..")
        model.load_state_dict(torch.load(pretrained_path))
        print("Done.\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    for i in range(1, 9):
        if os.path.exists(os.path.join(log_dir, f'quad_model_{i}')):
            model.load_state_dict(torch.load(os.path.join(log_dir, f'quad_model_{i}')))
            print(f"Model quad_model_{i} Exists. Reusing. Delete manually for fresh training.")

        print(f"Creating tmp model {i}..")
        torch.save(model.state_dict(), os.path.join(log_dir, f'quad_model_{i}'))
        torch.save(model.state_dict(), os.path.join(log_dir, 'tmp', f'quad_model_{i}'))
        torch.save(optimizer.state_dict(), os.path.join(log_dir, 'tmp', f'quad_optim_{i}'))
        print(f"Done")
    del model


if __name__ == "__main__":
    # from torchsummary import summary
    #
    # params = {'in_channels': 1,
    #           'out_channels': 16,
    #           'create_layer_1': False,
    #           'create_layer_2': False,
    #           'kernel_size': (5, 5, 5),
    #           'input_shape': (256, 256, 256),
    #           'patch_size': (128, 128, 128),
    #           'num_classes': 79,
    #           'input': True,
    #           'out': False,
    #           }
    #
    # m = RCVNet(params).cuda()
    # summary(m, input_size=(1, 64, 64, 64))

    print(populate_random_patch_list(input_shape=(256,256,256), patch_size=(128,128,128), num_samples=10))
