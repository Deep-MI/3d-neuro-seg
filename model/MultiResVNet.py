import torch
import torch.nn as nn

from numpy.random import random_sample

from .VNet import EncoderBlock, BottleNeck, DecoderBlock, DecoderBlockTrans, DecoderBlockTransML


def make_rand_coords(input_size=(256,256,256), patch_size=(64,64,64)):
    return [get_dims(input_size[0] - patch_size[0]),
            get_dims(input_size[1] - patch_size[1]),
            get_dims(input_size[2] - patch_size[2])]


def get_dims(upper):
    # Random value in the range [0, upper)
    return int(upper * random_sample())


class MultiResVNet(nn.Module):


    def __init__(self, params):
        super(MultiResVNet, self).__init__()

        self.coords = None
        self.input_shape = params['input_shape']
        self.patch_size = params['patch_size']

        # self.down_input_lower_1 = nn.Sequential(
        #         nn.Conv3d(in_channels=params['in_channels'], out_channels=4*params['out_channels'],
        #                   kernel_size=(1,1,1), padding=0, stride=1),
        #         nn.AvgPool3d(kernel_size=(8,8,8), stride=4, padding=0),
        #         nn.GroupNorm(num_groups=4, num_channels=4*params['out_channels']),
        #         nn.PReLU()
        #     )

        self.down_input_lower_1 = nn.Sequential(
                nn.Conv3d(in_channels=params['in_channels'], out_channels=4*params['out_channels'],
                          kernel_size=(4,4,4), padding=0, stride=4),
                nn.GroupNorm(num_groups=4, num_channels=4*params['out_channels']),
                nn.PReLU()
            )

        # self.norm_add = nn.Sequential(nn.GroupNorm(num_groups=4, num_channels=4*params['out_channels']),
        #         nn.PReLU()
        #     )

        # in_channels: 16, out_channels: 16
        self.encoder_block_1 = EncoderBlock(params)

        params['input'] = False
        params['create_layer_1'] = True
        params['in_channels'] = params['out_channels'] * 2 # 32
        params['out_channels'] = params['out_channels'] * 2 # 32
        self.encoder_block_2 = EncoderBlock(params)

        params['create_layer_2'] = True
        params['in_channels'] = params['out_channels'] * 2 # 64
        params['out_channels'] = params['out_channels'] * 2 # 64
        self.encoder_block_3 = EncoderBlock(params)

        params['in_channels'] = params['out_channels'] * 2 # 128
        params['out_channels'] = params['out_channels'] * 2 # 128
        self.encoder_block_4 = EncoderBlock(params)

        params['in_channels'] = params['out_channels'] * 2 # 256
        params['out_channels'] = int(params['out_channels'] * 2) # 256
        self.bottleneck_block = BottleNeck(params)

        enc_channels = 128
        params['in_channels'] = params['out_channels'] + enc_channels # 256 + 128
        params['out_channels'] = params['out_channels'] # 256
        self.decoder_block_4 = DecoderBlock(params)

        enc_channels = int(enc_channels/2)
        params['in_channels'] = int(params['out_channels']/2) + enc_channels # 128 + 64
        params['out_channels'] = int(params['out_channels'] / 2)  # 128
        self.decoder_block_3 = DecoderBlockTrans(params)

        enc_channels = int(enc_channels/2)
        params['in_channels'] = int(params['out_channels'] / 2) + enc_channels  # 64 + 32
        params['out_channels'] = int(params['out_channels'] / 2) # 64
        params['create_layer_2'] = False
        self.decoder_block_2 = DecoderBlock(params)

        enc_channels = int(enc_channels/2)
        params['in_channels'] = int(params['out_channels'] / 2) + enc_channels # 32 + 16
        params['out_channels'] = int(params['out_channels'] / 2) # 32
        params['create_layer_1'] = False
        params['out'] = True
        self.decoder_block_1 = DecoderBlock(params)
        params['out'] = False

        self.output_block = nn.Conv3d(in_channels=params['out_channels'], out_channels=params['num_classes'],
                                      kernel_size = (1, 1, 1), stride = 1, padding = 0)

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)


    def train_forward(self, x):

        # Generate Random Coordinates
        self.coords = make_rand_coords(self.input_shape, self.patch_size)

        # Create patching parameters for low res and high res patches
        coords_lower = [int(c/4) for c in self.coords]
        patch_lower  = [int(p/4) for p in self.patch_size]

        # Generate 4X downsampled patches
        x_lower = self.down_input_lower_1(x)
        # x_lower = self.down_input_lower_2(x_lower)

        # Cropped Patch for upper part of encoder
        x_upper = x[..., self.coords[0]:self.coords[0] + self.patch_size[0],
                  self.coords[1]:self.coords[1] + self.patch_size[1],
                  self.coords[2]:self.coords[2] + self.patch_size[2]]

        # Run upper encoder with cropped intput
        x_upper_res_1, x_down_1 = self.encoder_block_1(x_upper)
        x_upper_res_2, x_down_2 = self.encoder_block_2(x_down_1)

        # Location aware addition for feature map integration
        x_lower[..., coords_lower[0]:coords_lower[0] + patch_lower[0],
                     coords_lower[1]:coords_lower[1] + patch_lower[1],
                     coords_lower[2]:coords_lower[2] + patch_lower[2]] =\
        x_lower[..., coords_lower[0]:coords_lower[0] + patch_lower[0],
            coords_lower[1]:coords_lower[1] + patch_lower[1],
            coords_lower[2]:coords_lower[2] + patch_lower[2]] + x_down_2

        # x_lower = self.norm_add(x_lower)

        # Running lower encoder, bottleneck and decoder with
        x_lower_res_3, x_down_3 = self.encoder_block_3(x_lower)
        x_lower_res_4, x_down_4 = self.encoder_block_4(x_down_3)

        x_bottleneck = self.bottleneck_block(x_down_4)

        x_up_4 = self.decoder_block_4(x_lower_res_4, x_bottleneck)
        x_up_3 = self.decoder_block_3(x_lower_res_3, x_up_4, coords_lower, patch_lower)
        x_up_2 = self.decoder_block_2(x_upper_res_2, x_up_3)
        x_last = self.decoder_block_1(x_upper_res_1, x_up_2)

        out = self.output_block(x_last)

        return out


    def eval_forward(self, x):

        x_upper_res_1, x_down_1 = self.encoder_block_1(x)
        x_upper_res_2, x_down_2 = self.encoder_block_2(x_down_1)
        x_lower_res_3, x_down_3 = self.encoder_block_3(x_down_2)
        x_lower_res_4, x_down_4 = self.encoder_block_4(x_down_3)

        x_bottleneck = self.bottleneck_block(x_down_4)

        x_up_4 = self.decoder_block_4(x_lower_res_4, x_bottleneck)
        x_up_3 = self.decoder_block_3(x_lower_res_3, x_up_4, None, None)
        x_up_2 = self.decoder_block_2(x_upper_res_2, x_up_3)
        x_last = self.decoder_block_1(x_upper_res_1, x_up_2)

        out = self.output_block(x_last)
        return out


class MultiResVNetML(nn.Module):


    def __init__(self, params):
        super(MultiResVNetML, self).__init__()

        self.coords = None
        self.input_shape = params['input_shape']
        self.patch_size = params['patch_size']

        # self.down_input_lower_1 = nn.Sequential(
        #         nn.Conv3d(in_channels=params['in_channels'], out_channels=4*params['out_channels'],
        #                   kernel_size=(1,1,1), padding=0, stride=1),
        #         nn.AvgPool3d(kernel_size=(8,8,8), stride=4, padding=0),
        #         nn.GroupNorm(num_groups=4, num_channels=4*params['out_channels']),
        #         nn.PReLU()
        #     )

        self.down_input_lower_1 = nn.Sequential(
                nn.Conv3d(in_channels=params['in_channels'], out_channels=2*params['out_channels'],
                          kernel_size=(2,2,2), padding=0, stride=2),
                nn.GroupNorm(num_groups=4, num_channels=2*params['out_channels']),
                nn.PReLU()
            )

        # self.norm_add = nn.Sequential(nn.GroupNorm(num_groups=4, num_channels=4*params['out_channels']),
        #         nn.PReLU()
        #     )

        # in_channels: 16, out_channels: 16
        self.encoder_block_1 = EncoderBlock(params)

        params['input'] = False
        params['create_layer_1'] = True
        params['in_channels'] = params['out_channels'] * 2 # 32
        params['out_channels'] = params['out_channels'] * 2 # 32
        self.encoder_block_2 = EncoderBlock(params)

        params['create_layer_2'] = True
        params['in_channels'] = params['out_channels'] * 2 # 64
        params['out_channels'] = params['out_channels'] * 2 # 64
        self.encoder_block_3 = EncoderBlock(params)

        params['in_channels'] = params['out_channels'] * 2 # 128
        params['out_channels'] = params['out_channels'] * 2 # 128
        self.encoder_block_4 = EncoderBlock(params)

        params['in_channels'] = params['out_channels'] * 2 # 256
        params['out_channels'] = int(params['out_channels'] * 2) # 256
        self.bottleneck_block = BottleNeck(params)

        enc_channels = 128
        params['in_channels'] = params['out_channels'] + enc_channels # 256 + 128
        params['out_channels'] = params['out_channels'] # 256
        self.decoder_block_4 = DecoderBlock(params)

        enc_channels = int(enc_channels/2)
        params['in_channels'] = int(params['out_channels']/2) + enc_channels # 128 + 64
        params['out_channels'] = int(params['out_channels'] / 2)  # 128
        self.decoder_block_3 = DecoderBlock(params)

        enc_channels = int(enc_channels/2)
        params['in_channels'] = int(params['out_channels'] / 2) + enc_channels  # 64 + 32
        params['out_channels'] = int(params['out_channels'] / 2) # 64
        params['create_layer_2'] = False
        self.decoder_block_2 = DecoderBlockTransML(params)

        self.output_block_aux = nn.Conv3d(in_channels=params['out_channels'], out_channels=params['num_classes'],
                                      kernel_size = (1, 1, 1), stride = 1, padding = 0)

        enc_channels = int(enc_channels/2)
        params['in_channels'] = int(params['out_channels'] / 2) + enc_channels # 32 + 16
        params['out_channels'] = int(params['out_channels'] / 2) # 32
        params['create_layer_1'] = False
        params['out'] = True
        self.decoder_block_1 = DecoderBlock(params)
        params['out'] = False

        self.output_block_main = nn.Conv3d(in_channels=params['out_channels'], out_channels=params['num_classes'],
                                      kernel_size = (1, 1, 1), stride = 1, padding = 0)

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)


    def train_forward(self, x):

        # Generate Random Coordinates
        self.coords = make_rand_coords(self.input_shape, self.patch_size)

        # Create patching parameters for low res and high res patches
        coords_lower = [int(c/2) for c in self.coords]
        patch_lower  = [int(p/2) for p in self.patch_size]

        # Generate 4X downsampled patches
        x_lower = self.down_input_lower_1(x)
        # x_lower = self.down_input_lower_2(x_lower)

        # Cropped Patch for upper part of encoder
        x_upper = x[..., self.coords[0]:self.coords[0] + self.patch_size[0],
                  self.coords[1]:self.coords[1] + self.patch_size[1],
                  self.coords[2]:self.coords[2] + self.patch_size[2]]

        # Run upper encoder with cropped input
        x_upper_res_1, x_down_1 = self.encoder_block_1(x_upper)

        # Location aware addition for feature map integration
        x_lower[..., coords_lower[0]:coords_lower[0] + patch_lower[0],
                     coords_lower[1]:coords_lower[1] + patch_lower[1],
                     coords_lower[2]:coords_lower[2] + patch_lower[2]] =\
        x_lower[..., coords_lower[0]:coords_lower[0] + patch_lower[0],
            coords_lower[1]:coords_lower[1] + patch_lower[1],
            coords_lower[2]:coords_lower[2] + patch_lower[2]] + x_down_1

        # x_lower = self.norm_add(x_lower)

        # Running lower encoder, bottleneck and decoder with
        x_upper_res_2, x_down_2 = self.encoder_block_2(x_lower)
        x_lower_res_3, x_down_3 = self.encoder_block_3(x_down_2)
        x_lower_res_4, x_down_4 = self.encoder_block_4(x_down_3)

        x_bottleneck = self.bottleneck_block(x_down_4)

        x_up_4 = self.decoder_block_4(x_lower_res_4, x_bottleneck)
        x_up_3 = self.decoder_block_3(x_lower_res_3, x_up_4)
        x_dec_2, x_up_2 = self.decoder_block_2(x_upper_res_2, x_up_3, coords_lower, patch_lower)
        out_aux = self.output_block_aux(x_dec_2)
        x_last = self.decoder_block_1(x_upper_res_1, x_up_2)

        out_main = self.output_block_main(x_last)

        return out_aux, out_main


    def eval_forward(self, x):

        x_upper_res_1, x_down_1 = self.encoder_block_1(x)
        x_upper_res_2, x_down_2 = self.encoder_block_2(x_down_1)
        x_lower_res_3, x_down_3 = self.encoder_block_3(x_down_2)
        x_lower_res_4, x_down_4 = self.encoder_block_4(x_down_3)

        x_bottleneck = self.bottleneck_block(x_down_4)

        x_up_4 = self.decoder_block_4(x_lower_res_4, x_bottleneck)
        x_up_3 = self.decoder_block_3(x_lower_res_3, x_up_4)
        x_up_2 = self.decoder_block_2(x_upper_res_2, x_up_3, None, None)
        x_last = self.decoder_block_1(x_upper_res_1, x_up_2)

        out = self.output_block_main(x_last)
        return out


if __name__ == "__main__":
    params = {'in_channels': 1,
              'out_channels': 16,
              'create_layer_1': False,
              'create_layer_2': False,
              'kernel_size': (5, 5, 5),
              'input_shape': (64,64,64),
              'patch_size': (32,32,32),
              'num_classes': 79,
              'out': False,
              'input': True
              }

    m = MultiResVNet(params=params).cuda()
    # m.eval()
    # m = CompetitiveEncoderBlockInput(params=params).cuda()
    from torchsummary import summary
    # print(m)
    summary(m, input_size=(1,64,64,64))