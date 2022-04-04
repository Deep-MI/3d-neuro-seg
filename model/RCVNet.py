import torch
import torch.nn as nn

from numpy.random import random_sample

def make_rand_coords(input_size=(256,256,256), patch_size=(64,64,64)):
    return [get_dims(input_size[0] - patch_size[0]), \
           get_dims(input_size[1] - patch_size[1]), \
           get_dims(input_size[2] - patch_size[2])]


def get_dims(upper):
    # Random value in the range [0, upper)
    return int(upper * random_sample())


def multi_gpu_check(gpu_map, l_name, *args):
    """
    Can move computations to other GPUs if specified. The names of the layers and corresponding GPUs can be specified
    in GPU map.

    :param gpu_map:
    :param l_name:
    :param args:
    :return:
    """
    args = list(args)
    if l_name in gpu_map.keys():
        # print(l_name)
        for idx, l in enumerate(args):
            args[idx] = l.to(torch.device(gpu_map[l_name]))
            # print(args[idx].device, gpu_map[l_name])

    if len(args)==1:
        return args[0]
    return args

class RCVNet(nn.Module):

    """
    Random Cropping VNet. Model is designed to extract patches randomly during feedforward pass unless specifically
    prevented by setting a random patch coordinate manually. Can also move operations for individual layers to different
    GPUs if specified in params
    
    Standard VNet Architecture
    """

    def __init__(self, params):
        """
        Standard VNet Architecture
        """
        super(RCVNet, self).__init__()

        self.coords = None
        self.input_shape = params['input_shape']
        self.patch_size = params['patch_size']
        self.gen_random = params['gen_random']

        # Choose sub model
        if params['sub_model_name'] == 'vnet':
            from model.VNet import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_2d_3d':
            from model.VNet_2D_3D import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_asym':
            from model.VNetAsym import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_sym':
            from model.VNetSym import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_denseadd':
            from model.VNetDenseAdd import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_exclusion':
            from model.VNetExclusion import EncoderBlock, BottleNeck, DecoderBlock
        elif params['sub_model_name'] == 'vnet_se':
            from model.VNetSE import EncoderBlock, BottleNeck, DecoderBlock
        else:
            raise ValueError(f"{params['sub_model_name']} does not exist.")

        # Artefact from another file. Unfortunately was not removed and is a dangling layer in many of the saved models
        # i.e. registered as a parameter, but never used
        self.down_input_lower = nn.Sequential(
                nn.Conv3d(in_channels=params['in_channels'], out_channels=4*params['out_channels'],
                          kernel_size=(4,4,4), padding=0, stride=4),
                nn.GroupNorm(num_groups=4, num_channels=4*params['out_channels']),
                nn.PReLU()
            )

        # Start model creation
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

        self.gpu_map = params['gpu_map']

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)


    def train_forward(self, x):
        """
        Standard VNet Architecture
        """

        # Generate Random Coordinates if needed
        if self.gen_random: # For usage by QuadNet
            self.coords = make_rand_coords(self.input_shape, self.patch_size)
        assert self.coords is not None

        # Cropped Patch for the part of encoder
        x_upper = x[..., self.coords[0]:self.coords[0] + self.patch_size[0],
                  self.coords[1]:self.coords[1] + self.patch_size[1],
                  self.coords[2]:self.coords[2] + self.patch_size[2]]

        # Running Encoder side of network
        x_upper = multi_gpu_check(self.gpu_map, 'encoder_block_1', x_upper)
        x_upper_res_1, x_down_1 = self.encoder_block_1(x_upper)

        x_down_1 = multi_gpu_check(self.gpu_map, 'encoder_block_2', x_down_1)
        x_upper_res_2, x_down_2 = self.encoder_block_2(x_down_1)

        x_down_2 = multi_gpu_check(self.gpu_map, 'encoder_block_3', x_down_2)
        x_lower_res_3, x_down_3 = self.encoder_block_3(x_down_2)

        x_down_3 = multi_gpu_check(self.gpu_map, 'encoder_block_4', x_down_3)
        x_lower_res_4, x_down_4 = self.encoder_block_4(x_down_3)

        # Running bottleneck
        x_down_4 = multi_gpu_check(self.gpu_map, 'bottleneck_block', x_down_4)
        x_bottleneck = self.bottleneck_block(x_down_4)

        # Run decoder
        x_lower_res_4, x_bottleneck = multi_gpu_check(self.gpu_map, 'decoder_block_4', x_lower_res_4, x_bottleneck)
        x_up_4 = self.decoder_block_4(x_lower_res_4, x_bottleneck)

        x_lower_res_3, x_up_4 = multi_gpu_check(self.gpu_map, 'decoder_block_3', x_lower_res_3, x_up_4)
        x_up_3 = self.decoder_block_3(x_lower_res_3, x_up_4)

        x_upper_res_2, x_up_3 = multi_gpu_check(self.gpu_map, 'decoder_block_2', x_upper_res_2, x_up_3)
        x_up_2 = self.decoder_block_2(x_upper_res_2, x_up_3)

        x_upper_res_1, x_up_2 = multi_gpu_check(self.gpu_map, 'decoder_block_1', x_upper_res_1, x_up_2)
        x_last = self.decoder_block_1(x_upper_res_1, x_up_2)

        x_last = multi_gpu_check(self.gpu_map, 'output_block', x_last)
        out = self.output_block(x_last)

        return out


    def eval_forward(self, x):
        """
        Standard VNet Architecture
        """

        # Standard evaluation. No patch extraction
        x_upper = multi_gpu_check(self.gpu_map, 'encoder_block_1', x)
        x_upper_res_1, x_down_1 = self.encoder_block_1(x_upper)

        x_down_1 = multi_gpu_check(self.gpu_map, 'encoder_block_2', x_down_1)
        x_upper_res_2, x_down_2 = self.encoder_block_2(x_down_1)

        x_down_2 = multi_gpu_check(self.gpu_map, 'encoder_block_3', x_down_2)
        x_lower_res_3, x_down_3 = self.encoder_block_3(x_down_2)

        x_down_3 = multi_gpu_check(self.gpu_map, 'encoder_block_4', x_down_3)
        x_lower_res_4, x_down_4 = self.encoder_block_4(x_down_3)

        # Running bottlenext and decoder
        x_down_4 = multi_gpu_check(self.gpu_map, 'bottleneck_block', x_down_4)
        x_bottleneck = self.bottleneck_block(x_down_4)

        x_lower_res_4, x_bottleneck = multi_gpu_check(self.gpu_map, 'decoder_block_4', x_lower_res_4, x_bottleneck)
        x_up_4 = self.decoder_block_4(x_lower_res_4, x_bottleneck)

        x_lower_res_3, x_up_4 = multi_gpu_check(self.gpu_map, 'decoder_block_3', x_lower_res_3, x_up_4)
        x_up_3 = self.decoder_block_3(x_lower_res_3, x_up_4)

        x_upper_res_2, x_up_3 = multi_gpu_check(self.gpu_map, 'decoder_block_2', x_upper_res_2, x_up_3)
        x_up_2 = self.decoder_block_2(x_upper_res_2, x_up_3)

        x_upper_res_1, x_up_2 = multi_gpu_check(self.gpu_map, 'decoder_block_1', x_upper_res_1, x_up_2)
        x_last = self.decoder_block_1(x_upper_res_1, x_up_2)

        x_last = multi_gpu_check(self.gpu_map, 'output_block', x_last)
        out = self.output_block(x_last)

        return out


class RCVNetAttention(RCVNet):

    def __init__(self, params):
        super(RCVNet, self).__init__()

        from model.VNetAttention import EncoderBlock as AttEncoderBlock, BottleNeck as AttBottleNeck, \
            DecoderBlock as AttDecoderBlock

        self.coords = None
        self.input_shape = params['input_shape']
        self.patch_size = params['patch_size']
        self.gen_random = params['gen_random']

        self.down_input_lower = nn.Sequential(
                nn.Conv3d(in_channels=params['in_channels'], out_channels=4*params['out_channels'],
                          kernel_size=(4,4,4), padding=0, stride=4),
                nn.GroupNorm(num_groups=4, num_channels=4*params['out_channels']),
                nn.PReLU()
            )
        # in_channels: 16, out_channels: 16
        self.encoder_block_1 = AttEncoderBlock(params)

        params['input'] = False
        params['create_layer_1'] = True
        params['in_channels'] = params['out_channels'] * 2 # 32
        params['out_channels'] = params['out_channels'] * 2 # 32
        self.encoder_block_2 = AttEncoderBlock(params)

        params['create_layer_2'] = True
        params['in_channels'] = params['out_channels'] * 2 # 64
        params['out_channels'] = params['out_channels'] * 2 # 64
        self.encoder_block_3 = AttEncoderBlock(params)

        params['in_channels'] = params['out_channels'] * 2 # 128
        params['out_channels'] = params['out_channels'] * 2 # 128
        self.encoder_block_4 = AttEncoderBlock(params)

        params['in_channels'] = params['out_channels'] * 2 # 256
        params['out_channels'] = int(params['out_channels'] * 2) # 256
        self.bottleneck_block = AttBottleNeck(params)

        enc_channels = 128
        params['in_channels'] = params['out_channels'] + enc_channels # 256 + 128
        params['F_g'], params['F_l'], params['F_int'] = (256, 128, 128)
        params['out_channels'] = params['out_channels'] # 256
        self.decoder_block_4 = AttDecoderBlock(params)

        enc_channels = int(enc_channels/2)
        params['in_channels'] = int(params['out_channels']/2) + enc_channels # 128 + 64
        params['out_channels'] = int(params['out_channels'] / 2)  # 128
        params['F_g'], params['F_l'], params['F_int'] = (128, 64, 64)
        self.decoder_block_3 = AttDecoderBlock(params)

        enc_channels = int(enc_channels/2)
        params['in_channels'] = int(params['out_channels'] / 2) + enc_channels  # 64 + 32
        params['out_channels'] = int(params['out_channels'] / 2) # 64
        params['F_g'], params['F_l'], params['F_int'] = (64, 32, 32)
        params['create_layer_2'] = False
        self.decoder_block_2 = AttDecoderBlock(params)

        enc_channels = int(enc_channels/2)
        params['in_channels'] = int(params['out_channels'] / 2) + enc_channels # 32 + 16
        params['out_channels'] = int(params['out_channels'] / 2) # 32
        params['F_g'], params['F_l'], params['F_int'] = (32, 16, 16)
        params['create_layer_1'] = False
        params['out'] = True
        self.decoder_block_1 = AttDecoderBlock(params)
        params['out'] = False

        self.output_block = nn.Conv3d(in_channels=params['out_channels'], out_channels=params['num_classes'],
                                      kernel_size = (1, 1, 1), stride = 1, padding = 0)

        self.gpu_map = params['gpu_map']


if __name__ == "__main__":
    # TEST CODE [RUN THIS TO VERIFY MODELS]
    params = {'in_channels': 1,
              'out_channels': 16,
              'create_layer_1': False,
              'create_layer_2': False,
              'kernel_size': (5, 5, 5),
              'input_shape': (64,64,64),
              'patch_size': (64,64,64),
              'num_classes': 40,
              'out': False,
              'input': True,
              # 'F_g': None,
              # 'F_l': None,
              # 'F_int': None
              'gen_random' : True,
              'gpu_map':{}
              }

    m = RCVNet(params=params).cuda()
    # m.eval()
    # m = CompetitiveEncoderBlockInput(params=params).cuda()
    try:
        from torchsummary import summary
        # print([l for l in m.named_children()])
        summary(m, input_size=(1,64,64,64))
    except ImportError:
        pass
    #
    # print([l for l in m.decoder_block_1.parameters()])
    # print([l.device() for _, l in m.named_children()])