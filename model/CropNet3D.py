import torch.nn as nn

from numpy.random import random_sample

from model.VNet import EncoderBlock, BottleNeck, DecoderBlock

def make_rand_coords(input_size=(256,256,256), patch_size=(64,64,64)):
    return [get_dims(input_size[0] - patch_size[0]), \
           get_dims(input_size[1] - patch_size[1]), \
           get_dims(input_size[2] - patch_size[2])]


def get_dims(upper):
    # Random value in the range [0, upper)
    return int(upper * random_sample())


class CropNet3D(nn.Module):

    def __init__(self, params):
        super(CropNet3D, self).__init__()

        self.coords = None
        self.input_shape = params['input_shape']
        self.patch_size = params['patch_size']

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
        #
        # self.down_input_lower_2 = nn.Sequential(
        #         nn.Conv3d(in_channels=params['in_channels'], out_channels=2*params['out_channels'],
        #                   kernel_size=(2,2,2), padding=0, stride=2),
            #         nn.GroupNorm(num_groups=4, num_channels=2*params['out_channels']),
            #         nn.PReLU()
            #     )

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

    def forward(self, x):
        if self.training:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)


    def train_forward(self, x):

        # Generate Random Coordinates
        self.coords = make_rand_coords(self.input_shape, self.patch_size)

        # Run upper network with cropped intput
        x_res_1, x_down_1 = self.encoder_block_1(x)
        x_res_2, x_down_2 = self.encoder_block_2(x_down_1)
        x_res_3, x_down_3 = self.encoder_block_3(x_down_2)
        x_res_4, x_down_4 = self.encoder_block_4(x_down_3)

        x_bottleneck = self.bottleneck_block(x_down_4)

        x_res_4 = self.sync_crop(x_res_4, level=3)
        x_bottleneck = self.sync_crop(x_bottleneck, level=3)
        x_up_4 = self.decoder_block_4(x_res_4, x_bottleneck)

        x_res_3 = self.sync_crop(x_res_3, level=2)
        x_up_3 = self.decoder_block_3(x_res_3, x_up_4)

        x_res_2 = self.sync_crop(x_res_2, level=1)
        x_up_2 = self.decoder_block_2(x_res_2, x_up_3)

        x_res_1 = self.sync_crop(x_res_1, level=0)
        x_last = self.decoder_block_1(x_res_1, x_up_2)

        out = self.output_block(x_last)

        return out

    def eval_forward(self, x):

        x_upper_res_1, x_down_1 = self.encoder_block_1(x)
        x_upper_res_2, x_down_2 = self.encoder_block_2(x_down_1)
        x_lower_res_3, x_down_3 = self.encoder_block_3(x_down_2)
        x_lower_res_4, x_down_4 = self.encoder_block_4(x_down_3)

        x_bottleneck = self.bottleneck_block(x_down_4)

        x_up_4 = self.decoder_block_4(x_lower_res_4, x_bottleneck)
        x_up_3 = self.decoder_block_3(x_lower_res_3, x_up_4)
        x_up_2 = self.decoder_block_2(x_upper_res_2, x_up_3)
        x_last = self.decoder_block_1(x_upper_res_1, x_up_2)

        out = self.output_block(x_last)
        return out

    def sync_crop(self, x, level):
        input_size = [int(i / (2 ** level)) for i in self.coords]
        patch_size = [int(i / (2 ** level)) for i in self.patch_size]

        x = x[..., input_size[0]:input_size[0] + patch_size[0],
            input_size[1]: input_size[1] + patch_size[1],
            input_size[2]: input_size[2] + patch_size[2]
            ]
        return x

if __name__ == "__main__":
    params = {'in_channels': 1,
              'out_channels': 16,
              'create_layer_1': False,
              'create_layer_2': False,
              'kernel_size': (3, 3, 3),
              'input_shape': (64,64,64),
              'patch_size': (32,32,32),
              'num_classes': 40,
              'out': False,
              'input': True
              }

    m = CropNet3D(params=params).cuda()
    m.train()
    # m = CompetitiveEncoderBlockInput(params=params).cuda()
    from torchsummary import summary
    print(m)
    summary(m, input_size=(1,64,64,64))