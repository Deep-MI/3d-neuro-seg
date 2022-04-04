import torch
import torch.nn as nn


class BlockE(nn.Module):
    """
    Dense VNet/3DUNet styled Encoder blocks.
    """

    def __init__(self, params):
        """
        Args:
            params:
                 kernel_size
                 in_channels
                 out_channels
                 create_layer_1 2nd Conv3d Module in Block
                 create_layer_2 3rd Conv3d Module in Block
        """

        super(BlockE, self).__init__()

        padding_x = int((params['kernel_size'][0] - 1) / 2)
        padding_y = int((params['kernel_size'][1] - 1) / 2)
        padding_z = int((params['kernel_size'][2] - 1) / 2)

        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=params['in_channels'], out_channels=params['out_channels'],
                      kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                      stride=1),
            nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
            nn.PReLU()
        )

        self.res_norm_prelu1 = nn.Sequential(nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                                             nn.PReLU())

        if params['create_layer_1']:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                          stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
            self.res_norm_prelu2 = nn.Sequential(nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                                                 nn.PReLU())

        if params['create_layer_2']:
            self.conv2 = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                          stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
            self.res_norm_prelu3 = nn.Sequential(nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                                                 nn.PReLU())

    def forward(self, x):

        x1 = self.conv0(x)
        x_next = self.res_norm_prelu1(x + x1)

        try:
            x2 = self.conv1(x_next)
            x_next = self.res_norm_prelu2(x + x1 + x2)
        except:
            pass

        try:
            x3 = self.conv2(x_next)
            x = self.res_norm_prelu3(x + x1 + x2 + x3)
        except:
            pass

        return x



class BlockD(nn.Module):
    """
    Dense VNet/3DUNet styled Decoder blocks.
    """

    def __init__(self, params):
        """

        Args:
            params:
                 kernel_size
                 in_channels
                 out_channels
                 create_layer_1 2nd Conv3d Module in Block
                 create_layer_2 3rd Conv3d Module in Block
        """
        super(BlockD, self).__init__()

        padding_x = int((params['kernel_size'][0] - 1) / 2)
        padding_y = int((params['kernel_size'][1] - 1) / 2)
        padding_z = int((params['kernel_size'][2] - 1) / 2)

        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=params['in_channels'], out_channels=params['out_channels'],
                      kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                      stride=1),
            nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
            nn.PReLU()
        )

        self.res_norm_prelu1 = nn.Sequential(nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                                             nn.PReLU())

        if params['create_layer_1']:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                          stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
            self.res_norm_prelu2 = nn.Sequential(nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                                                 nn.PReLU())

        if params['create_layer_2']:
            self.conv2 = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                          stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
            self.res_norm_prelu3 = nn.Sequential(nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                                                 nn.PReLU())

    def forward(self, x, xup):

        x = self.conv0(x)
        x_next = self.res_norm_prelu1(xup + x)

        try:
            x1 = self.conv1(x_next)
            x_next = self.res_norm_prelu2(xup + x + x1)
        except:
            pass

        try:
            x2 = self.conv2(x_next)
            x = self.res_norm_prelu3(xup + x + x1 + x2)
        except:
            pass

        return x



class EncoderBlock(nn.Module):
    """
    EncoderBlock in VNet style

    optional input filter Conv3d(1-cubed, channelsize from in_channels to out_channels)
    Block
    Conv3d strided downsampling
    """
    def __init__(self, params):
        """

        Args:
            params:
                input
                in_channels
                out_channels
        """
        super(EncoderBlock, self).__init__()

        if params['input']:
            self.conv_input = nn.Sequential(
                nn.Conv3d(in_channels=params['in_channels'], out_channels=params['out_channels'],
                          kernel_size=(1, 1, 1), padding=0, stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
            params['in_channels'] = params['out_channels']
        self.encoder_block = BlockE(params)
        self.down_block = nn.Sequential(
            nn.Conv3d(in_channels=params['out_channels'], out_channels=2 * params['out_channels'],
                      kernel_size=(2, 2, 2), padding=0, stride=2),
            nn.GroupNorm(num_groups=4, num_channels=2 * params['out_channels']),
            nn.PReLU()
        )

    def forward(self, x):

        try:
            x = self.conv_input(x)
        except:
            pass

        x_res = self.encoder_block(x)
        # x_res = x_res + x     # Dense block already takes care of residual addition in this cases
        x_down = self.down_block(x_res)
        # print(x_down.shape, x_res.shape)
        return x_res, x_down


class DecoderBlock(nn.Module):
    """
    DecoderBlock in VNet style

    2 Inputs (Residual from encoder side, previous layer input)
    Block
    Conv3d strided upsampling
    """
    def __init__(self, params):
        """

       Args:
           params:
               input
               in_channels
               out_channels
       """
        super(DecoderBlock, self).__init__()
        # print(params)

        self.decoder_block = BlockD(params)

        if not params['out']:
            self.up_block = nn.Sequential(
                nn.ConvTranspose3d(in_channels=params['out_channels'], out_channels=int(params['out_channels'] / 2),
                                   kernel_size=(2, 2, 2), padding=0, stride=2),
                nn.GroupNorm(num_groups=4, num_channels=int(params['out_channels'] / 2)),
                nn.PReLU()
            )

    def forward(self, x_res, x_up):

        x = torch.cat((x_res, x_up), dim=1)
        x = self.decoder_block(x, x_up)
        # x1 = x + x_up # No need here
        # print(x1.shape)
        try:
            x1 = self.up_block(x)
        except:
            x1 = x
        return x1


class BottleNeck(nn.Module):
    """
    Bottleneck layer as in VNet

    1 Input
    Block,
    Conv Strided upsampling
    """
    def __init__(self, params):
        """
          Args:
              params:
                  input
                  in_channels
                  out_channels
        """
        super(BottleNeck, self).__init__()

        if params['input']:
            self.conv_input = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=(1, 1, 1), padding=0, stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
        # params['in_channels'] = params['out_channels']
        self.encoder_block = BlockE(params)
        self.up_block = nn.Sequential(
            nn.ConvTranspose3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                               kernel_size=(2, 2, 2), padding=0, stride=2),
            nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
            nn.PReLU()
        )

    def forward(self, x):

        try:
            x = self.conv_input(x)
        except:
            pass

        x_res = self.encoder_block(x)
        # x_res = x_res + x
        x_up = self.up_block(x_res)
        return x_up


if __name__ == "__main__":
    params = {'in_channels': 1,
              'out_channels': 16,
              'create_layer_1': True,
              'create_layer_2': True,
              'kernel_size': (5, 5, 5),
              'num_classes': 79,
              'input': True,
              'out': True
              }

    m = EncoderBlock(params=params).cuda()
    # m = CompetitiveEncoderBlockInput(params=params).cuda()
    from torchsummary import summary

    print(m)
    summary(m, input_size=(1, 64, 64, 64))