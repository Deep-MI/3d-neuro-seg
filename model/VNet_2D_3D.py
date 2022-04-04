import torch
import torch.nn as nn

class SliceContextBlock(nn.Module):


    def __init__(self, inp_ch, intra_ch, kernel_size, padding):

        super(SliceContextBlock, self).__init__()

        padding3D = padding
        paddingXY = (padding[0], padding[1], 0)
        paddingYZ = (0, padding[1], padding[2])
        paddingXZ = (padding[0], 0, padding[2])
        kernel_sizeXY = (kernel_size[0], kernel_size[1], 1)
        kernel_sizeYZ = (1, kernel_size[1], kernel_size[2])
        kernel_sizeXZ = (kernel_size[0], 1, kernel_size[2])

        self.block3D = nn.Sequential(nn.Conv3d(inp_ch, int(3*intra_ch/4), kernel_size=kernel_size, padding=padding3D),
                        nn.GroupNorm(4, num_channels=int(3*intra_ch/4)), nn.PReLU())

        self.blockXY = nn.Sequential(nn.Conv3d(inp_ch, int(intra_ch/4), kernel_size=kernel_sizeXY, padding=paddingXY),
                        nn.GroupNorm(4, num_channels=int(intra_ch/4)), nn.PReLU())
        self.blockYZ = nn.Sequential(nn.Conv3d(inp_ch, int(intra_ch/4), kernel_size=kernel_sizeYZ, padding=paddingYZ),
                        nn.GroupNorm(4, num_channels=int(intra_ch/4)), nn.PReLU())
        self.blockXZ = nn.Sequential(nn.Conv3d(inp_ch, int(intra_ch/4), kernel_size=kernel_sizeXZ, padding=paddingXZ),
                        nn.GroupNorm(4, num_channels=int(intra_ch/4)), nn.PReLU())

        #Example of 2d and 3D channel split of 50% each
        # self.block3D = nn.Sequential(nn.Conv3d(inp_ch, int(intra_ch/2), kernel_size=kernel_size, padding=padding3D,
        #                                        dilation=dilation),
        #                 nn.GroupNorm(4, num_channels=int(intra_ch/2)), nn.PReLU())
        #
        # self.blockXY = nn.Sequential(nn.Conv3d(inp_ch, int(intra_ch/2), kernel_size=kernel_sizeXY, padding=paddingXY,
        #                                        dilation=dilation),
        #                 nn.GroupNorm(4, num_channels=int(intra_ch/2)), nn.PReLU())
        # self.blockYZ = nn.Sequential(nn.Conv3d(inp_ch, int(intra_ch/2), kernel_size=kernel_sizeYZ, padding=paddingYZ,
        #                                        dilation=dilation),
        #                 nn.GroupNorm(4, num_channels=int(intra_ch/2)), nn.PReLU())
        # self.blockXZ = nn.Sequential(nn.Conv3d(inp_ch, int(intra_ch/2), kernel_size=kernel_sizeXZ, padding=paddingXZ,
        #                                        dilation=dilation),
        #                 nn.GroupNorm(4, num_channels=int(intra_ch/2)), nn.PReLU())



    def forward(self, x):

        x_3D = self.block3D(x)
        x_XY = self.blockXY(x)
        x_YZ = self.blockYZ(x)
        x_XZ = self.blockXZ(x)

        x_slice = x_XY + x_YZ + x_XZ

        x = torch.cat((x_3D,x_slice),dim=1)
        return x


class Block(nn.Module):
    """
    Simple VNet/3DUNet styled blocks.
    """

    def __init__(self, params):

        super(Block, self).__init__()

        padding_x = int((params['kernel_size'][0] - 1) / 2)
        padding_y = int((params['kernel_size'][1] - 1) / 2)
        padding_z = int((params['kernel_size'][2] - 1) / 2)

        self.conv0 = SliceContextBlock(inp_ch=params['in_channels'], intra_ch=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z))

        if params['create_layer_1']:
            self.conv1 = SliceContextBlock(inp_ch=params['out_channels'], intra_ch=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z))

        if params['create_layer_2']:
            self.conv2 = SliceContextBlock(inp_ch=params['out_channels'], intra_ch=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z))


    def forward(self, x):

        x = self.conv0(x)

        try:
            x = self.conv1(x)
        except:
            pass

        try:
            x = self.conv2(x)
        except:
            pass

        return x


class EncoderBlock(nn.Module):

    def __init__(self, params):

        super(EncoderBlock, self).__init__()

        if params['input']:
            self.conv_input = nn.Sequential(
                nn.Conv3d(in_channels=params['in_channels'], out_channels=params['out_channels'],
                          kernel_size=(1,1,1), padding=0, stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
            params['in_channels'] = params['out_channels']
        self.encoder_block = Block(params)
        self.down_block = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=2*params['out_channels'],
                          kernel_size=(2,2,2), padding=0, stride=2),
                nn.GroupNorm(num_groups=4, num_channels=2*params['out_channels']),
                nn.PReLU()
            )

    def forward(self, x):

        try:
            x = self.conv_input(x)
        except:
            pass

        x_res = self.encoder_block(x)
        x_res = x_res + x
        x_down = self.down_block(x_res)
        # print(x_down.shape, x_res.shape)
        return  x_res, x_down


class DecoderBlock(nn.Module):

    def __init__(self, params):

        super(DecoderBlock, self).__init__()
        # print(params)
        self.decoder_block = Block(params)

        if not params['out']:
            self.up_block = nn.Sequential(
                    nn.ConvTranspose3d(in_channels=params['out_channels'], out_channels=int(params['out_channels']/2),
                              kernel_size=(2,2,2), padding=0, stride=2),
                    nn.GroupNorm(num_groups=4, num_channels=int(params['out_channels']/2)),
                    nn.PReLU()
                )

    def forward(self, x_res, x_up):

        x = torch.cat((x_res, x_up), dim=1)
        x = self.decoder_block(x)
        x1 = x + x_up
        # print(x1.shape)
        try:
            x1 = self.up_block(x1)
        except:
            pass
        return x1


class BottleNeck(nn.Module):

    def __init__(self, params):

        super(BottleNeck, self).__init__()

        if params['input']:
            self.conv_input = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=(1, 1, 1), padding=0, stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
        # params['in_channels'] = params['out_channels']
        self.encoder_block = Block(params)
        self.up_block = nn.Sequential(
                nn.ConvTranspose3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=(2,2,2), padding=0, stride=2),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )

    def forward(self, x):

        try:
            x = self.conv_input(x)
        except:
            pass

        x_res = self.encoder_block(x)
        x_res = x_res + x
        x_up = self.up_block(x_res)
        return x_up


if __name__ == "__main__":
    params = {'in_channels': 1,
              'out_channels': 16,
              'create_layer_1': True,
              'create_layer_2': True,
              'kernel_size': (5, 5, 5),
              'num_classes': 55,
              'input': True,
              'out': True
              }

    m = EncoderBlock(params=params).cuda()
    # m = CompetitiveEncoderBlockInput(params=params).cuda()
    from torchsummary import summary
    print(m)
    summary(m, input_size=(1,32,32,32))