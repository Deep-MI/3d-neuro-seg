import torch
import torch.nn as nn


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(4, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(4, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):        # g: gating signal from lower spatial res.,     x: encoder side residual
        # print(g.shape, x.shape)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class Block(nn.Module):
    """
    Simple VNet/3DUNet styled blocks.
    """

    def __init__(self, params):

        super(Block, self).__init__()

        padding_x = int((params['kernel_size'][0] - 1) / 2)
        padding_y = int((params['kernel_size'][1] - 1) / 2)
        padding_z = int((params['kernel_size'][2] - 1) / 2)

        self.conv0 = nn.Sequential(
                        nn.Conv3d(in_channels=params['in_channels'], out_channels=params['out_channels'],
                               kernel_size=params['kernel_size'], padding = (padding_x, padding_y, padding_z),
                               stride=1),
                        nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                        nn.PReLU()
                        )

        if params['create_layer_1']:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                          stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )

        if params['create_layer_2']:
            self.conv2 = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                          stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )


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

        self.attention_block = Attention_block(params['F_g'], params['F_l'], params['F_int'])

    def forward(self, x_res, x_up):

        x_res = self.attention_block(x_up, x_res)
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