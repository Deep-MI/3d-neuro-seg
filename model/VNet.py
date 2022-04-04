import torch
import torch.nn as nn


class Block(nn.Module):
    """
    Simple VNet/3DUNet styled blocks.
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

        super(Block, self).__init__()

        padding_x = int((params['kernel_size'][0] - 1) / 2)
        padding_y = int((params['kernel_size'][1] - 1) / 2)
        padding_z = int((params['kernel_size'][2] - 1) / 2)

        # This is block 0
        self.conv0 = nn.Sequential(
                        nn.Conv3d(in_channels=params['in_channels'], out_channels=params['out_channels'],
                               kernel_size=params['kernel_size'], padding = (padding_x, padding_y, padding_z),
                               stride=1),
                        nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                        nn.PReLU()
                        )

        # If not first layer, create block 1
        if params['create_layer_1']:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                          stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
        else:
           self.conv1 = None

        # If not second layer, create block 2
        if params['create_layer_2']:
            self.conv2 = nn.Sequential(
                nn.Conv3d(in_channels=params['out_channels'], out_channels=params['out_channels'],
                          kernel_size=params['kernel_size'], padding=(padding_x, padding_y, padding_z),
                          stride=1),
                nn.GroupNorm(num_groups=4, num_channels=params['out_channels']),
                nn.PReLU()
            )
        else:
           self.conv2 = None


    def forward(self, x):

        x = self.conv0(x)

        if self.conv1 is not None:
            x = self.conv1(x)

        if self.conv2 is not None:
            x = self.conv2(x)

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
        self.decoder_block = Block(params)

        if not params['out']:
            self.up_block = nn.Sequential(
                    nn.ConvTranspose3d(in_channels=params['out_channels'], out_channels=int(params['out_channels']/2),
                              kernel_size=(2,2,2), padding=0, stride=2),
                    nn.GroupNorm(num_groups=4, num_channels=int(params['out_channels']/2)),
                    nn.PReLU()
                )
        else:
            self.up_block = None

    def forward(self, x_res, x_up):

        x = torch.cat((x_res, x_up), dim=1)
        x = self.decoder_block(x)
        x1 = x + x_up
        # print(x1.shape)
        if self.up_block is not None:
            x1 = self.up_block(x1)

        return x1


class DecoderBlockTrans(DecoderBlock):
    """
    Decoder blocks for transition layer in MultiResVNet in lower to upper layer. ONLY defined for MultiResVNet.
    """
    def __init__(self, params):
        """
        Inherits from Decoder Block.
        """
        super(DecoderBlockTrans, self).__init__(params)


    def forward(self, x_res, x_up, coords, patch_size):

        x = torch.cat((x_res, x_up), dim=1)
        x = self.decoder_block(x)
        x1 = x + x_up

        if self.training:
            x1 = x1[..., coords[0]:coords[0] + patch_size[0],
                            coords[1]:coords[1] + patch_size[1],
                            coords[2]:coords[2] + patch_size[2]]

        try:
            x1 = self.up_block(x1)
        except:
            pass
        return x1


class DecoderBlockTransML(DecoderBlock):
    """
    Decoder blocks for transition layer in MultiResVNet in lower to upper layer, with multiple losses. ONLY defined for
    MultiResVNet.
    """
    def __init__(self, params):
        """
        Inherits from Decoder Block
        """
        super(DecoderBlockTransML, self).__init__(params)

    def forward(self, x_res, x_up, coords, patch_size):

        x = torch.cat((x_res, x_up), dim=1)
        x = self.decoder_block(x)
        x_layer = x + x_up

        # only in training, crop the image
        if self.training:
            x_crop = x_layer[..., coords[0]:coords[0] + patch_size[0],
                            coords[1]:coords[1] + patch_size[1],
                            coords[2]:coords[2] + patch_size[2]]

        if self.training:
            x_up = self.up_block(x_crop)
            return x_layer, x_up
        else:
            x_up = self.up_block(x_layer)
            return x_up


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