'''
PyTorch implementation of weighted Cross Entropy Loss and Dice Loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
    '''
    2D/3D Dice Loss
    DOES NOT INCLUDE SOFTMAX AS A PART. MUST MANUALLY PROVIDE SOFT INPUTS
    '''

    def forward(self, output, target, weights=None, dice_per_channel=False):
        '''
        Inputs:
            -- output: N x C x H x W or N x C x H x W x D Variable
            -- target : N x C x W or .... LongTensor with starting class at 0
            -- weights: C FloatTensor with class wise weights
            -- ignore_index: ignore index from the loss
        '''

        eps = 0.001
        # torch.zeros_like(output)
        encoded_target =  output.detach() * 0 #Creates a tensor with same shape as output with all zeros
        encoded_target.scatter_(1, target, 1)

        if weights is None:
            weights = 1

        # Calculate 3d dice loss
        intersection = output * encoded_target

        intersection = intersection.sum(tuple(range(2, intersection.ndim)))

        numerator = 2 * intersection
        union = output + encoded_target
        denominator = union.sum(tuple(range(2, union.ndim)))

        denominator = denominator + eps

        loss_per_channel = weights * (1 - (numerator / denominator))  # batch/Channel-wise weights
		
		loss_per_channel = loss_per_channel.sum(0) # sum over batch dimension

        # Return per channel loss if needed otherwise only total loss
        if dice_per_channel is True:
            return loss_per_channel, loss_per_channel.sum() / output.size(1)
        else:
            return loss_per_channel.sum() / output.size(1)


class CrossEntropy3D(nn.Module):
    '''
    3D Cross-entropy loss implemented as negative log likelihood
    '''

    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropy3D, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, inputs, targets):
        #
        if inputs.size()!=targets.size(): #ONLY incase of batch_size>1 in Dataloader
            targets = targets.squeeze(0) # Because Dataloader adds an extra dimension
        return self.nll_loss(inputs, targets)



class CombinedLoss(nn.Module):
    '''
    For CrossEntropy the input has to be a long tensor
    Args:
        -- inputx N x C x H x W
        -- target - N x H x W - int type
        -- weight - N x H x W - float
    '''

    def __init__(self, weight_dice=1, weight_ce=1):
        super(CombinedLoss, self).__init__()
        # self.cross_entropy_loss = FocalCrossEntropy()
        self.cross_entropy_loss = CrossEntropy3D()
        #self.thresh_confidence_loss = ThresholdedConfidencePenalty()
        self.dice_loss = DiceLoss()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    # MEMORY:
    # inputx -> nn.out
    # encoded_target in DiceLoss -> 1-hot gt // target -> gt
    # input_soft -> softmax
    # total_loss -> delta  --> volume-dependent loss figure
    # cross-entropy loss???

    def forward(self, inputx, target, weight):
        # train_utils does this before fprop anyway
        #target = target.type(torch.LongTensor)  # Typecast to long tensor
        #if inputx.is_cuda:
        #    target = target.cuda()

        # Target should be of form (1 , 1, size_x, size_y, size_z), ie, NOT one-hot
        if len(target.size()) < 5:
            target = target.unsqueeze(0)

        if self.training is True:
            # Training mode. Returns dice loss
            dice_val = torch.mean(self.dice_loss(F.softmax(inputx, dim=1), target))
            ce_val = torch.mean(torch.mul(self.cross_entropy_loss.forward(inputx, target), weight))
            total_loss = torch.add(torch.mul(dice_val, self.weight_dice), torch.mul(ce_val, self.weight_ce))
            return total_loss, dice_val, ce_val
        else:
            # Eval mode. Returns Dice SCORE instead of dice Loss
            loss_per_channel, total_loss = self.dice_loss(inputx, target, weights=None, dice_per_channel=True)
            dice_score_per_channel = 1 - loss_per_channel  # Probably should have a mean here
            total_dice_score = 1 - total_loss
            return dice_score_per_channel, total_dice_score
