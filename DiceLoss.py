import torch.nn as nn

# from https://github.com/pytorch/pytorch/issues/1249
# https://towardsdatascience.com/choosing-and-customizing-loss-functions-for-image-processing-a0e4bf665b0a

# from above ^ said this loss was good for tensor overlap
class DiceLoss(nn.Module):
    '''
        Class to define loss give input, model output and groundtruth
    '''

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))

