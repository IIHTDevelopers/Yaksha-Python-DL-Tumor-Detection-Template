import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
    

    ## Write the code to first flatten the input and target tensors, then calculate dice loss and BCE loss and return their sum
    def forward(self, inputs, targets, smooth=1):
        ## Write code here and return type will be a tensor
        pass