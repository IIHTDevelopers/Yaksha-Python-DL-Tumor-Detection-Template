
import torch
import torch.nn as nn

class PolypModel(nn.Module):
    def __init__(self):
        super(PolypModel, self).__init__()

    def forward(self, x):
        pass

# if __name__ == "__main__":
#     model = PolypModel().cuda()
#     from torchsummary import summary
#     summary(model, (3, 512, 512))
