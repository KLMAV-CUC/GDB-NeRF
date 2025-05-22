import torch.nn as nn
import torch.nn.functional as F
from .ssim_loss import SSIM
from .vgg_perceptual_loss import VGGPerceptualLoss


class PhotometricLoss(nn.Module):
    def __init__(self, weights=[1.0, 0.1, 0.05]):
        super(PhotometricLoss, self).__init__()
        self.alpha, self.beta, self.gamma = weights

        self.ssim = SSIM(window_size=7, channel=3)
        self.perceptual_loss = VGGPerceptualLoss()

    def forward(self, img1, img2):
        mse_loss = F.mse_loss(img1, img2, reduction='mean')
        ssim = self.ssim(img1, img2)
        perceptual_loss = self.perceptual_loss(img1, img2)

        total_loss = self.alpha * mse_loss + self.beta * (1 - ssim) + self.gamma * perceptual_loss
        return total_loss, mse_loss, ssim, perceptual_loss

