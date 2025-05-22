##### the code is borrowed from https://github.com/Po-Hsun-Su/pytorch-ssim #####
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size: int, 
             sigma: float) -> torch.Tensor:
    gauss = torch.arange(window_size, dtype=torch.float32)
    gauss = torch.exp(-((gauss - window_size // 2) ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def create_window(window_size: int, 
                  channel: int) -> torch.Tensor:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    _2D_window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return _2D_window


class SSIM(nn.Module):
    def __init__(self, 
                 window_size: int = 11, 
                 channel: int = 3, 
                 size_average: bool = True) -> None:
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.register_buffer('window', create_window(window_size, self.channel))

    def forward(self, 
                img1: torch.Tensor, 
                img2: torch.Tensor) -> torch.Tensor:
        assert img1.shape == img2.shape, "Input images must have the same dimensions"
        assert img1.shape[1] == self.channel, "Input images must have the same number of channels"
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.square()
        mu2_sq = mu2.square()
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1.square(), self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2.square(), self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean([1, 2, 3])

