from collections import OrderedDict
from typing import Tuple, Union, List
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .clip import clip

# ------------------------------
# Building Blocks
# ------------------------------

def convert_weights_fp32(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class FiLM(nn.Module):
    """Simple implementation of Feature-wise Linear Modulation:
        FiLM(x) = \gamma(z)*x + \beta(z)
    """
    def __init__(
        self, 
        n_features: int, 
        n_channels: int,
        ):
        super(FiLM, self).__init__()
        self.sigma = nn.Linear(n_features, n_channels)
        self.beta = nn.Linear(n_features, n_channels)

    def forward(
        self, 
        z: torch.Tensor,
        ) -> torch.Tensor:
        return self.sigma(z), self.beta(z)
    
class UnetResBlock(nn.Module):
    """Residual Convolutional block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.downsample = in_channels != out_channels
        if self.downsample:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
            self.norm3 = nn.InstanceNorm2d(out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out

class UnetrPrUpBlock(nn.Module):
    """A projection upsampling module for UNETR.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: int = 3,
        ):
        super().__init__()
        self.transp_conv_init = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
        )
        
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    UnetResBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                    ),
                )
                for i in range(num_layer)
            ]
        )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x

class UnetrUpBlock(nn.Module):
    """An upsampling decoding block for UNETR.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        ):
        super().__init__()
        self.transp_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
        )

        self.conv_block = UnetResBlock(
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, inp, skip):
        h, w = skip.shape[-2:]
        inp = F.interpolate(inp, size=(h, w), mode="bilinear", align_corners=True)
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


# ------------------------------
# Model
# ------------------------------

class CLIPUNETR(nn.Module):
    """Segmentation from Prompt with a transformer decoder.
    - Use text embedding as query for attention block.
    - UNETR-like architecture.
    """
    def __init__(
        self,
        name: str = "ViT-B/16",
        channels: List[int] = [64, 128, 256, 512],
        ):
        super().__init__()
        self.name = name
        self.extract_layers = [11, 8, 5, 2]
        
        # CLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, _ = clip.load(name, device=device)
        for p in self.clip.parameters():
            p.requires_grad_(False)
        convert_weights_fp32(self.clip)     # NOTE: Use Float32 for now

        # Forward hook to extract intermediate features
        self.activation = {}

        # FiLM
        self.ln_1 = LayerNorm(self.clip.vision_width)
        self.film_1 = FiLM(self.clip.text_width, self.clip.vision_width)
        for i, layer_i in enumerate(self.extract_layers):
            self.clip.visual.transformer.resblocks[layer_i].register_forward_hook(self.get_activation(str(layer_i)))

        # Additional Encoders
        self.encoder1 = UnetResBlock(3, channels[0])
        self.encoder2 = UnetrPrUpBlock(self.clip.vision_width, channels[1], num_layer=2)
        self.encoder3 = UnetrPrUpBlock(self.clip.vision_width, channels[2], num_layer=1)
        self.encoder4 = UnetrPrUpBlock(self.clip.vision_width, channels[3], num_layer=0)

        # Decoders
        self.decoder5 = UnetrUpBlock(self.clip.vision_width, channels[3])
        self.decoder4 = UnetrUpBlock(channels[3], channels[2])
        self.decoder3 = UnetrUpBlock(channels[2], channels[1])
        self.decoder2 = UnetrUpBlock(channels[1], channels[0])

        # Sides
        self.side1 = nn.Conv2d(channels[0], 1, 3, padding=1)
        self.side2 = nn.Conv2d(channels[1], 1, 3, padding=1)
        self.side3 = nn.Conv2d(channels[2], 1, 3, padding=1)
        self.side4 = nn.Conv2d(channels[3], 1, 3, padding=1)
        self.side5 = nn.Conv2d(self.clip.vision_width, 1, 3, padding=1)
        self.outconv = nn.Conv2d(5, 1, 1)


    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def proj_feat(self, x):
        L, B, C = x.shape
        R = int(math.sqrt(L - 1))
        
        # Remove [CLS]
        x = x[1:, :, :]

        # To 2D feature maps
        x = x.view(R, R, B, C)
        x = x.permute(2, 3, 0, 1).contiguous()
        return x

    def forward(
        self,
        x_in,
        z_in,
        ):
        # Forward CLIP Text
        z = self.clip.encode_text(z_in)
        z = z.unsqueeze(1).permute(1, 0, 2)

        # Forward CLIP ViT
        _ = self.clip.encode_image(x_in)
        x = self.activation[str(self.extract_layers[0])]

        # Cond: P(x | z)
        gamma1, beta1 = self.film_1(z)
        x = gamma1 * x + beta1

        # Encoders
        enc1 = self.encoder1(x_in)

        x2 = self.activation[str(self.extract_layers[3])]
        enc2 = self.encoder2(self.proj_feat(x2))
        
        x3 = self.activation[str(self.extract_layers[2])]
        enc3 = self.encoder3(self.proj_feat(x3))

        x4 = self.activation[str(self.extract_layers[1])]
        enc4 = self.encoder4(self.proj_feat(x4))

        # Decoders
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        # Sides
        d1 = self.side1(out)
        h, w = x_in.shape[-2:]

        d5 = self.side5(dec4)
        d5 = F.interpolate(d5, size=(h, w), mode='bilinear', align_corners=True)

        d4 = self.side4(dec3)
        d4 = F.interpolate(d4, size=(h, w), mode='bilinear', align_corners=True)

        d3 = self.side3(dec2)
        d3 = F.interpolate(d3, size=(h, w), mode='bilinear', align_corners=True)

        d2 = self.side2(dec1)
        d2 = F.interpolate(d2, size=(h, w), mode='bilinear', align_corners=True)

        d0 = self.outconv(torch.cat([d5, d4, d3, d2, d1], dim=1))
        outputs = [d5, d4, d3, d2, d1, d0]
            
        return [torch.sigmoid(o) for o in outputs]