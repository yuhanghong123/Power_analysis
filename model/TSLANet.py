import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from model.utils import random_masking_3D


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.complex_weight_high = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)
        self.adaptive_filter = adaptive_filter

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        # Flattening H and W into a single dimension
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[
            0]  # Compute median
        # Reshape to match the original dimensions
        median_energy = median_energy.view(B, 1)

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        adaptive_mask = ((normalized_energy > self.threshold_param).float(
        ) - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm, use_icb=True, use_asb=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim,
                       hidden_features=mlp_hidden_dim, drop=drop)
        self.ICB = use_icb
        self.ASB = use_asb

    def forward(self, x):
        # Check if both ASB and ICB are true
        if self.ICB and self.ASB:
            x = x + \
                self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif self.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif self.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class TSLANet(nn.Module):

    def __init__(self, patch_size=16, seq_len=128, emb_dim=512, dropout=0.1,
                 depth=6, pred_len=2, mask_ratio=0.15):
        super(TSLANet, self).__init__()

        self.patch_size = patch_size
        self.stride = self.patch_size // 2
        num_patches = int((seq_len - self.patch_size) / self.stride + 1)

        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, emb_dim)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=emb_dim,
                          drop=dropout, drop_path=dpr[i])
            for i in range(depth)]
        )
        self.mask_ratio = mask_ratio
        # Parameters/Embeddings
        self.out_layer = nn.Linear(emb_dim * num_patches, pred_len)

    def pretrain(self, x_in):
        x = rearrange(x_in, 'b l m -> b m l')
        x_patched = x.unfold(
            dimension=-1, size=self.patch_size, step=self.stride)
        x_patched = rearrange(x_patched, 'b m n p -> (b m) n p')

        xb_mask, _, self.mask, _ = random_masking_3D(
            x_patched, mask_ratio=self.mask_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch]
        xb_mask = self.input_layer(xb_mask)

        for tsla_blk in self.tsla_blocks:
            xb_mask = tsla_blk(xb_mask)

        return xb_mask, self.input_layer(x_patched)

    def forward(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,
                           unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x = self.input_layer(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
