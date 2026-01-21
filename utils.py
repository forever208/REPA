import os
from torchvision.datasets.utils import download_url
import torch
import torchvision.models as torchvision_models
import timm
from models import mocov3_vit
import math
import warnings
import torch.nn.functional as F


# code from SiT repository
pretrained_models = {'last.pt'}

def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&st=apcc645o&dl=0'
        download_url(web_path, 'pretrained_models', filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model

def fix_mocov3_state_dict(state_dict):
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict

@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'mocov3':
            if architecture == 'vit':
                if model_config == 's':
                    encoder = mocov3_vit.vit_small()
                elif model_config == 'b':
                    encoder = mocov3_vit.vit_base()
                elif model_config == 'l':
                    encoder = mocov3_vit.vit_large()
                ckpt = torch.load(f'./ckpts/mocov3_vit{model_config}.pth')
                state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
                del encoder.head
                encoder.load_state_dict(state_dict, strict=True)
                encoder.head = torch.nn.Identity()
            elif architecture == 'resnet':
                raise NotImplementedError()
 
            encoder = encoder.to(device)
            encoder.eval()

        elif 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
        
        elif 'dinov1' == encoder_type:
            import timm
            from models import dinov1
            encoder = dinov1.vit_base()
            ckpt =  torch.load(f'./ckpts/dinov1_vit{model_config}.pth') 
            if 'pos_embed' in ckpt.keys():
                ckpt['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    ckpt['pos_embed'], [16, 16],
                )
            del encoder.head
            encoder.head = torch.nn.Identity()
            encoder.load_state_dict(ckpt, strict=True)
            encoder = encoder.to(device)
            encoder.forward_features = encoder.forward
            encoder.eval()

        elif encoder_type == 'clip':
            import clip
            from models.clip_vit import UpdatedVisionTransformer
            encoder_ = clip.load(f"ViT-{model_config}/14", device='cpu')[0].visual
            encoder = UpdatedVisionTransformer(encoder_).to(device)
             #.to(device)
            encoder.embed_dim = encoder.model.transformer.width
            encoder.forward_features = encoder.forward
            encoder.eval()
        
        elif encoder_type == 'mae':
            from models.mae_vit import vit_large_patch16
            import timm
            kwargs = dict(img_size=256)
            encoder = vit_large_patch16(**kwargs).to(device)
            with open(f"ckpts/mae_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )

        elif encoder_type == 'jepa':
            from models.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs).to(device)
            with open(f"ckpts/ijepa_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f, map_location=device)
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def load_legacy_checkpoints(state_dict, encoder_depth):
    new_state_dict = dict()
    for key, value in state_dict.items():
        if 'decoder_blocks' in key:
            parts =key.split('.')
            new_idx = int(parts[1]) + encoder_depth
            parts[0] = 'blocks'
            parts[1] = str(new_idx)
            new_key = '.'.join(parts)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def gaussian_kernel2d(kernel_size: int, sigma: float, device=None, dtype=None):
    assert kernel_size % 2 == 1, "kernel_size should be odd"
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, kernel_size: int = 7, sigma: float = 1.2):
    """
    x: (B,C,H,W) -> blurred x: (B,C,H,W)
    """
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()

    B, C, H, W = x.shape
    k = gaussian_kernel2d(kernel_size, sigma, device=x.device, dtype=x.dtype)
    k = k.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)  # (C,1,K,K)
    pad = kernel_size // 2
    return F.conv2d(x, k, padding=pad, groups=C)


def downsample_to(x: torch.Tensor, size_hw: tuple[int, int]):
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()

    return F.interpolate(x, size=size_hw, mode="bicubic", align_corners=False)


def rmsc(x: torch.Tensor, patch_sz: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """
    RMS Spatial Contrast (RMSC) where each patch token is formed by *stacking*
    a (patch_sz x patch_sz) block into the channel dimension (no pooling).

    Args:
        x: Tensor (B, C, H, W)
        patch_sz: 1, 2, 4, ... must divide H and W
        eps: numerical stability

    Returns:
        Tensor (B,)
    """
    if x.dim() != 4:
        raise ValueError(f"x must be 4D (B,C,H,W), got {tuple(x.shape)}")
    if patch_sz <= 0:
        raise ValueError(f"patch_sz must be positive, got {patch_sz}")

    B, C, H, W = x.shape
    if (H % patch_sz) != 0 or (W % patch_sz) != 0:
        raise ValueError(f"patch_sz={patch_sz} must divide H and W, got H={H}, W={W}")

    Hp, Wp = H // patch_sz, W // patch_sz
    T = Hp * Wp

    if patch_sz == 1:
        # tokens: (B, C, T)
        xt = x.reshape(B, C, T)
    else:
        # Split into non-overlapping patches and stack patch pixels into channel dim:
        # (B, C, H, W)
        # -> (B, C, Hp, patch_sz, Wp, patch_sz)
        # -> (B, Hp, Wp, C, patch_sz, patch_sz)
        # -> (B, Hp*Wp, C*patch_sz*patch_sz)
        # -> (B, C*patch_sz*patch_sz, T)
        xt = (
            x.view(B, C, Hp, patch_sz, Wp, patch_sz)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B, T, C * patch_sz * patch_sz)
             .transpose(1, 2)
        )

    # L2-normalize each token vector over "channel" dim: (B, C', T)
    xt_hat = xt / xt.norm(p=2, dim=1, keepdim=True).clamp_min(eps)

    # Mean of normalized tokens across spatial locations: (B, C', 1)
    x_bar = xt_hat.mean(dim=2, keepdim=True)

    # Per-token squared L2 distance to mean: (B, T)
    sq_dist = (xt_hat - x_bar).pow(2).sum(dim=1)

    # RMSC per sample: (B,)
    return torch.sqrt(sq_dist.mean(dim=1).clamp_min(eps))


def rmsc_tokens(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    RMS Spatial Contrast (RMSC) for token features that are already flattened.

    This matches your previous definition (with per-token L2 normalization):
        u_t = x_t / ||x_t||_2
        RMSC = sqrt( (1/T) * sum_t ||u_t - mean(u)||_2^2 )

    Args:
        x: Tensor of shape (B, T, C) where
           B = batch, T = num_tokens, C = channels/features per token.
        eps: numerical stability.

    Returns:
        Tensor of shape (B,)
    """
    if x.dim() != 3:
        raise ValueError(f"x must be 3D (B,T,C), got {tuple(x.shape)}")

    # (B, T, C)
    # L2-normalize each token over channel dim
    x_hat = x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

    # Mean token over spatial tokens: (B, 1, C)
    x_bar = x_hat.mean(dim=1, keepdim=True)

    # Per-token squared L2 distance to mean: (B, T)
    sq_dist = (x_hat - x_bar).pow(2).sum(dim=-1)

    # RMSC per sample: (B,)
    return torch.sqrt(sq_dist.mean(dim=1).clamp_min(eps))


"""Spatial normalization for vision encoder features."""
ALL_SPNORM_METHODS = ["none", "zscore"]

def spatial_zscore(feat: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score normalization along spatial dimension.

    Args:
        feat: (B, T, D) patch tokens
        alpha: scaling factor for mean subtraction (default 1.0)
        eps: small constant for numerical stability

    Returns:
        Normalized features (B, T, D)
    """
    mean = feat.mean(dim=1, keepdim=True)
    std = feat.std(dim=1, keepdim=True)
    return (feat - alpha * mean) / (std + eps)


class SpatialNormalization:
    """
    Spatial normalization wrapper for backward compatibility.

    Only supports "none" and "zscore" methods.
    """
    def __init__(self, method: str, *, eps: float = 1e-6):
        assert method in ALL_SPNORM_METHODS, f"Invalid method: {method}. Must be one of {ALL_SPNORM_METHODS}"
        self.method = method
        self.eps = eps

    def __call__(self, feat: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.method == "none":
            return feat
        alpha = kwargs.get('zscore_alpha', 1.0)
        return spatial_zscore(feat, alpha=alpha, eps=self.eps)


def tokens_to_grid(z: torch.Tensor) -> torch.Tensor:
    # z: (B, T, D) with T=H*W
    B, T, D = z.shape
    H = W = int(math.isqrt(T))
    assert H * W == T, f"T must be square, got T={T}"
    return z.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)


def grid_to_tokens(g: torch.Tensor) -> torch.Tensor:
    # g: (B, D, H, W)
    B, D, H, W = g.shape
    return g.permute(0, 2, 3, 1).reshape(B, H * W, D).contiguous()


def _gaussian_kernel2d(k: int, sigma: float, device, dtype):
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_depthwise(x: torch.Tensor, k: int, sigma: float) -> torch.Tensor:
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    kernel = _gaussian_kernel2d(k, sigma, x.device, x.dtype)  # (k,k)
    weight = kernel.view(1, 1, k, k).repeat(C, 1, 1, 1)       # (C,1,k,k)
    pad = k // 2
    return F.conv2d(x, weight, padding=pad, groups=C)


def dog_tokens(z: torch.Tensor, sigma1: float = 0.8, sigma2: float = 1.6,
               k1: int = 7, k2: int = 11) -> torch.Tensor:
    """
    DoG band-pass for tokens: blur(sigma1) - blur(sigma2).
    z: (B, T, D)
    returns: (B, T, D)
    """
    g = tokens_to_grid(z)  # (B, D, 16, 16)
    b1 = gaussian_blur_depthwise(g, k1, sigma1)
    b2 = gaussian_blur_depthwise(g, k2, sigma2)
    return grid_to_tokens(b1 - b2)


def gaussian_blur_tokens(z: torch.Tensor, sigma2: float = 1.6, k2: int = 11) -> torch.Tensor:
    """
    DoG band-pass for tokens: blur(sigma1) - blur(sigma2).
    z: (B, T, D)
    returns: (B, T, D)
    """
    g = tokens_to_grid(z)  # (B, D, 16, 16)
    b2 = gaussian_blur_depthwise(g, k2, sigma2)
    return grid_to_tokens(b2)