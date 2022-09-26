import typing as T

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class FamNetOriginalCountRegressor(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super().__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def forward(self, im):
        num_sample = im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))
            if self.pool == 'mean':
                output = torch.mean(output, dim=(0), keepdim=True)
                return output
            elif self.pool == 'max':
                output, _ = torch.max(output, 0, keepdim=True)
                return output
        else:
            for i in range(0, num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=(0), keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0, keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output, output), dim=0)
            return Output


class FamNetDensityEstimator(nn.Sequential):

    def __init__(self, num_sims, pool="mean"):
        super().__init__(
            nn.Conv2d(num_sims, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

        self.pool = pool

    def agglomorate(self, tensor):
        if self.pool == "mean":
            return torch.mean(tensor, dim=1, keepdim=True)
        if self.pool == "max":
            return torch.max(tensor, dim=1, keepdim=True)[0]
        if callable(self.pool):
            return self.pool(tensor)
        raise NotImplementedError(f"Agglomeration function is unknown: {self.pool}")

    def forward(self, input: torch.Tensor):
        batch_size, num_tlbr, num_sims, H, W = input.size()

        # batch_size, num_tlbr, num_sims, H, W # Input
        # batch_size * num_tlbr, num_sims, H, W # collapse
        # batch_size * num_tlbr, 1, H, W # after conv
        # batch_size, num_tlbr, 1, H, W # reshape
        # batch_size, num_tlbr, H, W # swqueeze
        # batch_size, 1, H, W # agglomorate

        input = input.view(batch_size * num_tlbr, num_sims, H, W)
        y = super().forward(input)

        assert y.shape[:2] == (batch_size * num_tlbr, 1)
        nH, nW = y.shape[2:]

        y = y.view(batch_size, num_tlbr, nH, nW)
        y = self.agglomorate(y)

        assert y.shape == (batch_size, 1, nH, nW)
        return y


class FamNetFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        import torchvision.models.resnet as resnet
        model = resnet.resnet50(pretrained=True)

        children = list(model.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

        self.mean = nn.parameter.Parameter(torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]).requires_grad_(False)
        self.std = nn.parameter.Parameter(torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]).requires_grad_(False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.conv2(x)
        map3 = x = self.conv3(x)
        map4 = self.conv4(x)

        return map3, map4


class FamNetSimilarity(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = self.initialize_feature_extractor()

    def initialize_feature_extractor(self) -> T.Callable[[torch.Tensor], T.Iterable[torch.Tensor]]:
        return FamNetFeatureExtractor()

    def resize_image_batch(self, bchw: torch.Tensor, size: T.Tuple[int, int], MODE="bilinear") -> torch.Tensor:
        return F.interpolate(bchw, size=size, mode=MODE)

    def compute_features_maps(self, image: torch.Tensor) -> T.Iterable[torch.Tensor]:
        for feature in self.feature_extractor(image.unsqueeze(0)):
            yield feature.squeeze(0)

    def generate_rescaled_patches(self, patches: torch.Tensor, SCALES=[0.9, 1.1]) -> T.Iterable[torch.Tensor]:
        _, _, H, W = patches.size()
        yield patches

        for scale in SCALES:

            # Scale H_f, if rescaled value (H_fs) is too small then just use original (H_f)
            H_scaled = np.ceil(H * scale)
            H_scaled = H if H_scaled <= 1 else H_scaled
            H_scaled = int(H_scaled)

            # Scale W_f, if rescaled value (W_fs) is too small then just use original (W_f)
            W_scaled = np.ceil(W * scale)
            W_scaled = W if W_scaled <= 1 else W_scaled
            W_scaled = int(W_scaled)

            yield self.resize_image_batch(patches, size=(H_scaled, W_scaled))

    def compute_similarity_scores(self, tensor: torch.Tensor, filter: torch.Tensor):
        _, _, FH, FW = filter.size()
        padded = F.pad(tensor, (
            FW // 2,
            (FW - 1) // 2,
            FH // 2,
            (FH - 1) // 2
        ))
        output = F.conv2d(padded, filter)

        assert tensor.shape[2:] == output.shape[2:]  # Padding is applied is such a way that the input and output tensor has the same size
        return output

    def rescale_box_features(self, tlbr: torch.Tensor, scale: T.Tuple[float, float], bounds: T.Tuple[int, int] = None) -> torch.IntTensor:
        # Separate components
        top = tlbr[:, 0]
        left = tlbr[:, 1]
        bottom = tlbr[:, 2]
        right = tlbr[:, 3]

        # Rescale
        scale_h, scale_w = scale
        top = top * scale_h
        left = left * scale_w
        bottom = bottom * scale_h
        right = right * scale_w

        # Round-off
        top = torch.floor(top)
        left = torch.floor(left)
        bottom = torch.ceil(bottom)
        right = torch.ceil(right)

        # Prevent out-of-bounds
        top = torch.clamp_min(top, 0)
        left = torch.clamp_min(left, 0)
        if bounds is not None:
            max_h, max_w = bounds
            # Inclusive to Exclusive Indexing (i.e. +1) (e.g. <= 100 → <101)
            bottom = torch.clamp_max(bottom + 1, max_h)
            right = torch.clamp_max(right + 1, max_w)

        # Recombine components
        full_tlbr = torch.stack([top, left, bottom, right], dim=1)
        return full_tlbr.int()

    def extract_box_features(self, feature: torch.Tensor, tlbr: torch.Tensor) -> T.List[torch.Tensor]:
        rect_features = []
        for t, l, b, r in tlbr.cpu().int().numpy():
            rect_feature = feature[:, t:b, l:r]
            rect_features.append(rect_feature)
        return rect_features

    def resize_to_uniform_height_width(self, features: T.List[torch.Tensor],) -> T.List[torch.Tensor]:
        assert len(features) > 0

        # Make the n-dim uniform (i.e. 4-dim, bchw)
        bchw_features = list(map(lambda f: f.unsqueeze(0) if f.dim() == 3 else f, features))

        heights = list(map(lambda f: f.size(2), bchw_features))
        widths = list(map(lambda f: f.size(3), bchw_features))

        max_height = max(heights)
        max_width = max(widths)

        bchw_features = list(map(lambda f: self.resize_image_batch(f, size=(max_height, max_width)), bchw_features))
        return bchw_features

    def similarities(self, image: torch.Tensor, tlbr: torch.Tensor) -> T.Generator[torch.Tensor, None, None]:
        for feature in self.compute_features_maps(image):
            _, H, W = image.size()
            _, HF, WF = feature.size()

            scaled_tlbr = self.rescale_box_features(tlbr, (HF / H, WF / W), (HF, WF))
            patches = self.extract_box_features(feature, scaled_tlbr)
            del scaled_tlbr

            patches = self.resize_to_uniform_height_width(patches)
            patches = torch.cat(patches, dim=0)
            # num_tlbr, feature_channels, uniform_height(max), uniform_width(max)

            for scaled_patches in self.generate_rescaled_patches(patches):
                similarity = self.compute_similarity_scores(feature.unsqueeze(0), scaled_patches)
                # 1, num_tlbr, uniform_height(max), uniform_width(max)

                del scaled_patches
                yield similarity
                del similarity

            del feature, patches

    def forward(self, image_batch: torch.Tensor, tlbr_batch: torch.Tensor):
        outputs = []

        for image, tlbr in zip(image_batch, tlbr_batch):
            assert image.dim() == 3
            assert tlbr.dim() == 2
            assert tlbr.size(1) == 4

            similarities = list(self.similarities(image, tlbr))
            similarities = self.resize_to_uniform_height_width(similarities)
            similarities = torch.cat(similarities, dim=0)
            # num_similarity_maps (num_features × num_scales), num_tlbr/exemplars, uniform_height(max), uniform_width(max)

            outputs.append(similarities)

        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(0, 2, 1, 3, 4).contiguous()
        # batch_size, num_tlbr/exemplars, num_similarity_maps (num_features × num_scales), uniform_height(max), uniform_width(max)

        return outputs
