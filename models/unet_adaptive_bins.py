import torch
import torch.nn as nn
import torch.nn.functional as F

from .miniViT import mViT


def risk_density_to_edges(risk_density, min_val, max_val, n_bins):
    batch_size, n_anchors = risk_density.shape
    device = risk_density.device
    dtype = risk_density.dtype
    anchors = torch.linspace(min_val, max_val, n_anchors, device=device, dtype=dtype)
    taus = torch.linspace(0.0, 1.0, n_bins + 1, device=device, dtype=dtype)
    cdf = torch.cumsum(risk_density, dim=1)
    edges = risk_density.new_empty(batch_size, n_bins + 1)
    edges[:, 0] = min_val
    edges[:, -1] = max_val

    for b in range(batch_size):
        for i in range(1, n_bins):
            tau = taus[i]
            right = torch.searchsorted(cdf[b], tau).clamp(0, n_anchors - 1)
            if right.item() == 0:
                cdf_left = cdf[b].new_tensor(0.0)
                anchor_left = anchors[0]
            else:
                cdf_left = cdf[b, right - 1]
                anchor_left = anchors[right - 1]
            cdf_right = cdf[b, right]
            alpha = (tau - cdf_left) / (cdf_right - cdf_left).clamp_min(1e-6)
            edges[b, i] = anchor_left + alpha * (anchors[right] - anchor_left)

    edges, _ = torch.sort(edges, dim=1)
    edges[:, 0] = min_val
    edges[:, -1] = max_val
    return edges


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[
            11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UnetAdaptiveBins(nn.Module):
    def __init__(self, backend, n_bins=100, min_val=0.1, max_val=10, norm='linear', bin_mode='adabins',
                 risk_num_anchors=128, risk_eps=1e-6):
        super(UnetAdaptiveBins, self).__init__()
        if bin_mode not in ('adabins', 'risk'):
            raise ValueError("bin_mode must be 'adabins' or 'risk'")
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_mode = bin_mode
        self.risk_num_anchors = risk_num_anchors
        self.risk_eps = risk_eps
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                                        dim_out=n_bins,
                                        embedding_dim=128, norm=norm)

        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.risk_density_head = None
        if self.bin_mode == 'risk':
            self.risk_density_head = nn.Conv2d(128, risk_num_anchors, kernel_size=1, stride=1, padding=0)

    def forward(self, x, return_details=False, **kwargs):
        unet_out = self.decoder(self.encoder(x), **kwargs)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        out = self.conv_out(range_attention_maps)

        # Post process
        # n, c, h, w = out.shape
        # hist = torch.sum(out.view(n, c, h * w), dim=2) / (h * w)  # not used for training

        risk_density = None
        if self.bin_mode == 'adabins':
            bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
            bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
            bin_edges = torch.cumsum(bin_widths, dim=1)
        else:
            risk_logits = self.risk_density_head(unet_out).mean(dim=(2, 3))
            risk_density = F.softplus(risk_logits) + self.risk_eps
            risk_density = risk_density / risk_density.sum(dim=1, keepdim=True)
            bin_edges = risk_density_to_edges(risk_density, self.min_val, self.max_val, self.num_classes)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        if return_details:
            return {
                "bin_edges": bin_edges,
                "pred": pred,
                "risk_density": risk_density,
                "bin_widths": bin_edges[:, 1:] - bin_edges[:, :-1]
            }

        return bin_edges, pred

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        if self.risk_density_head is not None:
            modules.append(self.risk_density_head)
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel_name = 'tf_efficientnet_b5_ap'

        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print('Building Encoder-Decoder model..', end='')
        m = cls(basemodel, n_bins=n_bins, **kwargs)
        print('Done.')
        return m


if __name__ == '__main__':
    model = UnetAdaptiveBins.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)
