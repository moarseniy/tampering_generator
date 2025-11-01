import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel
from timm.models.layers import DropPath

class MultiScaleContextFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x = x * sa

        return x

class FeaturePyramidDecoder(nn.Module):
    def __init__(self, encoder_channels=[64, 128, 320, 512], decoder_channels=256):
        super().__init__()

        # Progressive upsampling with skip connections
        self.lateral_conv1 = nn.Conv2d(encoder_channels[3], decoder_channels, 1)
        self.lateral_conv2 = nn.Conv2d(encoder_channels[2], decoder_channels, 1)
        self.lateral_conv3 = nn.Conv2d(encoder_channels[1], decoder_channels, 1)
        self.lateral_conv4 = nn.Conv2d(encoder_channels[0], decoder_channels, 1)

        self.fusion_conv1 = nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1)
        self.fusion_conv2 = nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1)
        self.fusion_conv3 = nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1)
        self.fusion_conv4 = nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1)

        self.context_blocks = nn.ModuleList([
            MultiScaleContextFusion(decoder_channels) for _ in range(4)
        ])

    def forward(self, features):
        # features: [c1, c2, c3, c4] from different stages

        # Top-down pathway with lateral connections
        p4 = self.lateral_conv1(features[3])  # 1/32
        p4 = self.context_blocks[0](p4)

        p3 = self.lateral_conv2(features[2])  # 1/16
        p3 = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False) + p3
        p3 = self.fusion_conv1(p3)
        p3 = self.context_blocks[1](p3)

        p2 = self.lateral_conv3(features[1])  # 1/8
        p2 = F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=False) + p2
        p2 = self.fusion_conv2(p2)
        p2 = self.context_blocks[2](p2)

        p1 = self.lateral_conv4(features[0])  # 1/4
        p1 = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False) + p1
        p1 = self.fusion_conv3(p1)
        p1 = self.context_blocks[3](p1)

        return p1

class BoundaryAwareHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()

        # Main segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, num_classes, 1)
        )

        # Boundary refinement head
        self.boundary_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, num_classes, 1)
        )

        # Feature enhancement
        self.enhance_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Main segmentation
        seg_main = self.seg_head(x)

        # Boundary prediction
        boundary = self.boundary_head(x)

        # Combine features
        enhanced = self.enhance_conv(torch.cat([x, boundary], dim=1))

        # Final segmentation with boundary guidance
        seg_final = self.seg_head(enhanced)

        return seg_final, boundary

class DocumentForgerySegmentor(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        # Encoder: SegFormer-B5 (largest variant)
        self.encoder = SegformerModel.from_pretrained(
            "nvidia/mit-b5" if pretrained else None,
            config=SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=[3, 6, 40, 3],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[64, 128, 320, 512],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
            )
        )

        # Feature Pyramid Decoder
        self.decoder = FeaturePyramidDecoder(
            encoder_channels=[64, 128, 320, 512],
            decoder_channels=512
        )

        # Boundary-aware segmentation head
        self.head = BoundaryAwareHead(512, num_classes)

        # Deep supervision
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(128, num_classes, 1),
            nn.Conv2d(320, num_classes, 1)
        ])

        # Multi-scale feature fusion
        self.msf_fusion = nn.Sequential(
            nn.Conv2d(512 * 3, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder forward pass
        encoder_features = self.encoder(x, output_hidden_states=True)
        features = encoder_features.hidden_states  # [c1, c2, c3, c4]

        # Multi-scale feature fusion
        feature_sizes = [f.shape[2:] for f in features[1:4]]  # c2, c3, c4

        # Resize features to common size (1/4 of input)
        target_size = features[0].shape[2:]  # c1 size (1/4)

        fused_features = []
        for i, feat in enumerate(features[1:4]):  # c2, c3, c4
            if feat.shape[2:] != target_size:
                feat_resized = F.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            else:
                feat_resized = feat
            fused_features.append(feat_resized)

        # Concatenate and fuse
        fused = torch.cat(fused_features, dim=1)
        fused = self.msf_fusion(fused)

        # Combine with highest resolution feature
        combined = features[0] + fused  # c1 + fused features

        # Decoder
        decoder_features = [combined] + features[1:4]
        decoded = self.decoder(decoder_features)

        # Main head
        seg_pred, boundary_pred = self.head(decoded)

        # Deep supervision (auxiliary outputs)
        aux_outputs = []
        for i, head in enumerate(self.aux_heads):
            aux_feat = features[i + 1]  # c2, c3
            aux_pred = head(aux_feat)
            aux_pred = F.interpolate(
                aux_pred, size=x.shape[2:], mode='bilinear', align_corners=False
            )
            aux_outputs.append(aux_pred)

        # Upsample to input size
        seg_pred = F.interpolate(
            seg_pred, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        boundary_pred = F.interpolate(
            boundary_pred, size=x.shape[2:], mode='bilinear', align_corners=False
        )

        return {
            'segmentation': seg_pred,
            'boundary': boundary_pred,
            'auxiliary': aux_outputs
        }
