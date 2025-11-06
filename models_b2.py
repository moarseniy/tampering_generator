import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel

class LightMultiScaleContextFusion(nn.Module):
    def __init__(self, in_channels, reduction=8):  # Уменьшили reduction
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 5, padding=2),  # Уменьшили ядро
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

class LightFeaturePyramidDecoder(nn.Module):
    def __init__(self, encoder_channels=[64, 128, 320, 512], decoder_channels=128):  # Уменьшили каналы
        super().__init__()

        # Progressive upsampling with skip connections
        self.lateral_conv1 = nn.Conv2d(encoder_channels[3], decoder_channels, 1)
        self.lateral_conv2 = nn.Conv2d(encoder_channels[2], decoder_channels, 1)
        self.lateral_conv3 = nn.Conv2d(encoder_channels[1], decoder_channels, 1)
        self.lateral_conv4 = nn.Conv2d(encoder_channels[0], decoder_channels, 1)

        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1)
            for _ in range(3)
        ])

        # Один контекстный блок вместо четырех
        self.context_block = LightMultiScaleContextFusion(decoder_channels)

    def forward(self, features):
        # features: [c1, c2, c3, c4] from different stages

        # Top-down pathway with lateral connections
        p4 = self.lateral_conv1(features[3])  # 1/32

        p3 = self.lateral_conv2(features[2])  # 1/16
        p3 = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False) + p3
        p3 = self.fusion_convs[0](p3)

        p2 = self.lateral_conv3(features[1])  # 1/8
        p2 = F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=False) + p2
        p2 = self.fusion_convs[1](p2)

        p1 = self.lateral_conv4(features[0])  # 1/4
        p1 = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False) + p1
        p1 = self.fusion_convs[2](p1)
        
        # Применяем контекстный блок только на последнем уровне
        p1 = self.context_block(p1)

        return p1

class LightBoundaryAwareHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()

        # Общий экстрактор признаков
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        # Головы сегментации и границ
        self.seg_head = nn.Conv2d(in_channels // 4, num_classes, 1)
        self.boundary_head = nn.Conv2d(in_channels // 4, num_classes, 1)

    def forward(self, x):
        # Общие признаки
        shared_features = self.shared_conv(x)

        # Основная сегментация и границы
        seg_main = self.seg_head(shared_features)
        boundary = self.boundary_head(shared_features)

        return seg_main, boundary

class LightDocumentForgerySegmentor(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        # Используем меньшую модель SegFormer-B2 вместо B5
        self.encoder = SegformerModel.from_pretrained(
            "nvidia/mit-b2" if pretrained else None,  # B2 вместо B5
            config=SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=[3, 4, 6, 3],  # Меньшая глубина
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[64, 128, 320, 512],  # Те же каналы, но меньше слоев
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
            )
        )

        # Облегченный декодер
        self.decoder = LightFeaturePyramidDecoder(
            encoder_channels=[64, 128, 320, 512],
            decoder_channels=256  # Уменьшили каналы декодера
        )

        # Облегченная голова
        self.head = LightBoundaryAwareHead(256, num_classes)

        # Только один вспомогательный выход вместо двух
        self.aux_head = nn.Conv2d(320, num_classes, 1)  # Только для c3

        # Упрощенное слияние признаков
        self.msf_fusion = nn.Sequential(
            nn.Conv2d(512 + 320, 256, 1),  # Только c3 + c4
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder forward pass
        encoder_features = self.encoder(x, output_hidden_states=True)
        features = encoder_features.hidden_states  # [c1, c2, c3, c4]

        # Упрощенное слияние признаков - только c3 и c4
        target_size = features[0].shape[2:]  # c1 size (1/4)

        # Ресайзим c3 и c4 до размера c1
        c3_resized = F.interpolate(
            features[2], size=target_size, mode='bilinear', align_corners=False
        )
        c4_resized = F.interpolate(
            features[3], size=target_size, mode='bilinear', align_corners=False
        )

        # Конкатенируем и сливаем
        fused = torch.cat([c3_resized, c4_resized], dim=1)
        fused = self.msf_fusion(fused)

        # Комбинируем с c1
        combined = features[0] + fused

        # Декодер
        decoder_features = [combined] + features[1:4]
        decoded = self.decoder(decoder_features)

        # Основная голова
        seg_pred, boundary_pred = self.head(decoded)

        # Один вспомогательный выход
        aux_pred = self.aux_head(features[2])  # c3
        aux_pred = F.interpolate(
            aux_pred, size=x.shape[2:], mode='bilinear', align_corners=False
        )

        # Апсемплим до размера входа
        seg_pred = F.interpolate(
            seg_pred, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        boundary_pred = F.interpolate(
            boundary_pred, size=x.shape[2:], mode='bilinear', align_corners=False
        )

        return {
            'segmentation': seg_pred,
            'boundary': boundary_pred,
            'auxiliary': [aux_pred]  # Все равно список для совместимости
        }