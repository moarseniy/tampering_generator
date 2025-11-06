import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel

class UltraLightDecoder(nn.Module):
    def __init__(self, encoder_channels=[64, 128, 320, 512], decoder_channels=64):
        super().__init__()
        
        # Простые боковые соединения
        self.lateral_conv4 = nn.Conv2d(encoder_channels[3], decoder_channels, 1)
        self.lateral_conv3 = nn.Conv2d(encoder_channels[2], decoder_channels, 1)
        self.lateral_conv2 = nn.Conv2d(encoder_channels[1], decoder_channels, 1)
        self.lateral_conv1 = nn.Conv2d(encoder_channels[0], decoder_channels, 1)
        
        # Одна свертка для слияния
        self.fusion_conv = nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1)

    def forward(self, features):
        # Простой top-down path
        p4 = self.lateral_conv4(features[3])
        
        p3 = self.lateral_conv3(features[2])
        p3 = F.interpolate(p4, size=p3.shape[2:], mode='bilinear') + p3
        
        p2 = self.lateral_conv2(features[1])
        p2 = F.interpolate(p3, size=p2.shape[2:], mode='bilinear') + p2
        
        p1 = self.lateral_conv1(features[0])
        p1 = F.interpolate(p2, size=p1.shape[2:], mode='bilinear') + p1
        p1 = self.fusion_conv(p1)

        return p1

class SimpleHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        # Очень простая голова
        self.seg_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, num_classes, 1)
        )
        
        # Boundary как дополнительный выход из тех же признаков
        self.boundary_conv = nn.Conv2d(in_channels // 2, num_classes, 1)

    def forward(self, x):
        features = self.seg_conv[:-1](x)  # Признаки до последнего слоя
        seg = self.seg_conv[-1](features)  # Сегментация
        boundary = self.boundary_conv(features)  # Границы
        
        return seg, boundary

class MiniDocumentForgerySegmentor(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        
        # Самый маленький SegFormer B0
        self.encoder = SegformerModel.from_pretrained(
            "nvidia/mit-b0" if pretrained else None,
            config=SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=[2, 2, 2, 2],  # Минимальная глубина
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[32, 64, 160, 256],  # Уменьшенные каналы
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
            )
        )

        # Мини-декодер
        self.decoder = UltraLightDecoder(
            encoder_channels=[32, 64, 160, 256],
            decoder_channels=64
        )

        # Простая голова
        self.head = SimpleHead(64, num_classes)

    def forward(self, x):
        # Энкодер
        encoder_features = self.encoder(x, output_hidden_states=True)
        features = encoder_features.hidden_states
        
        # Прямой декодер
        decoded = self.decoder(features)
        
        # Голова
        seg_pred, boundary_pred = self.head(decoded)
        
        # Апсемплинг
        seg_pred = F.interpolate(seg_pred, size=x.shape[2:], mode='bilinear')
        boundary_pred = F.interpolate(boundary_pred, size=x.shape[2:], mode='bilinear')

        return {
            'segmentation': seg_pred,
            'boundary': boundary_pred,
            'auxiliary': []  # Без вспомогательных выходов
        }