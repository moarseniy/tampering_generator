import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
from typing import Tuple, Optional, Callable, Dict
from .base_manipulator import BaseForgeryGenerator
from .splicing_operations import SplicingOperations

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.transforms as T

class ForgeryDataset(Dataset):
    """
    PyTorch Dataset для генерации подделок документов на лету
    """
    
    def __init__(
        self,
        config_path: str = "configs/generator_config.yaml",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (1024, 1024),
        num_samples: int = 10000
    ):
        """
        Args:
            config_path: путь к конфигурационному файлу
            transform: трансформации для изображений
            target_transform: трансформации для масок
            image_size: размер изображений (высота, ширина)
            num_samples: количество samples в датасете
        """
        self.generator = InlineForgeryGenerator(config_path)
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.num_samples = num_samples
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Генерация подделки на лету
        
        Returns:
            image: тензор изображения [C, H, W]
            mask: тензор маски [1, H, W]
        """
        # Генерируем подделку
        image, mask = self.generator.generate_forgery()
        
        # Применяем трансформации если заданы
        if self.transform:
            # transformed = self.transform(image=image)
            # image = transformed["image"]
            image = self.transform(image)
        else:
            # Базовая трансформация в тензор
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if self.target_transform:
            # transformed_mask = self.target_transform(image=mask)
            # mask = transformed_mask["image"]
            mask = self.target_transform(mask)
        else:
            # Базовая трансформация маски
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image, mask

class InlineForgeryGenerator(BaseForgeryGenerator):
    """
    Генератор подделок для работы в памяти (без сохранения на диск)
    """
    
    def __init__(self, config_path: str = "configs/generator_config.yaml"):
        super().__init__(config_path)
    
    def generate_forgery(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация одной подделки в памяти (делегируем базовому классу)
        """
        return self.generate_forgery_in_memory()


def get_train_transforms(image_size: Tuple[int, int] = (1024, 1024)):
    """
    Трансформации для обучения
    """
    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size[0], image_size[1]))
    ])

    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_val_transforms(image_size: Tuple[int, int] = (1024, 1024)):
    """
    Трансформации для валидации
    """
    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size[0], image_size[1]))
    ])

    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_test_transforms(image_size: Tuple[int, int] = (1024, 1024)):
    """
    Трансформации для тестирования (идентичны валидационным - без аугментаций)
    """
    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size[0], image_size[1]))
    ])


def get_mask_transforms(image_size: Tuple[int, int] = (1024, 1024)):
    """
    Трансформации для масок
    """
    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size[0], image_size[1]))
    ])

    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        ToTensorV2()
    ])