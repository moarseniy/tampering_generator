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
        num_samples: int = 10000,
        cache_size: int = 100
    ):
        """
        Args:
            config_path: путь к конфигурационному файлу
            transform: трансформации для изображений
            target_transform: трансформации для масок
            image_size: размер изображений (высота, ширина)
            num_samples: количество samples в датасете
            cache_size: размер кеша для ускорения загрузки
        """
        self.generator = InlineForgeryGenerator(config_path, cache_size)
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
            image = self.transform(image)
        else:
            # Базовая трансформация в тензор
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Базовая трансформация маски
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image, mask

class InlineForgeryGenerator(BaseForgeryGenerator):
    """
    Генератор подделок для работы в памяти (без сохранения на диск)
    """
    
    def __init__(self, config_path: str = "configs/generator_config.yaml", cache_size: int = 100):
        super().__init__(config_path)
        self.splicing_ops = SplicingOperations(self.sources)
        self.cache_size = cache_size
        self._image_cache = {}
        
    def _get_cached_image(self, key: str) -> np.ndarray:
        """Получение изображения из кеша"""
        if key not in self._image_cache:
            self._image_cache[key] = self.sources['images'][key]
            # Ограничиваем размер кеша
            if len(self._image_cache) > self.cache_size:
                # Удаляем самый старый элемент
                oldest_key = next(iter(self._image_cache))
                del self._image_cache[oldest_key]
        return self._image_cache[key]
    
    def generate_forgery(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация одной подделки в памяти
        
        Returns:
            image: изображение подделки [H, W, C]
            mask: бинарная маска [H, W]
        """
        # Выбор базового документа
        base_key, base_image, base_markup = self.get_random_source()
        
        # Используем кешированное изображение
        base_image = self._get_cached_image(base_key)
        
        # Инициализация маски
        mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        result_image = base_image.copy()
        
        # Применение splicing операций
        num_operations = random.randint(1, self.config['generation']['max_operations_per_image'])
        
        for op_idx in range(num_operations):
            result_image, op_mask = self.apply_splicing_operation(result_image, base_markup)
            mask = np.maximum(mask, op_mask)
        
        # Применение деградации качества
        if random.random() < 0.7:  # 70% chance
            result_image = self.apply_quality_degradation(result_image)
        
        return result_image, mask
    
    def apply_splicing_operation(self, image: np.ndarray, markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Применение случайной splicing операции (переопределяем для работы с кешем)"""
        splicing_config = self.config['splicing']['operations']
        
        # Выбор операции на основе вероятностей
        operations = []
        probabilities = []
        
        for op_name, op_config in splicing_config.items():
            if op_config.get('enabled', False):
                operations.append(op_name)
                probabilities.append(op_config.get('probability', 0.5))
        
        if not operations:
            return image, np.zeros(image.shape[:2], dtype=np.uint8)
        
        chosen_op = random.choices(operations, weights=probabilities)[0]
        
        if chosen_op == 'bbox_swap':
            # Выбор целевого документа для обмена
            target_key, target_image, target_markup = self.get_random_source()
            target_image = self._get_cached_image(target_key)
            return self.splicing_ops.bbox_swap(image, markup, target_image, target_markup)
        
        elif chosen_op == 'external_patch':
            # Выбор документа-донора
            patch_key, patch_image, patch_markup = self.get_random_source()
            patch_image = self._get_cached_image(patch_key)
            return self.splicing_ops.external_patch_insertion(image, markup, patch_image, patch_markup)
        
        elif chosen_op == 'internal_swap':
            return self.splicing_ops.internal_bbox_swap(image, markup)
        
        return image, np.zeros(image.shape[:2], dtype=np.uint8)


def get_train_transforms(image_size: Tuple[int, int] = (1024, 1024)):
    """
    Трансформации для обучения
    """
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
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_mask_transforms(image_size: Tuple[int, int] = (1024, 1024)):
    """
    Трансформации для масок
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        ToTensorV2()
    ])