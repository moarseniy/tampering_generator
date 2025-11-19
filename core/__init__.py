import cv2
import numpy as np
import random
from pathlib import Path
from typing import Dict, Tuple
from .base_manipulator import BaseForgeryGenerator
from .splicing_operations import SplicingOperations
from .digital_signature import DigitalSignatureGenerator

from .utils import save_forgery_result, ensure_directory
from .utils import (
    ensure_directory, 
    save_forgery_result, 
    validate_bbox
)

from .dataset import (
    ForgeryDataset, 
    InlineForgeryGenerator,     
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
    get_mask_transforms
)


class DocumentForgeryGenerator(BaseForgeryGenerator):
    def __init__(self, config_path: str = "configs/generator_config.yaml"):
        super().__init__(config_path)
        self.create_output_directories()
    
    def generate_single_forgery(self, sample_id: int) -> Tuple[str, str]:
        """Генерация одной подделки"""
        # Общая генерация в памяти
        result_image, mask = self.generate_forgery_in_memory()
            
        # Сохранение результата
        filename = f"forgery_{sample_id:06d}"
        image_path, mask_path = save_forgery_result(
            result_image, mask, filename,
            {
                'images': self.config['paths']['output_images'],
                'masks': self.config['paths']['output_masks']
            },
            self.config['generation']['image_format']
        )
        
        return image_path, mask_path
    
    def generate_dataset(self):
        """Генерация всего набора данных"""
        num_samples = self.config['generation']['num_samples']
        
        print(f"Начало генерации {num_samples} подделок...")
        
        for i in range(num_samples):
            # try:
            image_path, mask_path = self.generate_single_forgery(i)
            # if i % 100 == 0:
            print(f"Сгенерировано {i+1}/{num_samples}")
            # except Exception as e:
            #     print(f"Ошибка при генерации sample {i}: {e}")
            #     continue
        
        print(f"Генерация завершена! Создано {num_samples} подделок")
