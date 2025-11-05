import cv2
import numpy as np
import random
from pathlib import Path
from typing import Dict, Tuple
from .base_manipulator import BaseForgeryGenerator
from .splicing_operations import SplicingOperations
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
    get_mask_transforms
)


class DocumentForgeryGenerator(BaseForgeryGenerator):
    def __init__(self, config_path: str = "configs/generator_config.yaml"):
        super().__init__(config_path)
        self.splicing_ops = SplicingOperations(self.sources, self.config)
        self.create_output_directories()
    
    def generate_single_forgery(self, sample_id: int) -> Tuple[str, str]:
        """Генерация одной подделки"""
        # Выбор базового документа
        base_key, base_image, base_markup = self.get_random_source()
        
        # Инициализация маски
        mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        result_image = base_image.copy()
        
        # Применение splicing операций
        num_operations = random.randint(self.config['generation']['operations_per_image'][0], 
                                        self.config['generation']['operations_per_image'][1])
        
        for op_idx in range(num_operations):
            result_image, op_mask = self.apply_splicing_operation(result_image, base_markup)
            mask = np.maximum(mask, op_mask)

        # Применение деградации качества
        if 'jpeg_compression' in self.config['quality'] and self.config['quality']['jpeg_compression']['enabled']:
            if random.random() < self.config['quality']['jpeg_compression']['prob']:
                result_image = self.apply_quality_degradation(result_image)
            
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
    
    # Реализация apply_splicing_operation унаследована от BaseForgeryGenerator
     
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
