import cv2
import numpy as np
import random
from pathlib import Path
from typing import Dict, Tuple
from .base_manipulator import BaseForgeryGenerator
from .splicing_operations import SplicingOperations
from .utils import save_forgery_result, ensure_directory

class DocumentForgeryGenerator(BaseForgeryGenerator):
    def __init__(self, config_path: str = "configs/generator_config.yaml"):
        super().__init__(config_path)
        self.splicing_ops = SplicingOperations(self.sources)
        self.create_output_directories()
    
    def generate_single_forgery(self, sample_id: int) -> Tuple[str, str]:
        """Генерация одной подделки"""
        # Выбор базового документа
        base_key, base_image, base_markup = self.get_random_source()
        
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
    
    def apply_splicing_operation(self, image: np.ndarray, markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Применение случайной splicing операции"""
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
            # print("bbox_swap")
            target_key, target_image, target_markup = self.get_random_source()
            return self.splicing_ops.bbox_swap(image, markup, target_image, target_markup)
        
        elif chosen_op == 'external_patch':
            # Выбор документа-донора
            # print("external_patch")
            patch_key, patch_image, patch_markup = self.get_random_source()
            return self.splicing_ops.external_patch_insertion(image, markup, patch_image, patch_markup)
        
        elif chosen_op == 'internal_swap':
            # print("internal_swap")
            return self.splicing_ops.internal_bbox_swap(image, markup)
        
        return image, np.zeros(image.shape[:2], dtype=np.uint8)
    
    def generate_dataset(self):
        """Генерация всего набора данных"""
        num_samples = self.config['generation']['num_samples']
        
        print(f"Начало генерации {num_samples} подделок...")
        
        for i in range(num_samples):
            try:
                image_path, mask_path = self.generate_single_forgery(i)
                # if i % 100 == 0:
                print(f"Сгенерировано {i+1}/{num_samples}")
            except Exception as e:
                print(f"Ошибка при генерации sample {i}: {e}")
                continue
        
        print(f"Генерация завершена! Создано {num_samples} подделок")
