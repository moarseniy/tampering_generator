import os
import yaml
import random
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .splicing_operations import SplicingOperations

class BaseForgeryGenerator:
    def __init__(self, config_path: str = "configs/generator_config.yaml"):
        self.config = self.load_config(config_path)
        self.sources = self.load_sources()
        self.splicing_ops = SplicingOperations(self.sources, self.config)
        self.forgeries_per_source = self.config['generation'].get('forgeries_per_source', 0)
        self.forgery_prob = self.forgeries_per_source / (self.forgeries_per_source + 1)

    def load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_sources(self) -> Dict:
        """Индексирование путей исходных изображений и загрузка разметки (без кеша изображений)."""
        sources = {
            'images': {},  # key -> absolute image path
            'markup': {}
        }
        
        # Индексирование путей изображений
        images_dir = Path(self.config['paths']['source_images']).resolve()
        for img_path in images_dir.rglob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                sources['images'][img_path.name] = str(img_path)
        
        # Загрузка разметки
        markup_dir = Path(self.config['paths']['source_markup']).resolve()
        for json_path in markup_dir.rglob("*.json"):
            with open(json_path, 'r', encoding='utf-8') as f:
                markup_data = json.load(f)
                sources['markup'][json_path.stem] = markup_data
        
        print(f"Загружено {len(sources['images'])} путей изображений и {len(sources['markup'])} разметок")
        return sources
    
    def get_random_source(self) -> Tuple[str, np.ndarray, Dict]:
        """Получение случайного исходного документа (ленивая загрузка изображения с диска)."""
        available_keys = list(self.sources['images'].keys())
        if not available_keys:
            raise ValueError("Нет доступных исходных документов")
        
        key = random.choice(available_keys)
        img_path = self.sources['images'][key]
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        return key, image, self.sources['markup'].get(key, {})

    def apply_splicing_operation(self, image: np.ndarray, markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Применение случайной splicing операции (общая реализация без кеша)."""
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
            target_key, target_image, target_markup = self.get_random_source()
            return self.splicing_ops.bbox_swap(image, markup, target_image, target_markup)
        elif chosen_op == 'external_patch':
            patch_key, patch_image, patch_markup = self.get_random_source()
            return self.splicing_ops.external_patch_insertion(image, markup, patch_image, patch_markup)
        elif chosen_op == 'internal_swap':
            return self.splicing_ops.internal_bbox_swap(image, markup)
        
        return image, np.zeros(image.shape[:2], dtype=np.uint8)
    
    def create_output_directories(self):
        """Создание выходных директорий"""
        output_dirs = [
            self.config['paths']['output_images'],
            self.config['paths']['output_masks']
        ]
        
        for dir_path in output_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def apply_quality_degradation(self, image: np.ndarray) -> np.ndarray:
        """Применение деградации качества (JPEG сжатие) для реалистичности"""
        img = image.copy()
        
        quality = random.randint(*self.config['quality']['jpeg_compression']['quality'])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    
        return img

    def generate_forgery_in_memory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Общая генерация подделки в памяти (используется и офлайн, и в датасете)."""
        # Выбор базового документа
        base_key, base_image, base_markup = self.get_random_source()
        
        # Инициализация маски
        mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        result_image = base_image.copy()
        
        if random.random() < self.forgery_prob:
            num_operations = random.randint(
                self.config['generation']['operations_per_image'][0],
                self.config['generation']['operations_per_image'][1]
            )
            
            for _ in range(num_operations):
                result_image, op_mask = self.apply_splicing_operation(result_image, base_markup)
                mask = np.maximum(mask, op_mask)
            
            # Применение деградации качества
            # if 'jpeg_compression' in self.config['quality'] and self.config['quality']['jpeg_compression']['enabled']:
            #     if random.random() < self.config['quality']['jpeg_compression']['prob']:
            #         result_image = self.apply_quality_degradation(result_image)
            
        return result_image, mask
