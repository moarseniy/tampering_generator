import os
import yaml
import random
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class BaseForgeryGenerator:
    def __init__(self, config_path: str = "configs/generator_config.yaml"):
        self.config = self.load_config(config_path)
        self.sources = self.load_sources()
        
    def load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_sources(self) -> Dict:
        """Загрузка исходных изображений и разметки"""
        sources = {
            'images': {},
            'markup': {}
        }
        
        # Загрузка изображений
        images_dir = Path(self.config['paths']['source_images'])
        for img_path in images_dir.glob("*.*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image = cv2.imread(str(img_path))
                if image is not None:
                    # print(img_path, img_path.name)
                    sources['images'][img_path.name] = image
        
        # Загрузка разметки
        markup_dir = Path(self.config['paths']['source_markup'])
        for json_path in markup_dir.glob("*.json"):
            with open(json_path, 'r', encoding='utf-8') as f:
                markup_data = json.load(f)
                # print(json_path, json_path.stem)
                sources['markup'][json_path.stem] = markup_data
        
        # print(sources['markup'])

        print(f"Загружено {len(sources['images'])} изображений и {len(sources['markup'])} разметок")
        return sources
    
    def get_random_source(self) -> Tuple[str, np.ndarray, Dict]:
        """Получение случайного исходного документа"""
        available_keys = list(self.sources['images'].keys())
        if not available_keys:
            raise ValueError("Нет доступных исходных документов")
        
        key = random.choice(available_keys)
        # print(self.sources['markup'], key)
        return key, self.sources['images'][key], self.sources['markup'].get(key, {})
    
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
