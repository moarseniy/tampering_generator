import cv2
import numpy as np
import random
from typing import Dict, Tuple, List
from .bbox_processor import BBoxProcessor

class SplicingOperations:
    def __init__(self, sources: Dict, config):
        self.config = config
        self.sources = sources
        self.bbox_processor = BBoxProcessor()
    
    def bbox_swap(self, base_image: np.ndarray, base_markup: Dict, 
                 target_image: np.ndarray, target_markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Замена bbox между двумя документами"""
        mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        
        # Поиск подходящих пар bbox
        suitable_pairs = self.bbox_processor.find_suitable_bboxes(base_markup, target_markup)
        
        if not suitable_pairs:
            print("No suitable pairs found!")
            return base_image, mask
        
        # Выбор случайной пары для замены
        bbox_base, bbox_target = random.choice(suitable_pairs)
        
        try:
            # Корректируем bbox под фактические размеры изображений
            bbox_base_adjusted = self.bbox_processor.adjust_bbox_to_image(
                bbox_base, base_markup, base_image.shape
            )
            bbox_target_adjusted = self.bbox_processor.adjust_bbox_to_image(
                bbox_target, target_markup, target_image.shape
            )
            
            # Извлечение и замена областей
            base_region = self.bbox_processor.extract_bbox_region(base_image, bbox_base_adjusted)
            target_region = self.bbox_processor.extract_bbox_region(target_image, bbox_target_adjusted)
            
            # Проверка на нулевую размерность
            if base_region.size == 0 or target_region.size == 0:
                print(f"Warning: Empty region extracted. base_region.shape={base_region.shape}, target_region.shape={target_region.shape}")
                return base_image, mask
            
            # Вставка target_region в base_image по координатам bbox_base
            result_image = self.bbox_processor.paste_bbox_region(base_image, target_region, bbox_base_adjusted)
            
            # Конвертируем координаты bbox_target из target_image в base_image для вставки base_region
            # Но на самом деле, мы не хотим вставлять base_region в base_image - это было бы странно
            # Вместо этого, просто обновляем маску только для bbox_base (где мы вставили target_region)
            mask = np.maximum(mask, self.bbox_processor.create_bbox_mask(base_image.shape, bbox_base_adjusted))
        except (ValueError, IndexError) as e:
            print(f"Error in bbox_swap: {e}")
            print(f"  bbox_base: {bbox_base}, base_image.shape: {base_image.shape[:2]}")
            print(f"  bbox_target: {bbox_target}, target_image.shape: {target_image.shape[:2]}")
            if 'image_size' in base_markup:
                print(f"  base_markup image_size: {base_markup['image_size']}")
            if 'image_size' in target_markup:
                print(f"  target_markup image_size: {target_markup['image_size']}")
            return base_image, mask

        return result_image, mask
    
    def external_patch_insertion(self, base_image: np.ndarray, base_markup: Dict,
                               patch_image: np.ndarray, patch_markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Вставка случайного патча из другого документа"""
        mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        
        # Получение случайного bbox из patch документа
        patch_bbox = self.bbox_processor.get_random_bbox(patch_markup)
        if not patch_bbox:
            print("No bboxes found in markup!")
            return base_image, mask
        
        # Извлечение патча
        patch_region = self.bbox_processor.extract_bbox_region(patch_image, patch_bbox)
        
        # Случайная позиция для вставки
        h, w = base_image.shape[:2]
        patch_h, patch_w = patch_region.shape[:2]
        
        # Ограничение размера патча
        max_patch_size = self.config['splicing']['operations']['external_patch']['patch_size_range'][0]
        if patch_h > max_patch_size or patch_w > max_patch_size:
            scale = max_patch_size / max(patch_h, patch_w)
            new_w, new_h = int(patch_w * scale), int(patch_h * scale)
            patch_region = cv2.resize(patch_region, (new_w, new_h))
            patch_h, patch_w = new_h, new_w
        
        # Случайные координаты для вставки
        max_x = w - patch_w - 1
        max_y = h - patch_h - 1
        
        if max_x < 0 or max_y < 0:
            print("Random coordinates are out of bounds!")
            return base_image, mask
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # Создание искусственного bbox для вставки
        insert_bbox = {'bbox': [x, y, patch_w, patch_h]}
        
        # Вставка патча
        result_image = self.bbox_processor.paste_bbox_region(base_image, patch_region, insert_bbox, resize=False)
        
        # Обновление маски
        mask = self.bbox_processor.create_bbox_mask(base_image.shape, insert_bbox)
        
        return result_image, mask
    
    def internal_bbox_swap(self, base_image: np.ndarray, base_markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Внутренняя замена bbox внутри документа"""
        mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        # print(base_markup)
        if 'bboxes' not in base_markup or len(base_markup['bboxes']) < 2:
            print("No bboxes found in markup!")
            return base_image, mask
        
        # Выбор двух случайных bbox
        bbox1, bbox2 = random.sample(base_markup['bboxes'], 2)
        
        try:
            # Извлечение областей
            region1 = self.bbox_processor.extract_bbox_region(base_image, bbox1)
            region2 = self.bbox_processor.extract_bbox_region(base_image, bbox2)
            
            # Проверка на нулевую размерность
            if region1.size == 0 or region2.size == 0:
                print(f"Warning: Empty region extracted. region1.shape={region1.shape}, region2.shape={region2.shape}")
                return base_image, mask
            
            # Замена областей
            result_image = self.bbox_processor.paste_bbox_region(base_image, region1, bbox2)
            result_image = self.bbox_processor.paste_bbox_region(result_image, region2, bbox1)
            
            # Обновление маски
            mask = np.maximum(mask, self.bbox_processor.create_bbox_mask(base_image.shape, bbox1))
            mask = np.maximum(mask, self.bbox_processor.create_bbox_mask(base_image.shape, bbox2))
        except (ValueError, IndexError) as e:
            print(f"Error in internal_bbox_swap: {e}")
            print(f"  bbox1: {bbox1}, bbox2: {bbox2}, image.shape: {base_image.shape[:2]}")
            return base_image, mask
        
        return result_image, mask
