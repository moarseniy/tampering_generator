import cv2
import numpy as np
import random
from typing import List, Dict, Tuple, Optional

class BBoxProcessor:
    @staticmethod
    def extract_bbox_region(image: np.ndarray, bbox: Dict) -> np.ndarray:
        """Извлечение области по bounding box"""
        x, y, w, h = bbox['bbox']
        return image[y:y+h, x:x+w].copy()
    
    @staticmethod
    def paste_bbox_region(target_image: np.ndarray, source_region: np.ndarray, 
                         bbox: Dict, resize: bool = True) -> np.ndarray:
        """Вставка области в bounding box"""
        x, y, w, h = bbox['bbox']

        # Проверяем, не выходит ли bbox за границы изображения
        img_h, img_w = target_image.shape[:2]
        
        # Корректируем координаты и размеры, если выходят за границы
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if resize and source_region.shape[:2] != (h, w):
            source_region = cv2.resize(source_region, (w, h))
        
        target_image[y:y+h, x:x+w] = source_region
        return target_image
    
    @staticmethod
    def get_random_bbox(markup: Dict, allowed_categories: List[str] = None) -> Optional[Dict]:
        """Получение случайного bbox из разметки"""
        if 'bboxes' not in markup or not markup['bboxes']:
            return None
        
        available_bboxes = markup['bboxes']
        if allowed_categories:
            available_bboxes = [bbox for bbox in available_bboxes 
                              if bbox.get('category', 'text') in allowed_categories]
        
        return random.choice(available_bboxes) if available_bboxes else None
    
    @staticmethod
    def create_bbox_mask(image_shape: Tuple[int, int], bbox: Dict) -> np.ndarray:
        """Создание маски для bounding box"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        x, y, w, h = bbox['bbox']
        mask[y:y+h, x:x+w] = 255
        return mask
    
    @staticmethod
    def find_suitable_bboxes(markup1: Dict, markup2: Dict, 
                           similarity_threshold: float = 0.3) -> List[Tuple[Dict, Dict]]:
        """Поиск подходящих пар bbox для замены"""
        if 'bboxes' not in markup1 or 'bboxes' not in markup2:
            return []
        
        suitable_pairs = []
        
        for bbox1 in markup1['bboxes']:
            for bbox2 in markup2['bboxes']:
                # Проверка схожести размеров
                w1, h1 = bbox1['bbox'][2], bbox1['bbox'][3]
                w2, h2 = bbox2['bbox'][2], bbox2['bbox'][3]
                
                size_similarity = min(w1/w2, w2/w1) * min(h1/h2, h2/h1)
                
                if size_similarity >= similarity_threshold:
                    suitable_pairs.append((bbox1, bbox2))
        
        return suitable_pairs
