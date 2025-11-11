import cv2
import numpy as np
import random
from typing import List, Dict, Tuple, Optional

class BBoxProcessor:
    @staticmethod
    def extract_bbox_region(image: np.ndarray, bbox: Dict) -> np.ndarray:
        """Извлечение области по bounding box с проверкой границ"""
        x, y, w, h = bbox['bbox']
        
        # Проверка на нулевые или отрицательные размеры
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox dimensions: w={w}, h={h}. bbox={bbox}")
        
        # Получаем размеры изображения
        img_h, img_w = image.shape[:2]
        
        # Проверка, что bbox хотя бы частично внутри изображения
        if x >= img_w or y >= img_h or x + w <= 0 or y + h <= 0:
            raise ValueError(
                f"Bbox is completely outside image bounds: "
                f"bbox=[x={x}, y={y}, w={w}, h={h}], image_shape=({img_h}, {img_w})"
            )
        
        # Корректируем координаты начала среза (не могут быть отрицательными)
        x_start = max(0, x)
        y_start = max(0, y)
        # Корректируем координаты конца среза (не могут выходить за границы)
        x_end = min(x + w, img_w)
        y_end = min(y + h, img_h)
        
        # Вычисляем скорректированные размеры
        w_corrected = x_end - x_start
        h_corrected = y_end - y_start
        
        # Финальная проверка на валидность после коррекции
        if w_corrected <= 0 or h_corrected <= 0:
            raise ValueError(
                f"After correction, bbox has invalid dimensions: "
                f"w_corrected={w_corrected}, h_corrected={h_corrected}. "
                f"Original bbox=[x={x}, y={y}, w={w}, h={h}], image_shape=({img_h}, {img_w})"
            )
        
        # Извлекаем область
        region = image[y_start:y_end, x_start:x_end].copy()
        
        # Финальная проверка, что регион не пустой
        if region.size == 0:
            raise ValueError(
                f"Extracted region is empty: region.shape={region.shape}. "
                f"bbox=[x={x}, y={y}, w={w}, h={h}], "
                f"corrected=[x={x_start}, y={y_start}, w={w_corrected}, h={h_corrected}], "
                f"image_shape=({img_h}, {img_w})"
            )
        
        # Если размеры изменились из-за обрезки по границам, делаем resize до исходного размера
        # Это нужно для того, чтобы вставка работала корректно
        if w_corrected != w or h_corrected != h:
            region = cv2.resize(region, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Еще одна проверка после resize
        if region.size == 0:
            raise ValueError(f"Region became empty after resize: original_shape=({h_corrected}, {w_corrected}), target_shape=({h}, {w})")
        
        return region
    
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
    def scale_bbox(bbox: Dict, scale_x: float, scale_y: float) -> Dict:
        """Масштабирование bbox относительно исходных размеров"""
        x, y, w, h = bbox['bbox']
        scaled_bbox = {
            'bbox': [int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)],
            'category': bbox.get('category', 'text')
        }
        return scaled_bbox
    
    @staticmethod
    def adjust_bbox_to_image(bbox: Dict, markup: Dict, image_shape: Tuple[int, int]) -> Dict:
        """Корректировка bbox если размеры изображения не совпадают с разметкой"""
        img_h, img_w = image_shape[:2]
        
        # Если в разметке указаны размеры изображения
        if 'image_size' in markup:
            markup_w, markup_h = markup['image_size']  # в разметке [width, height]
            
            # Проверяем, нужно ли масштабировать
            if markup_w != img_w or markup_h != img_h:
                scale_x = img_w / markup_w
                scale_y = img_h / markup_h
                return BBoxProcessor.scale_bbox(bbox, scale_x, scale_y)
        
        return bbox.copy()
    
    @staticmethod
    def convert_bbox_between_images(bbox: Dict, source_markup: Dict, source_shape: Tuple[int, int],
                                   target_markup: Dict, target_shape: Tuple[int, int]) -> Dict:
        """Конвертация bbox из координат source_image в координаты target_image"""
        # Сначала корректируем bbox под source_image
        bbox_source = BBoxProcessor.adjust_bbox_to_image(bbox, source_markup, source_shape)
        
        # Получаем размеры в нормализованных координатах (относительно разметки)
        if 'image_size' in source_markup:
            source_markup_w, source_markup_h = source_markup['image_size']
            x, y, w, h = bbox['bbox']
            
            # Нормализуем координаты (0-1)
            norm_x = x / source_markup_w
            norm_y = y / source_markup_h
            norm_w = w / source_markup_w
            norm_h = h / source_markup_h
            
            # Денормализуем для target_image
            if 'image_size' in target_markup:
                target_markup_w, target_markup_h = target_markup['image_size']
                target_x = int(norm_x * target_markup_w)
                target_y = int(norm_y * target_markup_h)
                target_w = int(norm_w * target_markup_w)
                target_h = int(norm_h * target_markup_h)
            else:
                # Если нет разметки, используем фактические размеры
                target_h, target_w = target_shape[:2]
                target_x = int(norm_x * target_w)
                target_y = int(norm_y * target_h)
                target_w = int(norm_w * target_w)
                target_h = int(norm_h * target_h)
            
            converted_bbox = {
                'bbox': [target_x, target_y, target_w, target_h],
                'category': bbox.get('category', 'text')
            }
            
            # Финальная корректировка под фактические размеры target_image
            return BBoxProcessor.adjust_bbox_to_image(converted_bbox, target_markup, target_shape)
        else:
            # Если нет разметки, просто корректируем пропорционально
            source_h, source_w = source_shape[:2]
            target_h, target_w = target_shape[:2]
            
            scale_x = target_w / source_w
            scale_y = target_h / source_h
            
            return BBoxProcessor.scale_bbox(bbox_source, scale_x, scale_y)
    
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
                
                # Защита от деления на ноль
                if w2 == 0 or h2 == 0 or w1 == 0 or h1 == 0:
                    continue
                
                size_similarity = min(w1/w2, w2/w1) * min(h1/h2, h2/h1)
                
                if size_similarity >= similarity_threshold:
                    suitable_pairs.append((bbox1, bbox2))
        
        return suitable_pairs
