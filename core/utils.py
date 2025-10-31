import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

def ensure_directory(path: str):
    """Создание директории если не существует"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_forgery_result(image: np.ndarray, mask: np.ndarray, filename: str, 
                       output_dirs: Dict, image_format: str = "png"):
    """Сохранение результата подделки и маски"""
    # Сохранение изображения
    image_path = Path(output_dirs['images']) / f"{filename}.{image_format}"
    cv2.imwrite(str(image_path), image)
    
    # Сохранение маски
    mask_path = Path(output_dirs['masks']) / f"{filename}_mask.{image_format}"
    cv2.imwrite(str(mask_path), mask)
    
    return str(image_path), str(mask_path)

def validate_bbox(bbox: Dict, image_shape: Tuple[int, int]) -> bool:
    """Проверка валидности bounding box"""
    if 'bbox' not in bbox:
        return False
    
    x, y, w, h = bbox['bbox']
    img_h, img_w = image_shape[:2]
    
    return (x >= 0 and y >= 0 and 
            x + w <= img_w and 
            y + h <= img_w and 
            w > 0 and h > 0)
