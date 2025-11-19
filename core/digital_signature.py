# core/digital_signature.py
import random
import numpy as np
from PIL import Image, ImageDraw
import cv2
from typing import Tuple, List, Optional, Dict

def random_bezier_curve(draw, width, height, steps=200, thickness=3):
    """Генерация случайной кривой Безье для подписи"""
    # Случайные контрольные точки с более естественным распределением
    points = np.array([
        [random.randint(0, width//4), random.randint(height//4, 3*height//4)],
        [random.randint(width//6, width//2), random.randint(0, height)],
        [random.randint(width//2, 5*width//6), random.randint(0, height)],
        [random.randint(3*width//4, width), random.randint(height//4, 3*height//4)]
    ], dtype=float)

    def bezier(t):
        return (
            (1-t)**3 * points[0] +
            3*(1-t)**2*t * points[1] +
            3*(1-t)*t**2 * points[2] +
            t**3 * points[3]
        )

    prev = bezier(0)
    for i in range(1, steps):
        t = i / (steps - 1)
        cur = bezier(t)
        draw.line((prev[0], prev[1], cur[0], cur[1]), fill=0, width=thickness)
        prev = cur

def generate_realistic_signature(width=400, height=120) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерация реалистичной подписи с использованием кривых Безье
    
    Returns:
        signature: цветное изображение подписи (BGR)
        mask: бинарная маска подписи
    """
    # Создаем изображение с белым фоном
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    
    # Количество основных штрихов подписи
    num_strokes = random.randint(2, 5)
    
    # Основные штрихи подписи
    for _ in range(num_strokes):
        thickness = random.randint(2, 4)
        random_bezier_curve(draw, width, height, thickness=thickness)
    
    # Дополнительные мелкие штрихи для реалистичности
    if random.random() > 0.3:
        for _ in range(random.randint(1, 3)):
            thickness = random.randint(1, 2)
            random_bezier_curve(draw, width, height, 
                              steps=random.randint(50, 100), 
                              thickness=thickness)
    
    # Конвертируем PIL Image в numpy array
    signature_gray = np.array(img)
    
    # Создаем цветное изображение (белый фон, черная подпись)
    signature_color = np.stack([signature_gray] * 3, axis=-1)
    
    # Создаем маску (белая подпись на черном фоне)
    mask = 255 - signature_gray
    
    return signature_color, mask

class DigitalSignatureGenerator:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.signature_cache = []
        self._init_cache()
    
    def _init_cache(self):
        """Предварительная генерация подписей"""
        cache_size = self.config.get('signature_cache_size', 30)
        
        for _ in range(cache_size):
            try:
                # Используем параметры из конфига или значения по умолчанию
                width = self.config.get('signature_width', 400)
                height = self.config.get('signature_height', 120)
                
                signature, mask = generate_realistic_signature(width, height)
                if signature is not None and mask is not None:
                    self.signature_cache.append((signature, mask))
            except Exception as e:
                print(f"Signature generation failed: {e}")
                continue
        
        print(f"Generated {len(self.signature_cache)} realistic signatures")
    
    def get_random_signature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает случайную подпись из кэша"""
        if not self.signature_cache:
            self._init_cache()
        
        return random.choice(self.signature_cache)
    
    def apply_signature_to_image(self, 
                               image: np.ndarray, 
                               position: Optional[Tuple[int, int]] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Накладывает подпись на изображение
        
        Args:
            image: исходное изображение (BGR)
            position: позиция для вставки (x, y), None = случайная позиция
        
        Returns:
            image_with_signature: изображение с подписью
            signature_mask: маска подписи
        """
        try:
            h, w = image.shape[:2]
            
            # Получаем случайную подпись
            signature_img, signature_mask = self.get_random_signature()
            sig_h, sig_w = signature_img.shape[:2]
            
            # Масштабирование подписи
            scale_range = self.config.get('scale_range', [0.8, 1.2])
            scale = random.uniform(scale_range[0], scale_range[1])
            new_w = max(30, int(sig_w * scale))
            new_h = max(15, int(sig_h * scale))
            
            signature_img = cv2.resize(signature_img, (new_w, new_h))
            signature_mask = cv2.resize(signature_mask, (new_w, new_h))
            sig_h, sig_w = signature_img.shape[:2]
            
            # Определяем позицию
            if position is None:
                margin_x = int(w * 0.05)
                margin_y = int(h * 0.05)
                max_x = w - sig_w - margin_x
                max_y = h - sig_h - margin_y
                
                if max_x > margin_x and max_y > margin_y:
                    x = random.randint(margin_x, max_x)
                    y = random.randint(margin_y, max_y)
                else:
                    x = max(0, (w - sig_w) // 2)
                    y = max(0, (h - sig_h) // 2)
            else:
                x, y = position
            
            # Гарантируем что подпись влезает
            x = max(0, min(x, w - sig_w))
            y = max(0, min(y, h - sig_h))
            
            # Создаем копии для результата
            result_image = image.copy()
            result_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Область для вставки
            roi = result_image[y:y+sig_h, x:x+sig_w]
            
            if roi.shape[:2] == signature_img.shape[:2]:
                # Смешиваем с прозрачностью
                opacity = self.config.get('opacity', 0.9)
                signature_alpha = signature_mask.astype(float) / 255.0 * opacity
                alpha_3d = np.stack([signature_alpha] * 3, axis=-1)
                
                blended = roi * (1 - alpha_3d) + signature_img * alpha_3d
                result_image[y:y+sig_h, x:x+sig_w] = blended.astype(np.uint8)
                
                # Создаем полную маску
                result_mask[y:y+sig_h, x:x+sig_w] = (signature_mask > 127).astype(np.uint8) * 255
            
            return result_image, result_mask
            
        except Exception as e:
            print(f"Signature application failed: {e}")
            return image.copy(), np.zeros(image.shape[:2], dtype=np.uint8)