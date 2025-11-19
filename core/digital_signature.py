# core/digital_signature.py
import cv2
import numpy as np
import random
from typing import Tuple, List, Optional

class DigitalSignatureGenerator:
    """
    Генератор компьютерных подписей для создания подделок
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.signature_cache = []
        self._init_cache()
    
    def _init_cache(self):
        """Предварительная генерация подписей для производительности"""
        cache_size = self.config.get('signature_cache_size', 20)
        for _ in range(cache_size):
            signature, mask = self._generate_single_signature()
            self.signature_cache.append((signature, mask))
    
    def _generate_smooth_curve(self, width: int, height: int, points: int) -> List[Tuple[int, int]]:
        """Генерация сглаженной кривой как от мыши"""
        x = np.linspace(0, width * 0.8, points)
        y_base = np.sin(x / (width * 0.8) * 4 * np.pi) * (height * 0.3)
        
        # Добавляем случайные колебания
        noise = np.random.normal(0, height * 0.05, points)
        y = y_base + noise + height // 2
        
        # Сглаживание
        y_smooth = np.convolve(y, np.ones(5)/5, mode='same')
        
        return [(int(x[i]), int(y_smooth[i])) for i in range(points)]
    
    def _generate_single_signature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация одной подписи"""
        # Случайный размер подписи
        width = random.randint(150, 300)
        height = random.randint(60, 120)
        
        # Создаем белый холст
        signature = np.ones((height, width, 3), dtype=np.uint8) * 255
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Стиль подписи
        thickness = random.randint(1, 2)
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        
        # Генерируем линии подписи
        num_lines = random.randint(2, 4)
        
        for _ in range(num_lines):
            points = random.randint(3, 6)
            line_points = self._generate_smooth_curve(width, height, points)
            
            # Рисуем линию с антиалиасингом
            for i in range(len(line_points) - 1):
                pt1, pt2 = line_points[i], line_points[i + 1]
                cv2.line(signature, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
                cv2.line(mask, pt1, pt2, 255, thickness, lineType=cv2.LINE_AA)
        
        # Добавляем случайные точки
        if random.random() > 0.5:
            for _ in range(random.randint(1, 3)):
                x = random.randint(10, width - 10)
                y = random.randint(10, height - 10)
                cv2.circle(signature, (x, y), 2, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(mask, (x, y), 2, 255, -1, lineType=cv2.LINE_AA)
        
        return signature, mask
    
    def get_random_signature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает случайную подпись из кэша"""
        if not self.signature_cache:
            self._init_cache()
        return random.choice(self.signature_cache)
    
    def apply_signature_to_image(self, 
                               image: np.ndarray, 
                               position: Optional[Tuple[int, int]] = None,
                               signature: Optional[Tuple[np.ndarray, np.ndarray]] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Накладывает подпись на изображение
        
        Args:
            image: исходное изображение
            position: (x, y) позиция для вставки (None = случайная)
            signature: готовая подпись (None = генерируется случайная)
        
        Returns:
            image_with_signature: изображение с подписью
            signature_mask: маска подписи
        """
        h, w = image.shape[:2]
        
        # Получаем подпись
        if signature is None:
            signature_img, signature_mask = self.get_random_signature()
        else:
            signature_img, signature_mask = signature
        
        sig_h, sig_w = signature_img.shape[:2]
        
        # Определяем позицию
        if position is None:
            # Случайная позиция с отступами
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            max_x = w - sig_w - margin_x
            max_y = h - sig_h - margin_y
            
            if max_x > margin_x and max_y > margin_y:
                x = random.randint(margin_x, max_x)
                y = random.randint(margin_y, max_y)
            else:
                # Если изображение маленькое - центрируем
                x = (w - sig_w) // 2
                y = (h - sig_h) // 2
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
            # Смешиваем подпись с изображением
            alpha = signature_mask.astype(float) / 255.0
            alpha_3d = np.stack([alpha] * 3, axis=-1)
            
            blended = roi * (1 - alpha_3d) + signature_img * alpha_3d
            result_image[y:y+sig_h, x:x+sig_w] = blended.astype(np.uint8)
            
            # Создаем полную маску
            result_mask[y:y+sig_h, x:x+sig_w] = signature_mask
        
        return result_image, result_mask
