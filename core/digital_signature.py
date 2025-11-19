# core/digital_signature.py
import cv2
import numpy as np
import random
from typing import Tuple, List, Optional
import math

class DigitalSignatureGenerator:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.signature_cache = []
        self._init_cache()
    
    def _init_cache(self):
        """Предварительная генерация реалистичных подписей"""
        cache_size = self.config.get('signature_cache_size', 50)
        
        for i in range(cache_size):
            try:
                signature, mask = self._generate_realistic_signature()
                if signature is not None and mask is not None:
                    self.signature_cache.append((signature, mask))
            except Exception as e:
                print(f"Signature generation failed: {e}")
                continue
        
        print(f"Generated {len(self.signature_cache)} signatures")
    
    def _generate_signature_style(self) -> dict:
        """Генерация уникального стиля для каждой подписи на основе конфига"""
        cfg = self.config.get('digital_signature', {})
        
        return {
            'base_amplitude': random.uniform(*cfg.get('base_amplitude_range', [0.2, 0.4])),
            'noise_level': random.uniform(*cfg.get('noise_level_range', [0.02, 0.08])),
            'smoothness': random.uniform(*cfg.get('smoothness_range', [0.7, 0.95])),
            'thickness_variation': random.uniform(*cfg.get('thickness_variation_range', [0.1, 0.3])),
            'color_variance': random.randint(*cfg.get('color_variance_range', [10, 40])),
            'complexity': random.choice(cfg.get('complexity_options', ['simple', 'medium', 'complex'])),
            'slant': random.uniform(*cfg.get('slant_range', [-0.3, 0.3])),
            'speed_effect': random.uniform(*cfg.get('speed_effect_range', [0.1, 0.5])),
        }
    
    def _generate_handwriting_curve(self, width: int, height: int, style: dict) -> List[Tuple[int, int]]:
        """Генерация кривой, имитирующей реальный почерк"""
        # Реалистичное количество точек для почерка
        points_count = random.randint(10, 20)
        
        # Базовые точки с естественным распределением
        base_x = np.linspace(width * 0.1, width * 0.9, points_count)
        
        # Основной ритм подписи - комбинация нескольких волн
        main_freq = random.uniform(1.0, 2.0)
        main_wave = np.sin(base_x / width * 2 * math.pi * main_freq)
        
        # Вторичные волны для естественности
        secondary_freq = random.uniform(3.0, 5.0)
        secondary_wave = np.cos(base_x / width * 2 * math.pi * secondary_freq) * 0.3
        
        # Третичные волны для мелких деталей
        detail_freq = random.uniform(8.0, 12.0)
        detail_wave = np.sin(base_x / width * 2 * math.pi * detail_freq) * 0.1
        
        # Комбинируем волны
        combined_wave = (main_wave * 0.6 + secondary_wave * 0.3 + detail_wave * 0.1)
        
        # Применяем амплитуду и центрируем
        y_center = height // 2
        y_base = combined_wave * height * style['base_amplitude'] + y_center
        
        # Добавляем естественные колебания почерка
        handwriting_tremor = np.random.normal(0, height * 0.02, points_count)
        y_base += handwriting_tremor
        
        # Применяем наклон
        for i in range(len(base_x)):
            y_base[i] += base_x[i] * style['slant']
        
        # Ограничиваем координаты
        y_final = np.clip(y_base, height * 0.1, height * 0.9)
        
        return [(int(base_x[i]), int(y_final[i])) for i in range(points_count)]
    
    def _generate_signature_elements(self, width: int, height: int, style: dict) -> List[List[Tuple[int, int]]]:
        """Генерация элементов подписи с реалистичными параметрами"""
        elements = []
        cfg = self.config.get('digital_signature', {})
        
        try:
            # Основная линия подписи
            main_line = self._generate_handwriting_curve(width, height, style)
            elements.append(main_line)
            
            # Дополнительные элементы в зависимости от сложности
            if style['complexity'] in ['medium', 'complex']:
                # Подчеркивание
                if random.random() < cfg.get('underline_prob', 0.3):
                    underline_y = height - random.randint(5, 15)
                    # Естественное подчеркивание с небольшими изгибами
                    underline_points = [
                        (int(width * 0.1), underline_y),
                        (int(width * 0.4), underline_y + random.randint(-2, 2)),
                        (int(width * 0.6), underline_y + random.randint(-2, 2)),
                        (int(width * 0.9), underline_y)
                    ]
                    elements.append(underline_points)
                
                # Начальные элементы (инициалы или завитки)
                if random.random() < cfg.get('initial_element_prob', 0.4):
                    # Создаем небольшой начальный элемент
                    start_x = random.randint(5, 15)
                    start_y = main_line[0][1] + random.randint(-10, 10)
                    initial_points = [(start_x, start_y)]
                    
                    # Небольшой завиток или дополнительная черта
                    for i in range(2):
                        last_x, last_y = initial_points[-1]
                        new_x = last_x + random.randint(8, 15)
                        new_y = last_y + random.randint(-8, 8)
                        initial_points.append((new_x, new_y))
                    
                    elements.append(initial_points)
                
                # Конечные росчерки
                if random.random() < cfg.get('flourish_prob', 0.5) and main_line:
                    end_x, end_y = main_line[-1]
                    flourish_points = [(end_x, end_y)]
                    
                    # Естественный росчерк - не слишком длинный
                    for i in range(random.randint(2, 4)):
                        last_x, last_y = flourish_points[-1]
                        # Росчерк обычно идет вправо и немного вниз/вверх
                        new_x = last_x + random.randint(10, 20)
                        new_y = last_y + random.randint(-10, 5)
                        flourish_points.append((new_x, new_y))
                    
                    elements.append(flourish_points)
            
            # Точки над i или другие маркеры
            if random.random() < cfg.get('dots_prob', 0.3):
                for _ in range(random.randint(1, 2)):
                    dot_x = random.randint(int(width * 0.3), int(width * 0.7))
                    dot_y = random.randint(int(height * 0.2), int(height * 0.4))
                    # Точка - это маленький отрезок или круг
                    dot_size = random.randint(2, 4)
                    elements.append([(dot_x, dot_y), (dot_x + dot_size, dot_y)])
        
        except Exception as e:
            print(f"Signature elements generation failed: {e}")
        
        return elements
    
    def _generate_realistic_signature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация одной реалистичной подписи"""
        cfg = self.config.get('digital_signature', {})
        
        try:
            # Размеры из конфига
            width_options = cfg.get('width_options', [180, 200, 220, 250, 280, 300])
            height_options = cfg.get('height_options', [50, 60, 70, 80, 90, 100])
            
            width = random.choice(width_options)
            height = random.choice(height_options)
            
            # Стиль подписи
            style = self._generate_signature_style()
            
            # Создаем холст
            signature = np.ones((height, width, 3), dtype=np.uint8) * 255
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Цвет из конфига - темные оттенки для реалистичности
            base_color_range = cfg.get('base_color_range', [10, 40])
            base_color = random.randint(base_color_range[0], base_color_range[1])
            color_variation = style['color_variance']
            
            color = (
                max(0, min(80, base_color + random.randint(-color_variation, color_variation))),
                max(0, min(80, base_color + random.randint(-color_variation, color_variation))),
                max(0, min(80, base_color + random.randint(-color_variation, color_variation)))
            )
            
            # Генерируем элементы подписи
            elements = self._generate_signature_elements(width, height, style)
            
            # Толщина линии из конфига
            thickness_options = cfg.get('base_thickness_options', [2, 3])
            base_thickness = random.choice(thickness_options)
            
            # Рисуем элементы с естественной вариацией толщины
            for element in elements:
                if len(element) > 1:
                    current_thickness = base_thickness
                    
                    for i in range(len(element) - 1):
                        pt1, pt2 = element[i], element[i + 1]
                        
                        # Естественное изменение толщины в разных частях подписи
                        if random.random() < 0.3:  # 30% chance to vary thickness
                            current_thickness = max(1, base_thickness + random.randint(-1, 1))
                        
                        # Рисуем с антиалиасингом для плавности
                        cv2.line(signature, pt1, pt2, color, current_thickness, 
                                lineType=cv2.LINE_AA)
                        cv2.line(mask, pt1, pt2, 255, current_thickness, 
                                lineType=cv2.LINE_AA)
            
            # Применяем эффекты
            signature, mask = self._apply_signature_effects(signature, mask, style)
            
            return signature, mask
            
        except Exception as e:
            print(f"Realistic signature generation failed: {e}")
            return self._generate_fallback_signature()
    
    def _generate_fallback_signature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация простой, но реалистичной подписи как fallback"""
        cfg = self.config.get('digital_signature', {})
        width_options = cfg.get('width_options', [180, 200, 220])
        height_options = cfg.get('height_options', [50, 60, 70])
        
        width = random.choice(width_options)
        height = random.choice(height_options)
        
        signature = np.ones((height, width, 3), dtype=np.uint8) * 255
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Базовый цвет
        base_color_range = cfg.get('base_color_range', [10, 40])
        color = (random.randint(base_color_range[0], base_color_range[1]),) * 3
        
        # Простая, но естественная кривая
        points = []
        num_segments = random.randint(3, 5)
        
        for i in range(num_segments + 1):
            x = int(width * i / num_segments)
            base_y = height // 2
            
            # Естественные колебания
            if i == 0:
                y = base_y
            else:
                y_variation = random.randint(-height//4, height//4)
                y = base_y + y_variation
            
            points.append((x, y))
        
        # Рисуем плавными линиями
        for i in range(len(points) - 1):
            thickness = random.choice([2, 3])
            cv2.line(signature, points[i], points[i+1], color, thickness, lineType=cv2.LINE_AA)
            cv2.line(mask, points[i], points[i+1], 255, thickness, lineType=cv2.LINE_AA)
        
        return signature, mask
    
    def _apply_signature_effects(self, signature: np.ndarray, mask: np.ndarray, 
                               style: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Применяет дополнительные эффекты к подписи"""
        cfg = self.config.get('digital_signature', {})
        
        try:
            # Легкое размытие для естественности
            if random.random() < cfg.get('blur_prob', 0.3):
                kernel_options = cfg.get('blur_kernel_options', [1, 3])
                blur_amount = random.choice(kernel_options)
                if blur_amount > 0 and blur_amount % 2 == 1:
                    signature = cv2.GaussianBlur(signature, (blur_amount, blur_amount), 0)
                    mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
                    mask = (mask > 127).astype(np.uint8) * 255
            
            # Легкий шум для имитации сканирования
            if random.random() < cfg.get('noise_prob', 0.4):
                noise_strength_range = cfg.get('noise_strength_range', [0.5, 1.5])
                noise_strength = random.uniform(noise_strength_range[0], noise_strength_range[1])
                noise = np.random.normal(0, noise_strength, signature.shape).astype(np.uint8)
                signature = cv2.add(signature, noise)
        
        except Exception as e:
            print(f"Signature effects failed: {e}")
        
        return signature, mask
    
    def get_random_signature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает случайную подпись"""
        if not self.signature_cache:
            self._init_cache()
        
        return random.choice(self.signature_cache)
    
    def apply_signature_to_image(self, 
                               image: np.ndarray, 
                               position: Optional[Tuple[int, int]] = None
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Накладывает подпись на изображение"""
        cfg = self.config.get('digital_signature', {})
        
        try:
            h, w = image.shape[:2]
            
            # Получаем подпись
            signature_img, signature_mask = self.get_random_signature()
            sig_h, sig_w = signature_img.shape[:2]
            
            # Масштабирование из конфига
            scale_range = cfg.get('scale_range', [0.7, 1.3])
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
                # Прозрачность из конфига
                opacity_range = cfg.get('opacity_range', [0.8, 0.95])
                alpha = random.uniform(opacity_range[0], opacity_range[1])
                signature_alpha = signature_mask.astype(float) / 255.0 * alpha
                alpha_3d = np.stack([signature_alpha] * 3, axis=-1)
                
                blended = roi * (1 - alpha_3d) + signature_img * alpha_3d
                result_image[y:y+sig_h, x:x+sig_w] = blended.astype(np.uint8)
                
                # Создаем маску
                result_mask[y:y+sig_h, x:x+sig_w] = (signature_mask > 127).astype(np.uint8) * 255
            
            return result_image, result_mask
            
        except Exception as e:
            print(f"Signature application failed: {e}")
            return image.copy(), np.zeros(image.shape[:2], dtype=np.uint8)