import os
import cv2
import numpy as np
import random
from typing import Dict, Tuple, List, Optional
from .bbox_processor import BBoxProcessor

from .text_printing_utils import random_text, render_text_into_bbox, font_supports_alphabet

# from simple_lama_inpainting import SimpleLama
# from PIL import Image
# simple_lama = SimpleLama()

from lama import LaMa

model = LaMa('cuda:0')

def inpaint_with_lama3(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # img_path = "image.png"
    # mask_path = "mask.png"

    # image = Image.open(img_path)
    # mask = Image.open(mask_path).convert('L')
    # print("AAAAAAAAAA")
    # print(type(image), type(mask))
    # # print(np.max(mask))
    # image_temp = image.copy()
    # mask_temp = mask.copy()

    # print(image.shape, mask.shape)

    result = model(image, mask)
    result_np = np.array(result)

    if result_np.shape != image.shape:
        result_np = result_np[:image.shape[0], :image.shape[1]]

    # print(np.array(result).shape, image.shape, mask.shape)
    
    # print(type(result), type(image), type(mask))

    # result.save("/home/arseniy/python-dev/forensics/out.png")
    
    # result = cv2.cvtColor(np.array(result), cv2.COLOR_BGR2RGB)
    
    return result_np

def inpaint_with_lama2(image: np.ndarray, mask: np.ndarray) -> np.ndarray:

    

    # img_path = "image.png"
    # mask_path = "mask.png"

    # image = Image.open(img_path)
    # mask = Image.open(mask_path).convert('L')
    # print("AAAAAAAAAA")
    # print(type(image), type(mask))
    # # print(np.max(mask))
    # image_temp = image.copy()
    # mask_temp = mask.copy()

    # print(image.shape, mask.shape)

    result = simple_lama(image, mask)
    result_np = np.array(result)

    if result_np.shape != image.shape:
        result_np = result_np[:image.shape[0], :image.shape[1]]

    # print(np.array(result).shape, image.shape, mask.shape)
    
    # print(type(result), type(image), type(mask))

    # result.save("/home/arseniy/python-dev/forensics/out.png")
    
    # result = cv2.cvtColor(np.array(result), cv2.COLOR_BGR2RGB)
    
    return result_np

def inpaint_with_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Использование LaMa для высококачественного инпейнтинга
    Устанавливается: pip install lama-cleaner
    """
    try:
        from lama_cleaner import LaMa
        
        # Инициализация модели
        model = LaMa(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Преобразование в PIL для обработки
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        
        # Применение инпейнтинга
        result = model(image, mask)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(e)
        return image

from .digital_signature import DigitalSignatureGenerator

class SplicingOperations:
    def __init__(self, sources: Dict, config):
        self.config = config
        self.sources = sources
        self.bbox_processor = BBoxProcessor()
        self.signature_generator = DigitalSignatureGenerator(self.config['splicing']['operations'].get('digital_signature', {}))
        
        # alphabet = load_alphabet(cfg["alphabet_path"])
        self.alphabet = " АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789"

        self._collect_ttf_fonts()

    # В класс SplicingOperations добавьте:
    def digital_signature(self, 
                        base_image: np.ndarray, 
                        base_markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Добавление цифровой подписи к документу
        """
        cfg = self.config['splicing']['operations'].get('digital_signature', {})
        
        if not cfg.get('enabled', True):
            return base_image, np.zeros(base_image.shape[:2], dtype=np.uint8)
        
        # Определяем позицию для подписи
        position = None
        
        if cfg['random_position_prob'] < random.random():
            # Если есть bbox для подписей - используем их
            if 'bboxes' in base_markup and base_markup['bboxes']:
                signature_bboxes = [b for b in base_markup['bboxes'] 
                                  if b.get('category') in ['signature', 'handwriting', 'text']]
                
                if signature_bboxes:
                    chosen_bbox = random.choice(signature_bboxes)
                    x, y, w, h = chosen_bbox['bbox']
                    # Случайная позиция внутри bbox
                    pos_x = x + random.randint(0, max(0, w - 50))
                    pos_y = y + random.randint(0, max(0, h - 30))
                    position = (pos_x, pos_y)
        
        # Применяем подпись
        result_image, signature_mask = self.signature_generator.apply_signature_to_image(
            base_image, position, cfg
        )
        
        return result_image, signature_mask

    def heal_image_with_mask(
        self,
        image: np.ndarray, 
        mask: np.ndarray,
        sample_radius: int = 3,
        opacity: float = 1.0,
        blur_sigma: Optional[float] = None
    ) -> np.ndarray:
        """
        Автоматическое залечивание изображения с использованием маски
        комбинацией inpainting и Poisson blending
        
        Параметры:
            image: исходное изображение (BGR или RGB)
            mask: бинарная маска (белые области = что залечивать)
            sample_radius: радиус для inpainting (чем больше, тем больше область анализа)
            opacity: прозрачность наложения (0.0 - 1.0)
            blur_sigma: sigma для размытия маски (None = авто расчет)
        
        Возвращает:
            Обработанное изображение
        """
        if image is None or mask is None:
            raise ValueError("Image and mask cannot be None")
        
        # Создаем копию изображения
        result = image.copy()
        
        # Преобразуем маску если нужно
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Создаем размытую маску для плавных переходов
        if blur_sigma is None:
            # Автоматический расчет sigma на основе размера маски
            blur_sigma = max(mask.shape) * 0.01
        
        blurred_mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
        
        # Применяем inpainting для генерации содержимого
        inpainted_area = cv2.inpaint(
            image, 
            mask, 
            sample_radius, 
            cv2.INPAINT_TELEA
        )
        
        # Находим центр маски для seamlessClone
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        else:
            # Если не можем найти центр, используем центр bounding rect
            y, x = np.where(mask > 0)
            center_x, center_y = int(np.mean(x)), int(np.mean(y))
        
        # Применяем Poisson blending для бесшовного смешивания
        blended_result = cv2.seamlessClone(
            inpainted_area,    # Сгенерированная область
            image,             # Исходное изображение
            blurred_mask,      # Размытая маска
            (center_x, center_y),  # Центр области
            cv2.NORMAL_CLONE   # Режим смешивания
        )
        
        # Плавное наложение результата
        if opacity == 1.0:
            # Полная замена в области маски
            result[mask > 0] = blended_result[mask > 0]
        else:
            # Линейная интерполяция
            mask_normalized = blurred_mask / 255.0 * opacity
            mask_3d = cv2.merge([mask_normalized, mask_normalized, mask_normalized])
            
            result = (blended_result * mask_3d + result * (1 - mask_3d)).astype(np.uint8)
        
        return result

    def _opencv_inpaint(self, image: np.ndarray, mask: np.ndarray, radius: int, method: str) -> np.ndarray:
        """Вспомогательный метод для вызова OpenCV inpaint."""
        inpaint_method = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        radius = max(1, int(radius))
        # Маска должна быть 8-bit 1 channel, ненулевые пиксели помечают область
        mask_bin = (mask > 0).astype(np.uint8)
        return cv2.inpaint(image, mask_bin, radius, inpaint_method)

    def _get_random_font(self) -> str:
        """
        Возвращает случайный путь к TTF файлу из списка.
        Если список пуст — вызывает исключение.
        """
        if not self.ttf_list:
            raise ValueError("TTF font list is empty!")
        return random.choice(self.ttf_list)

    def _collect_ttf_fonts(self):
        """
        Рекурсивно ищет все файлы .ttf в директории и её поддиректориях.
        Возвращает список абсолютных путей.
        """
        fonts_dir = self.config['generation']['resources_dir']

        self.ttf_list = []
        for root, dirs, files in os.walk(fonts_dir):
            for f in files:
                if f.lower().endswith(".ttf"):
                    full_path = os.path.join(root, f)
                    if font_supports_alphabet(full_path, self.alphabet):
                        self.ttf_list.append(full_path)
        print(f"Успешно загружено {len(self.ttf_list)} шрифтов!")

    def inpaint_borders(self, base_image: np.ndarray, base_markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        h, w = base_image.shape[:2]
        cfg = self.config['splicing']['operations'].get('inpaint_borders', {})

        print_text_prob = cfg.get("print_text_prob", 0.5)
        cfg_text_len = cfg.get("text_len", [1, 5])

        radius = cfg.get('radius', 3)
        method = cfg.get('method', 'telea')

        num_bboxes_cfg = cfg.get('num_bboxes', [1, 1])
        allowed_categories = cfg.get('allowed_categories', None)

        if 'bboxes' not in base_markup or not base_markup['bboxes']:
            return base_image, np.zeros((h, w), dtype=np.uint8)

        bboxes = base_markup['bboxes']
        if allowed_categories:
            bboxes = [b for b in bboxes if b.get('category', 'text') in allowed_categories]
        if not bboxes:
            return base_image, np.zeros((h, w), dtype=np.uint8)

        min_k, max_k = max(1, num_bboxes_cfg[0]), max(1, num_bboxes_cfg[1])
        k = random.randint(min_k, min(max_k, len(bboxes)))
        chosen_bboxes = random.sample(bboxes, k) if len(bboxes) > k else bboxes

        mask = np.zeros((h, w), dtype=np.uint8)

        for bbox in chosen_bboxes:
            bbox_mask = self.bbox_processor.create_bbox_mask((h, w), bbox)
            mask = np.maximum(mask, bbox_mask)

        result_image = self._opencv_inpaint(base_image, mask, radius, method)
        
        # printing text on inpainted image
        for bbox in chosen_bboxes:
            if random.random() < print_text_prob:
                text_len = random.randint(cfg_text_len[0], cfg_text_len[1])
                text = random_text(self.alphabet, text_len)
                font_path = self._get_random_font()
                result_image = render_text_into_bbox(result_image, text, bbox['bbox'], font_path)

        return result_image, mask

    
    def inpaint_random(self, base_image: np.ndarray, base_markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Инпейнтинг в случайных местах с использованием небольших пятен (кружки/прямоугольники).
        """
        h, w = base_image.shape[:2]
        cfg = self.config['splicing']['operations'].get('inpaint_random', {})
        num_spots_range = cfg.get('num_spots', [1, 5])
        spot_size_range = cfg.get('spot_size', [8, 64])  # диаметр/сторона в пикселях
        shapes = cfg.get('shapes', ['circle', 'rect'])
        radius = cfg.get('radius', 3)
        method = cfg.get('method', 'telea')
        
        num_spots = random.randint(num_spots_range[0], num_spots_range[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for _ in range(num_spots):
            size = random.randint(spot_size_range[0], spot_size_range[1])
            shape = random.choice(shapes) if shapes else 'circle'
            # Случайная позиция с запасом, чтобы фигура влезла
            cx = random.randint(0, max(0, w - 1))
            cy = random.randint(0, max(0, h - 1))
            
            if shape == 'rect':
                rw = max(3, size)
                rh = max(3, size // 2 if random.random() < 0.5 else size)
                x1 = max(0, cx - rw // 2)
                y1 = max(0, cy - rh // 2)
                x2 = min(w, x1 + rw)
                y2 = min(h, y1 + rh)
                mask[y1:y2, x1:x2] = 255
            else:
                # circle
                r = max(2, size // 2)
                cv2.circle(mask, (cx, cy), r, 255, thickness=-1)
        
        result_image = self._opencv_inpaint(base_image, mask, radius, method)
        return result_image, mask
    
    def random_patch_copy_paste(self, base_image: np.ndarray, base_markup: Dict,
                                source_image: np.ndarray, source_markup: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Копирует случайный прямоугольный патч из source_image и вставляет его в base_image.
        Место вставки: либо существующий bbox в base_markup, либо случайное место.
        Размер патча берётся из конфига и гарантированно помещается в границы.
        """
        h_src, w_src = source_image.shape[:2]
        h_dst, w_dst = base_image.shape[:2]
        cfg = self.config['splicing']['operations'].get('random_patch_paste', {})
        size_range = cfg.get('patch_size_range', [32, 256])
        insert_into_bbox_prob = cfg.get('insert_into_bbox_prob', 0.5)
        
        min_size = max(2, int(size_range[0]))
        max_size = max(min_size, int(size_range[1]))
        
        # Выбор размера патча, не превышающего размеры source_image
        max_allowed = max(2, min(w_src, h_src))
        side = random.randint(min(min_size, max_allowed), min(max_size, max_allowed))
        patch_w = side
        patch_h = side if random.random() < 0.5 else random.randint(max(2, side // 2), side)
        patch_w = min(patch_w, w_src)
        patch_h = min(patch_h, h_src)
        
        # Случайная позиция в source_image
        max_x_src = max(0, w_src - patch_w)
        max_y_src = max(0, h_src - patch_h)
        x_src = random.randint(0, max_x_src)
        y_src = random.randint(0, max_y_src)
        patch_region = source_image[y_src:y_src+patch_h, x_src:x_src+patch_w].copy()
        
        # Определяем, вставлять в bbox или в случайное место
        use_bbox = False
        chosen_bbox = None
        if random.random() < insert_into_bbox_prob and 'bboxes' in base_markup and base_markup['bboxes']:
            chosen_bbox = random.choice(base_markup['bboxes'])
            use_bbox = True
        
        mask = np.zeros((h_dst, w_dst), dtype=np.uint8)
        if use_bbox and chosen_bbox is not None:
            # Ресайзим патч под bbox и заполняем его
            result_image = self.bbox_processor.paste_bbox_region(base_image, patch_region, chosen_bbox, resize=True)
            mask = self.bbox_processor.create_bbox_mask((h_dst, w_dst), chosen_bbox)
        else:
            # Вставляем в случайное место; подберём координаты так, чтобы влезло
            max_x_dst = max(0, w_dst - patch_w)
            max_y_dst = max(0, h_dst - patch_h)
            x_dst = random.randint(0, max_x_dst)
            y_dst = random.randint(0, max_y_dst)
            insert_bbox = {'bbox': [x_dst, y_dst, patch_w, patch_h]}
            result_image = self.bbox_processor.paste_bbox_region(base_image, patch_region, insert_bbox, resize=True)
            mask = self.bbox_processor.create_bbox_mask((h_dst, w_dst), insert_bbox)
        
        return result_image, mask
    
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
        
        # Извлечение и замена областей
        base_region = self.bbox_processor.extract_bbox_region(base_image, bbox_base)
        target_region = self.bbox_processor.extract_bbox_region(target_image, bbox_target)
        
        # Вставка областей
        result_image = self.bbox_processor.paste_bbox_region(base_image, target_region, bbox_base)
        result_image = self.bbox_processor.paste_bbox_region(result_image, base_region, bbox_target)
        
        # Обновление маски
        mask = np.maximum(mask, self.bbox_processor.create_bbox_mask(base_image.shape, bbox_base))
        mask = np.maximum(mask, self.bbox_processor.create_bbox_mask(base_image.shape, bbox_target))

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
        
        # Извлечение областей
        region1 = self.bbox_processor.extract_bbox_region(base_image, bbox1)
        region2 = self.bbox_processor.extract_bbox_region(base_image, bbox2)
        
        # Замена областей
        result_image = self.bbox_processor.paste_bbox_region(base_image, region1, bbox2)
        result_image = self.bbox_processor.paste_bbox_region(result_image, region2, bbox1)
        
        # Обновление маски
        mask = np.maximum(mask, self.bbox_processor.create_bbox_mask(base_image.shape, bbox1))
        mask = np.maximum(mask, self.bbox_processor.create_bbox_mask(base_image.shape, bbox2))
        
        return result_image, mask
