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


    @staticmethod
    def get_random_patch(image, cfg):
        """
           Генерирует случайный патч где угодно по изображению.
           Размеры патча задаются из диапазонов cfg['patch_width_range'], cfg['patch_height_range'].
        """
        image_h, image_w = image.shape[:2]

        # размеры патча
        patch_w = random.randint(*cfg['patch_width_range'])
        patch_h = random.randint(*cfg['patch_height_range'])

        # случайная позиция
        x = random.randint(0, image_w - patch_w)
        y = random.randint(0, image_h - patch_h)

        return image[y:y+patch_h, x:x+patch_w].copy(), {"bbox": [x, y, patch_w, patch_h]}

    
    @staticmethod
    def get_random_patch_like_bbox(image, cfg, bbox):
        image_h, image_w = image.shape[:2]
        bx, by, bw, bh = bbox["bbox"]

        pct_w = random.uniform(*cfg.get('patch_width_pct_range', (0.6, 0.9)))
        pct_h = random.uniform(*cfg.get('patch_height_pct_range', (0.6, 0.9)))

        patch_w = max(1, int(bw * pct_w))
        patch_h = max(1, int(bh * pct_h))

        x = random.randint(0, image_w - patch_w)
        y = random.randint(0, image_h - patch_h)

        return image[y:y+patch_h, x:x+patch_w].copy(), {"bbox": [x, y, patch_w, patch_h]}


    @staticmethod
    def get_patch_inside_bbox(image, bbox, cfg, bbox2=None):
        """
           Генерирует случайный патч ВНУТРИ указанного бокса.
           Размеры ПАТЧА — проценты от размера бокса.
        """
        image_h, image_w = image.shape[:2]
        bx, by, bw, bh = bbox["bbox"]

        if bbox2:
            bx2, by2, bw2, bh2 = bbox2["bbox"]
            bw, bh = min(bw, bw2), min(bh, bh2)

        # проценты → реальные размеры
        pct_w = random.uniform(*cfg.get('patch_width_pct_range', (0.6, 0.9)))
        pct_h = random.uniform(*cfg.get('patch_height_pct_range', (0.6, 0.9)))

        patch_w = max(1, int(bw * pct_w))
        patch_h = max(1, int(bh * pct_h))

        # чтобы патч был внутри бокса
        x = random.randint(bx, bx + bw - patch_w)
        y = random.randint(by, by + bh - patch_h)

        return image[y:y+patch_h, x:x+patch_w].copy(), {"bbox": [x, y, patch_w, patch_h]}


    @staticmethod
    def get_patch_outside_bboxes(image, bboxes, cfg, target_bbox):
        """
        Генерирует случайный патч, который НЕ пересекается ни с одним боксом.
        Если не нашёл — падает обратно на функцию 1.
        """

        image_h, image_w = image.shape[:2]
        
        tx, ty, tw, th = target_bbox["bbox"]

        pct_w = random.uniform(*cfg.get('patch_width_pct_range', (0.7, 1.0)))
        pct_h = random.uniform(*cfg.get('patch_height_pct_range', (0.7, 1.0)))

        patch_w = max(1, int(tw * pct_w))
        patch_h = max(1, int(th * pct_h))

        # min_w, max_w = cfg.get('patch_width_pct_range', (40, 80))
        # min_h, max_h = cfg.get('patch_height_pct_range', (40, 50))

        # max_w = min(max_w, tw)  # патч не шире target_bbox
        # max_h = min(max_h, th)  # патч не выше target_bbox

        # генерируем размер патча
        # patch_w = min(image_w, random.randint(*cfg.get('patch_width_range', (40, 80))))
        # patch_h = min(image_h, random.randint(*cfg.get('patch_height_range', (40, 50))))

        # patch_w = random.randint(max(min_w, 1), max_w)
        # patch_h = random.randint(max(min_h, 1), max_h)
        
        # patch_w = min(patch_w, image_w)
        # patch_h = min(patch_h, image_h)

        # пытаемся найти место
        for _ in range(50):
            x = random.randint(0, image_w - patch_w)
            y = random.randint(0, image_h - patch_h)

            # проверяем пересечение
            intersects = False
            for bbox in bboxes:
                bx, by, bw, bh = bbox["bbox"]

                if not (x + patch_w < bx or x > bx + bw or
                        y + patch_h < by or y > by + bh):
                    intersects = True
                    break

            if not intersects:
                return image[y:y+patch_h, x:x+patch_w].copy(), {"bbox": [x, y, patch_w, patch_h]}

        # fallback
        return get_random_patch_like_bbox(image, cfg, target_bbox)

    @staticmethod
    def paste_patch_random_place(image, patch):
        image_h, image_w = image.shape[:2]
        patch_h, patch_w = patch.shape[:2]

        x = random.randint(0, image_w - patch_w)
        y = random.randint(0, image_h - patch_h)

        target_image = image.copy()

        target_image[y:y+patch_h, x:x+patch_w] = patch

        return target_image, {"bbox": [x, y, patch_w, patch_h]}

    @staticmethod
    def paste_patch_into_bbox(image, patch, bbox):
        """
        Вставляет patch внутрь bbox случайным образом.
        Гарантируется, что patch меньше bbox.

        image : np.ndarray (H, W, 3)
        patch : np.ndarray (ph, pw, 3)
        bbox  : словарь {"bbox": [x, y, w, h]}

        Возвращает: новое изображение
        """
        x, y, bw, bh = bbox["bbox"]

        H, W = image.shape[:2]
        ph, pw = patch.shape[:2]

        max_x = x + bw - pw
        max_y = y + bh - ph

        place_x = random.randint(x, max_x)
        place_y = random.randint(y, max_y)

        target_image = image.copy()

        target_image[place_y:place_y+ph, place_x:place_x+pw] = patch

        return target_image, {"bbox": [place_x, place_y, pw, ph]}
