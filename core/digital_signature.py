# core/digital_signature.py
"""
Генератор реалистичных подписей.
Каждая подпись создаётся уникально при вызове — кэша нет.

Основные возможности:
- Предустановленные стили (mouse_basic, mouse_smooth, mouse_rapid, pen_clean, pen_pressure, scribble)
- Регистрация и установка пользовательских стилей
- Передача style_override в apply_signature_to_image для разового переопределения
- Реалистичные эффекты: jitter (дрожание), hand-lift (разрывы), variable sampling (скорость движения),
  минимальная/максимальная толщина, антиалиасинг (super-sampling), сглаживание
- Возвращает BGRA-подпись и бинарную маску, а при применении — blended BGR изображение и маску
"""

import random
import hashlib
from typing import Tuple, Optional, Dict, Any, List, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

# -----------------------------
# PREDEFINED STYLES
# -----------------------------
PREDEFINED_STYLES: Dict[str, Dict[str, Any]] = {
    # Мышиный базовый — резкая, с сильной дрожью, без имитации давления (фигня в крапинку)
    "mouse_basic": {
        "antialias_scale": 1,
        "smoothing": 0.0,
        "stroke_thickness_range": [2, 3],
        "pressure_variation": False,
        "jitter_intensity": 4.0,   # пиксели
        "jitter_frequency": 0.9,   # вероятность применить шум в точке
        "steps_per_stroke": 140,
        "hand_lift_chance": 0.15,   # шанс разрыва в штрихе
        "hand_lift_max_segments": 3,
        "ink_color": (0, 0, 0),
        "opacity": 1.0,
        "splat_chance": 0.0,
        "extra_small_strokes_prob": 0.02
    },
    # Мышиный аккуратный — немного сглаженный, слабая дрожь (ну норм, есть толстые)
    "mouse_smooth": {
        "antialias_scale": 1,
        "smoothing": 0.6,
        "stroke_thickness_range": [2, 3],
        "pressure_variation": False,
        "jitter_intensity": 1.0,
        "jitter_frequency": 0.4,
        "steps_per_stroke": 200,
        "hand_lift_chance": 0.08,
        "hand_lift_max_segments": 2,
        "ink_color": (20, 20, 20),
        "opacity": 0.95,
        "splat_chance": 0.0,
        "extra_small_strokes_prob": 0.02
    },
    # Мышиный быстрый — резкие скачки, высокая дискретизация, синий оттенок (фигня в синюю крапинку)
    "mouse_rapid": {
        "antialias_scale": 1,
        "smoothing": 0.0,
        "stroke_thickness_range": [2, 4],
        "pressure_variation": False,
        "jitter_intensity": 5.0,
        "jitter_frequency": 0.95,
        "steps_per_stroke": 120,
        "hand_lift_chance": 0.25,
        "hand_lift_max_segments": 4,
        "ink_color": (0, 0, 80),
        "opacity": 1.0,
        "splat_chance": 0.0,
        "extra_small_strokes_prob": 0.08
    },
    # Чистая ручка (перьевая/стилус) — плавная с вариацией давления (базовый тонкий вариант)
    "pen_clean": {
        "antialias_scale": 2,
        "smoothing": 1.0,
        "stroke_thickness_range": [1, 3],
        "pressure_variation": True,
        "jitter_intensity": 0.6,
        "jitter_frequency": 0.25,
        "steps_per_stroke": 260,
        "hand_lift_chance": 0.05,
        "ink_color": (10, 10, 10),
        "opacity": 0.98,
        "splat_chance": 0.05,
        "extra_small_strokes_prob": 0.25
    },
    # Ручка с вариацией давления (таблетка/граф. планшет) жирные линии с точками норм
    "pen_pressure": {
        "antialias_scale": 2,
        "smoothing": 1.2,
        "stroke_thickness_range": [1, 6],
        "pressure_variation": True,
        "jitter_intensity": 0.8,
        "jitter_frequency": 0.2,
        "steps_per_stroke": 300,
        "hand_lift_chance": 0.07,
        "ink_color": (0, 0, 0),
        "opacity": 1.0,
        "splat_chance": 0.12,
        "extra_small_strokes_prob": 0.35
    },
    # Каракуля/scribble — сильная неряшливость, много мелких штрихов (каракули и грязь + точки штрихи)
    "scribble": {
        "antialias_scale": 1,
        "smoothing": 0.0,
        "stroke_thickness_range": [1, 3],
        "pressure_variation": False,
        "jitter_intensity": 3.0,
        "jitter_frequency": 0.8,
        "steps_per_stroke": 120,
        "hand_lift_chance": 0.35,
        "hand_lift_max_segments": 6,
        "ink_color": (0, 0, 0),
        "opacity": 1.0,
        "splat_chance": 0.05,
        "extra_small_strokes_prob": 0.6
    },
    "paint_pencil": {
        "antialias_scale": 1,            # Paint-карандаш почти без сглаживания
        "smoothing": 0.0,               # чуть-чуть, чтобы линии были менее пиксельные
        "stroke_thickness_range": [1, 1],  # карандаш тонкий
        "pressure_variation": False,     # в Paint карандаш постоянной толщины
        "jitter_intensity": 0.6,         # легкая дрожь, как у мышки
        "jitter_frequency": 0.45,        # не на каждой точке, но часто
        "steps_per_stroke": 400,         # более плавные ходы, меньше рваностей
        "hand_lift_chance": 0.0,        # иногда прерывает линию — поведение Paint
        "hand_lift_max_segments": 0,     # не слишком часто
        "ink_color": (0, 0, 0),       # серый карандашный цвет (не чёрный!)
        "opacity": 1.0,                 # лёгкая прозрачность, как графит
        "splat_chance": 0.0,             # никаких клякс — карандаш не делает брызги
        "extra_small_strokes_prob": 0.0 # совсем немного дополнительных штрихов
    }
}


# -----------------------------
# HELPERS
# -----------------------------
def _random_control_points(width: int, height: int, complexity: float = 1.0) -> np.ndarray:
    """Генерация 4 контрольных точек (кубический Безье)"""
    margin_x = max(1, int(width * 0.03))
    margin_y = max(1, int(height * 0.08))
    p0 = [random.randint(margin_x, max(margin_x + 1, width // 8)), random.randint(height // 4, 3 * height // 4)]
    p1 = [random.randint(width // 8, max(width // 8 + 1, int(width * 0.5 * complexity))), random.randint(0, height)]
    p2 = [random.randint(int(width * 0.4), max(int(width * 0.4) + 1, int(width * (0.6 + 0.2 * complexity)))), random.randint(0, height)]
    p3 = [random.randint(int(width * 0.7), width - margin_x), random.randint(height // 4, 3 * height // 4)]
    return np.array([p0, p1, p2, p3], dtype=float)


def _bezier_points(points: np.ndarray, steps: int, variable_sampling: bool = True) -> np.ndarray:
    """
    Возвращает набор точек вдоль кубического Безье.
    variable_sampling: делает распределение t неравномерным, имитируя изменение скорости.
    """
    if not variable_sampling:
        t = np.linspace(0.0, 1.0, steps)
    else:
        # создаём базовый равномерный т и затем искажает его: более короткие интервалы (медленнее) и длинные (быстрее)
        base = np.linspace(0.0, 1.0, steps * 3)
        # random warping: применяем случайный нелинейный warp через cumulative sum
        noise = np.random.normal(scale=0.6, size=base.shape)
        weights = np.abs(noise) + 0.1
        cum = np.cumsum(weights)
        cum = (cum - cum.min()) / (cum.max() - cum.min())
        # подвыбор для требуемого числа точек (убираем дубли)
        idx = np.linspace(0, len(cum) - 1, steps).astype(int)
        t = cum[idx]
    p0, p1, p2, p3 = points
    t = t.reshape(-1, 1)
    pts = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
    return pts


def _add_ink_splat(draw: ImageDraw.ImageDraw, x: int, y: int, size: int) -> None:
    """Рисуем пятно и несколько разбрызгиваний"""
    bbox = [x - size, y - size, x + size, y + size]
    draw.ellipse(bbox, fill=0)
    for _ in range(random.randint(2, 6)):
        rx = x + random.randint(-size, size)
        ry = y + random.randint(-size, size)
        r = random.randint(1, max(1, size // 4))
        draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=0)


# -----------------------------
# CORE: generate_realistic_signature
# -----------------------------
def generate_realistic_signature(width: int = 400,
                                 height: int = 120,
                                 style: Optional[Dict[str, Any]] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерирует уникальную подпись и маску для данного стиля.

    Возвращает:
      signature_bgra: numpy.ndarray (H,W,4) — B,G,R,Alpha
      mask: numpy.ndarray (H,W) uint8 — бинарная маска чернил (0/255)
    style: словарь опций (см. PREDEFINED_STYLES)
    """
    # базовые значения (шляпа)
    cfg: Dict[str, Any] = {
        "num_strokes_range": [2, 5],
        "stroke_thickness_range": [2, 5],
        "pressure_variation": True,
        "ink_color": (0, 0, 0),
        "opacity": 1.0,
        "antialias_scale": 2,
        "smoothing": 0.8,
        "jitter_intensity": 1.0,
        "jitter_frequency": 0.3,
        "steps_per_stroke": 220,
        "hand_lift_chance": 0.1,
        "hand_lift_max_segments": 3,
        "splat_chance": 0.1,
        "splat_size_range": [6, 18],
        "extra_small_strokes_prob": 0.25,
        "variable_sampling": True,
        "seed": None
    }

    style = PREDEFINED_STYLES['paint_pencil']#['mouse_rapid']
    if style:
        cfg.update(style)

    # reproducibility optional
    if cfg.get("seed") is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    scale = max(1, int(cfg.get("antialias_scale", 2)))
    W, H = width * scale, height * scale
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)

    num_strokes = random.randint(int(cfg["num_strokes_range"][0]), int(cfg["num_strokes_range"][1]))

    for stroke_idx in range(num_strokes):
        points = _random_control_points(W, H, complexity=1.0 + random.random() * 0.6)
        pts = _bezier_points(points, steps=int(cfg["steps_per_stroke"]), variable_sampling=cfg.get("variable_sampling", True))

        # Имитация hand-lift: разобьём pts на несколько сегментов (с шансом)
        segments = [(0, len(pts))]
        if random.random() < float(cfg.get("hand_lift_chance", 0.0)):
            max_seg = int(cfg.get("hand_lift_max_segments", 3))
            n_seg = random.randint(2, max_seg)
            indices = sorted(random.sample(range(1, len(pts) - 1), n_seg - 1))
            segs = []
            prev = 0
            for idx in indices:
                segs.append((prev, idx))
                prev = idx
            segs.append((prev, len(pts)))
            segments = segs

        # основная толщина
        base_thickness = random.randint(int(cfg["stroke_thickness_range"][0] * scale),
                                        int(cfg["stroke_thickness_range"][1] * scale))
        pressure_variation = bool(cfg.get("pressure_variation", True))
        jitter_intensity = float(cfg.get("jitter_intensity", 1.0)) * scale
        jitter_freq = float(cfg.get("jitter_frequency", 0.3))

        for seg_idx, (s_start, s_end) in enumerate(segments):
            # пропускаем небольшой gap между сегментами (hand lift) — не рисуем переход
            seg_pts = pts[s_start:s_end]
            if seg_pts.shape[0] < 2:
                continue

            # jitter: добавляем шум по точкам с вероятностью jitter_freq
            if jitter_intensity > 0:
                noise_mask = np.random.rand(*seg_pts.shape[:1]) < jitter_freq
                noise = np.zeros_like(seg_pts)
                noise[noise_mask] = np.random.normal(scale=jitter_intensity, size=(noise_mask.sum(), 2))
                seg_pts = seg_pts + noise

            for i, (px, py) in enumerate(seg_pts.astype(int)):
                t = i / max(1, (len(seg_pts) - 1))
                if pressure_variation:
                    # имитация легкой вариации давления
                    pressure = 0.7 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * (t + random.random() * 0.2)))
                else:
                    pressure = 1.0
                r = max(1, int(base_thickness * (0.6 + 0.8 * pressure)))
                draw.ellipse([px - r, py - r, px + r, py + r], fill=0)

            # иногда добавляем маленькие вспомогательные штрихи внутри сегмента
            if random.random() < float(cfg.get("extra_small_strokes_prob", 0.25)):
                extra_steps = random.randint(20, 80)
                extra_points = _bezier_points(_random_control_points(W, H, complexity=0.5), steps=extra_steps,
                                              variable_sampling=False)
                for (px, py) in extra_points.astype(int):
                    rr = max(1, int(base_thickness * 0.3))
                    draw.ellipse([px - rr, py - rr, px + rr, py + rr], fill=0)

        # иногда пятна чернил
        if random.random() < float(cfg.get("splat_chance", 0.0)):
            mid_idx = len(pts) // 2 + random.randint(-8, 8)
            mx, my = pts[mid_idx].astype(int)
            spl_min, spl_max = cfg.get("splat_size_range", [6, 18])
            _add_ink_splat(draw, mx, my, random.randint(int(spl_min * scale), int(spl_max * scale)))

    # Сглаживание (Gaussian blur) для антиалиасинга
    blur_radius = float(cfg.get("smoothing", 0.0)) * scale
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Суперсемплинг назад
    img_small = img.resize((width, height), resample=Image.LANCZOS) if scale > 1 else img
    signature_gray = np.array(img_small)
    mask = (255 - signature_gray).clip(0, 255).astype(np.uint8)

    # BGRA составление (цвет + альфа)
    ink_rgb = tuple(int(c) for c in cfg.get("ink_color", (0, 0, 0)))
    opacity = float(cfg.get("opacity", 1.0))
    h, w = mask.shape
    b = np.full((h, w), ink_rgb[2], dtype=np.uint8)
    g = np.full((h, w), ink_rgb[1], dtype=np.uint8)
    r = np.full((h, w), ink_rgb[0], dtype=np.uint8)
    alpha = (mask.astype(np.float32) * opacity).clip(0, 255).astype(np.uint8)
    signature_bgra = np.stack([b, g, r, alpha], axis=-1)

    return signature_bgra, mask


# -----------------------------
# DigitalSignatureGenerator (без кэша)
# -----------------------------
class DigitalSignatureGenerator:
    """
    Генератор подписей без кэширования — каждая подпись уникальна.
    Использование:
      gen = DigitalSignatureGenerator(config)
      gen.set_style('mouse_basic')  # или gen.set_style(dict)
      res_img, mask = gen.apply_signature_to_image(image, position=(x,y), style_override=...)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config.copy() if config else {}
        # реестр стилей
        self._styles: Dict[str, Dict[str, Any]] = PREDEFINED_STYLES.copy()
        # активный стиль — имя или dict
        default = self.config.get("default_style", "mouse_basic")
        self.active_style_name: Optional[str] = None
        self.active_style: Optional[Dict[str, Any]] = None
        self.set_style(default)

    # ---- стиль API ----
    def register_style(self, name: str, style: Dict[str, Any]) -> None:
        """Зарегистрировать или перезаписать пресет стиля"""
        if not isinstance(name, str):
            raise TypeError("Style name must be str")
        self._styles[name] = style.copy()

    def set_style(self, name_or_style: Optional[Union[str, Dict[str, Any]]]) -> None:
        """
        Установить активный стиль:
         - строка: имя зарегистрированного пресета
         - dict: inline-стиль
         - None: сброс на default из config или 'mouse_basic'
        """
        if isinstance(name_or_style, str):
            if name_or_style not in self._styles:
                raise ValueError(f"Style '{name_or_style}' not found")
            self.active_style_name = name_or_style
            self.active_style = self._styles[name_or_style].copy()
        elif isinstance(name_or_style, dict):
            self.active_style_name = None
            self.active_style = name_or_style.copy()
        elif name_or_style is None:
            ds = self.config.get("default_style", "mouse_basic")
            if isinstance(ds, str) and ds in self._styles:
                self.active_style_name = ds
                self.active_style = self._styles[ds].copy()
            elif isinstance(ds, dict):
                self.active_style_name = None
                self.active_style = ds.copy()
            else:
                self.active_style_name = "mouse_basic"
                self.active_style = self._styles["mouse_basic"].copy()
        else:
            raise TypeError("set_style expects str, dict or None")

    def get_style(self) -> Dict[str, Any]:
        """Возвращает текущий активный стиль (словарь)"""
        return self.active_style.copy() if self.active_style else {}

    # ---- генерация и применение ----
    def generate_signature(self,
                           width: Optional[int] = None,
                           height: Optional[int] = None,
                           style_override: Optional[Dict[str, Any]] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сгенерировать уникальную подпись (BGRA, mask).
        width/height берутся из config если не заданы.
        """
        w = int(width or self.config.get("signature_width", 400))
        h = int(height or self.config.get("signature_height", 120))
        style = self.get_style()
        if style_override:
            style = {**style, **style_override}
        return generate_realistic_signature(w, h, style)

    def apply_signature_to_image(self,
                                 image: np.ndarray,
                                 position: Optional[Tuple[int, int]] = None,
                                 style_override: Optional[Dict[str, Any]] = None
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сгенерировать подпись (уникальную) и наложить на BGR изображение.
        Возвращает (result_image (BGR uint8), result_mask (H,W uint8 0/255)).
        """
        img_h, img_w = image.shape[:2]
        sig_bgra, sig_mask = self.generate_signature(
            width=self.config.get("signature_width", 400),
            height=self.config.get("signature_height", 120),
            style_override=style_override
        )
        sig_h, sig_w = sig_mask.shape

        # масштаб (scale_range в конфиге)
        scale_range = self.config.get("scale_range", [0.8, 1.2])
        scale = random.uniform(float(scale_range[0]), float(scale_range[1])) if scale_range else 1.0
        new_w = max(8, int(sig_w * scale))
        new_h = max(6, int(sig_h * scale))

        sig_bgra = cv2.resize(sig_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sig_mask = cv2.resize(sig_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        sig_h, sig_w = sig_mask.shape

        # позиция
        if position is None:
            margin_x = int(img_w * 0.03)
            margin_y = int(img_h * 0.03)
            max_x = max(margin_x, img_w - sig_w - margin_x)
            max_y = max(margin_y, img_h - sig_h - margin_y)
            x = random.randint(margin_x, max_x)
            y = random.randint(margin_y, max_y)
        else:
            x, y = position
        x = max(0, min(x, img_w - sig_w))
        y = max(0, min(y, img_h - sig_h))

        result = image.copy()

        sig_b = sig_bgra[:, :, 0].astype(np.float32)
        sig_g = sig_bgra[:, :, 1].astype(np.float32)
        sig_r = sig_bgra[:, :, 2].astype(np.float32)
        sig_a = sig_bgra[:, :, 3].astype(np.float32) / 255.0

        roi = result[y:y + sig_h, x:x + sig_w].astype(np.float32)
        alpha_3c = np.stack([sig_a, sig_a, sig_a], axis=-1)
        src_rgb = np.stack([sig_b, sig_g, sig_r], axis=-1)

        blended = roi * (1 - alpha_3c) + src_rgb * alpha_3c
        result[y:y + sig_h, x:x + sig_w] = blended.clip(0, 255).astype(np.uint8)

        result_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        binary_sig = (sig_mask > 127).astype(np.uint8) * 255
        result_mask[y:y + sig_h, x:x + sig_w] = binary_sig

        return result, result_mask

    # Утилита: сохранить сгенерированную подпись в PNG (с прозрачностью)
    def save_signature_png(self,
                           path: str,
                           width: Optional[int] = None,
                           height: Optional[int] = None,
                           style_override: Optional[Dict[str, Any]] = None) -> None:
        sig_bgra, _ = self.generate_signature(width=width, height=height, style_override=style_override)
        # BGRA -> RGBA для PIL
        rgba = np.dstack([sig_bgra[:, :, 2], sig_bgra[:, :, 1], sig_bgra[:, :, 0], sig_bgra[:, :, 3]])
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(path, format="PNG")

# -----------------------------
# Пример использования (комментарии)
# -----------------------------
# from core.digital_signature import DigitalSignatureGenerator
# import cv2
# cfg = {
#     "signature_width": 480,
#     "signature_height": 140,
#     "scale_range": [0.7, 1.1],
#     "default_style": "mouse_basic"
# }
# gen = DigitalSignatureGenerator(cfg)
# # Сгенерировать и применить подпись:
# img = cv2.imread("photo.jpg")  # BGR
# res, mask = gen.apply_signature_to_image(img)  # уникальная подпись
# cv2.imwrite("photo_signed.jpg", res)
#
# # Применить стиль rapid только для этой вставки:
# res2, mask2 = gen.apply_signature_to_image(img, style_override={"ink_color": (0,0,60), "jitter_intensity": 6.0})
#
# # Сохранить подпись в PNG:
# gen.save_signature_png("signature_demo.png", width=600, height=160, style_override={"ink_color": (10,10,120)})
