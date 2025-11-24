# core/digital_signature.py
"""
Генератор реалистичных подписей — каждый вызов уникален.
Параметры генерации задаются единым набором диапазонов (style_ranges) в конфиге.
"""

import random
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2

# -----------------------------
# Helpers: сэмплинг из диапазонов
# -----------------------------
def _sample_scalar(val):
    """Если val — [min,max] — вернуть uniform(min,max). Иначе вернуть val (скаляр)."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        a, b = val
        # если целые — вернуть int
        if isinstance(a, int) and isinstance(b, int):
            return random.randint(int(a), int(b))
        else:
            return float(random.uniform(float(a), float(b)))
    return val


def _sample_int_range(maybe_range):
    """Ожидает либо список [min,max], либо одно число, возвращает tuple(min,max) ints."""
    if isinstance(maybe_range, (list, tuple)) and len(maybe_range) == 2:
        return int(maybe_range[0]), int(maybe_range[1])
    elif isinstance(maybe_range, int):
        return int(maybe_range), int(maybe_range)
    else:
        # fallback
        return 1, 3


def _sample_color(ink_color_range, default=(0, 0, 0)):
    """
    ink_color_range: [[rmin,rmax],[gmin,gmax],[bmin,bmax]] либо фиксированный кортеж.
    """
    if isinstance(ink_color_range, (list, tuple)) and len(ink_color_range) == 3:
        r_rng, g_rng, b_rng = ink_color_range
        r = int(random.randint(int(r_rng[0]), int(r_rng[1])))
        g = int(random.randint(int(g_rng[0]), int(g_rng[1])))
        b = int(random.randint(int(b_rng[0]), int(b_rng[1])))
        return (r, g, b)
    if isinstance(ink_color_range, (list, tuple)) and len(ink_color_range) == 2 and isinstance(ink_color_range[0], int):
        # случай: [min, max] -> градация серого
        v = int(random.randint(int(ink_color_range[0]), int(ink_color_range[1])))
        return (v, v, v)
    if isinstance(ink_color_range, tuple) and len(ink_color_range) == 3:
        return ink_color_range
    return default


# -----------------------------
# Bézier и рисование
# -----------------------------
def _random_control_points(width: int, height: int, complexity: float = 1.0):
    margin_x = max(1, int(width * 0.03))
    margin_y = max(1, int(height * 0.08))
    p0 = [random.randint(margin_x, max(margin_x + 1, width // 8)), random.randint(height // 4, 3 * height // 4)]
    p1 = [random.randint(width // 8, max(width // 8 + 1, int(width * 0.5 * complexity))), random.randint(0, height)]
    p2 = [random.randint(int(width * 0.4), max(int(width * 0.4) + 1, int(width * (0.6 + 0.2 * complexity)))), random.randint(0, height)]
    p3 = [random.randint(int(width * 0.7), width - margin_x), random.randint(height // 4, 3 * height // 4)]
    return np.array([p0, p1, p2, p3], dtype=float)


def _bezier_points(points: np.ndarray, steps: int, variable_sampling: bool = True) -> np.ndarray:
    if not variable_sampling:
        t = np.linspace(0.0, 1.0, steps)
    else:
        base = np.linspace(0.0, 1.0, steps * 3)
        noise = np.random.normal(scale=0.6, size=base.shape)
        weights = np.abs(noise) + 0.1
        cum = np.cumsum(weights)
        cum = (cum - cum.min()) / (cum.max() - cum.min())
        idx = np.linspace(0, len(cum) - 1, steps).astype(int)
        t = cum[idx]
    p0, p1, p2, p3 = points
    t = t.reshape(-1, 1)
    pts = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
    return pts


def _add_ink_splat(draw: ImageDraw.ImageDraw, x: int, y: int, size: int) -> None:
    bbox = [x - size, y - size, x + size, y + size]
    draw.ellipse(bbox, fill=0)
    for _ in range(random.randint(2, 5)):
        rx = x + random.randint(-size, size)
        ry = y + random.randint(-size, size)
        r = random.randint(1, max(1, size // 4))
        draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=0)


# -----------------------------
# generate_realistic_signature (старая логика, но использует style dict)
# -----------------------------
def generate_realistic_signature(width: int = 400,
                                 height: int = 120,
                                 style: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерация подписи, style — словарь с конкретными значениями (не диапазонами).
    """
    cfg = {
        "num_strokes_range": [2, 4],
        "stroke_thickness_range": [1, 1],
        "pressure_variation": False,
        "antialias_scale": 1,
        "smoothing": 0.0,
        "pressure_variation": False,
        "jitter_intensity": 0.6,
        "jitter_frequency": 0.45,
        "steps_per_stroke": 400,
        "hand_lift_prob": 0.0,
        "hand_lift_max_segments": 0,
        "ink_color": (0, 0, 0),
        "opacity": 1.0,
        "splat_prob": 0.0,
        "extra_small_strokes_prob": 0.0,
        "variable_sampling": True,
        "seed": None
    }

    if style:
        cfg.update(style)

    # print(cfg)

    if cfg.get("seed") is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    scale = max(1, int(cfg.get("antialias_scale", 1)))
    W, H = width * scale, height * scale
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)

    num_strokes = random.randint(int(cfg["num_strokes_range"][0]), int(cfg["num_strokes_range"][1]))

    for stroke_idx in range(num_strokes):
        points = _random_control_points(W, H, complexity=1.0 + random.random() * 0.6)
        pts = _bezier_points(points, steps=int(cfg["steps_per_stroke"]), variable_sampling=bool(cfg["variable_sampling"]))

        # hand-lift segmentation
        segments = [(0, len(pts))]
        if random.random() < float(cfg.get("hand_lift_prob", 0.0)):
            max_seg = max(1, int(cfg.get("hand_lift_max_segments", 2)))
            n_seg = random.randint(1, max_seg)
            indices = sorted(random.sample(range(1, len(pts) - 1), n_seg - 1))
            segs = []
            prev = 0
            for idx in indices:
                segs.append((prev, idx))
                prev = idx
            segs.append((prev, len(pts)))
            segments = segs

        base_thickness_min, base_thickness_max = _sample_int_range(cfg.get("stroke_thickness_range", [1, 3]))
        pressure_variation = bool(cfg.get("pressure_variation", False))
        jitter_intensity = float(cfg.get("jitter_intensity", 0.0)) * scale
        jitter_freq = float(cfg.get("jitter_frequency", 0.0))

        for (s_start, s_end) in segments:
            seg_pts = pts[s_start:s_end]
            if seg_pts.shape[0] < 2:
                continue

            # jitter: добавить шум с вероятностью
            if jitter_intensity > 0:
                mask = np.random.rand(seg_pts.shape[0]) < jitter_freq
                noise = np.zeros_like(seg_pts)
                noise[mask] = np.random.normal(scale=jitter_intensity, size=(mask.sum(), 2))
                seg_pts = seg_pts + noise

            # draw points as overlapping circles
            for i, (px, py) in enumerate(seg_pts.astype(int)):
                t = i / max(1, (len(seg_pts) - 1))
                if pressure_variation:
                    pressure = 0.7 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * (t + random.random() * 0.2)))
                else:
                    pressure = 1.0
                base_thickness = random.randint(base_thickness_min, base_thickness_max)
                r = max(1, int(base_thickness * (0.6 + 0.8 * pressure)))
                draw.ellipse([px - r, py - r, px + r, py + r], fill=0)

            # extra small strokes
            if random.random() < float(cfg.get("extra_small_strokes_prob", 0.0)):
                extra_steps = random.randint(20, 70)
                extra_points = _bezier_points(_random_control_points(W, H, complexity=0.5), steps=extra_steps, variable_sampling=False)
                for (px, py) in extra_points.astype(int):
                    rr = max(1, int((base_thickness_min + base_thickness_max) / 4.0))
                    draw.ellipse([px - rr, py - rr, px + rr, py + rr], fill=0)

        # splat
        if random.random() < float(cfg.get("splat_prob", 0.0)):
            mid_idx = len(pts) // 2 + random.randint(-8, 8)
            mx, my = pts[mid_idx].astype(int)
            smin, smax = cfg.get("splat_size_range", [4, 10])
            _add_ink_splat(draw, mx, my, random.randint(int(smin * scale), int(smax * scale)))

    # blur + downsample
    blur_radius = float(cfg.get("smoothing", 0.0)) * scale
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    img_small = img.resize((width, height), resample=Image.LANCZOS) if scale > 1 else img
    signature_gray = np.array(img_small)
    mask = (255 - signature_gray).clip(0, 255).astype(np.uint8)

    # color + alpha
    # ink_rgb = tuple(int(c) for c in cfg.get("ink_color", (0, 0, 0)))
    ink_color = cfg.get("ink_color", (0, 0, 0))

    # Если передано "random" — генерируем случайный цвет
    if ink_color == "random":
        ink_rgb = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        )
    else:
        # обычная обработка
        ink_rgb = tuple(int(c) for c in ink_color)

    opacity = float(cfg.get("opacity", 1.0))
    h, w = mask.shape
    b = np.full((h, w), ink_rgb[2], dtype=np.uint8)
    g = np.full((h, w), ink_rgb[1], dtype=np.uint8)
    r = np.full((h, w), ink_rgb[0], dtype=np.uint8)
    alpha = (mask.astype(np.float32) * opacity).clip(0, 255).astype(np.uint8)
    signature_bgra = np.stack([b, g, r, alpha], axis=-1)

    return signature_bgra, mask


# -----------------------------
# DigitalSignatureGenerator: новый API (без кэша)
# -----------------------------
class DigitalSignatureGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        config keys:
          - signature_width, signature_height
          - style_ranges: словарь диапазонов
          - scale_range: [min,max] — масштаб подписи при вставке
        """
        self.config: Dict[str, Any] = config.copy() if config else {}
        self.style_ranges: Dict[str, Any] = self.config.get("style_ranges", {})
        # опционально можно фиксировать seed для воспроизводимости
        self.global_seed = self.config.get("seed", None)

    def _sample_style_once(self, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Берёт style_ranges (self.style_ranges), применяет override (диапазоны переопределяются),
        и возвращает словарь style с конкретными значениями (не диапазонами).
        """
        if override:
            # merge shallow: override keys replace ranges
            merged = {**self.style_ranges, **override}
        else:
            merged = dict(self.style_ranges)

        # если глобальный seed задан — каждый вызов можно сдвигать seed чтобы не было одинаковых
        if self.global_seed is not None:
            # меняем локальный RNG состояние (не глобально) для reproducibility per call
            random.seed(self.global_seed + random.randint(0, 10_000_000))
            np.random.seed(self.global_seed + random.randint(0, 10_000_000))

        style: Dict[str, Any] = {}

        # num_strokes_range (если задан диапазон, сэмплим int в этом диапазоне и так же сохраняем диапазер для генератора)
        nsr = merged.get("num_strokes_range", [2, 4])
        if isinstance(nsr, (list, tuple)) and len(nsr) == 2:
            style["num_strokes_range"] = [int(nsr[0]), int(nsr[1])]
        else:
            style["num_strokes_range"] = nsr

        # stroke_thickness_range передаём как есть (пользователь указывает два числа)
        style["stroke_thickness_range"] = merged.get("stroke_thickness_range", [1, 3])

        # pressure_variation: либо probability поле, либо фикс
        pprob = merged.get("pressure_variation_prob", None)
        if pprob is None:
            style["pressure_variation"] = bool(merged.get("pressure_variation", False))
        else:
            # если probability — сэмплим
            if isinstance(pprob, (list, tuple)):
                prob = float(random.uniform(float(pprob[0]), float(pprob[1])))
            else:
                prob = float(pprob)
            style["pressure_variation"] = random.random() < prob

        # antialias_scale (int)
        aas = merged.get("antialias_scale", [1, 2])
        style["antialias_scale"] = int(_sample_scalar(aas))

        # smoothing
        style["smoothing"] = float(_sample_scalar(merged.get("smoothing", [0.0, 0.5])))

        # jitter intensity & freq
        style["jitter_intensity"] = float(_sample_scalar(merged.get("jitter_intensity", [0.0, 1.0])))
        style["jitter_frequency"] = float(_sample_scalar(merged.get("jitter_frequency", [0.0, 1.0])))

        # steps_per_stroke
        style["steps_per_stroke"] = int(_sample_scalar(merged.get("steps_per_stroke", [120, 220])))

        # hand_lift
        style["hand_lift_prob"] = float(_sample_scalar(merged.get("hand_lift_prob", [0.0, 0.1])))
        style["hand_lift_max_segments"] = int(_sample_scalar(merged.get("hand_lift_max_segments", [1, 2])))

        # ink color: support fixed ink_color or ink_color_range
        if "ink_color" in merged:
            style["ink_color"] = merged["ink_color"]
        else:
            style["ink_color"] = _sample_color(merged.get("ink_color_range", None), default=(0, 0, 0))

        # opacity
        style["opacity"] = float(_sample_scalar(merged.get("opacity", [0.8, 1.0])))

        # splat
        style["splat_prob"] = float(_sample_scalar(merged.get("splat_prob", [0.0, 0.05])))
        # splat_size_range can be either [min,max] or [[min,max],[min,max]]; normalize to [min,max]
        ssr = merged.get("splat_size_range", [4, 10])
        if isinstance(ssr, (list, tuple)) and len(ssr) == 2 and isinstance(ssr[0], (int, float)):
            style["splat_size_range"] = [int(ssr[0]), int(ssr[1])]
        elif isinstance(ssr, (list, tuple)) and len(ssr) == 2 and isinstance(ssr[0], (list, tuple)):
            # take first pair's min and second pair's max as fallback
            style["splat_size_range"] = [int(ssr[0][0]), int(ssr[1][1])]
        else:
            style["splat_size_range"] = [4, 10]

        style["extra_small_strokes_prob"] = float(_sample_scalar(merged.get("extra_small_strokes_prob", [0.0, 0.05])))

        # variable_sampling as bool sampled from probability
        vsp = merged.get("variable_sampling_prob", 1.0)
        if isinstance(vsp, (list, tuple)) and len(vsp) == 2:
            prob = float(random.uniform(float(vsp[0]), float(vsp[1])))
        else:
            prob = float(vsp)
        style["variable_sampling"] = random.random() < prob

        # seed optional (if provided as fixed number)
        style["seed"] = merged.get("seed", None)

        return style

    # API: генерировать подпись и сразу применить
    def generate_signature(self, width: Optional[int] = None, height: Optional[int] = None,
                           style_override: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        w = int(width or self.config.get("signature_width", 400))
        h = int(height or self.config.get("signature_height", 120))
        style = self._sample_style_once(style_override)
        # преобразуем некоторые поля в формат, ожидаемый генератором
        # например: num_strokes_range уже готов, stroke_thickness_range уже готов, ink_color, opacity и т.д.
        return generate_realistic_signature(w, h, style)

    def apply_signature_to_image(self, image: np.ndarray,
                                 position: Optional[Tuple[int, int]] = None,
                                 style_ranges_override: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сгенерировать уникальную подпись (параметры сэмплируются из style_ranges или style_ranges_override)
        и наложить на BGR изображение. Возвращает (result_image, result_mask).
        """
        img_h, img_w = image.shape[:2]
        # sample style values and generate signature
        style_override = style_ranges_override or None

        sig_bgra, sig_mask = self.generate_signature(width=self.config.get("signature_width", 400),
                                                     height=self.config.get("signature_height", 120),
                                                     style_override=style_override['style_ranges'])
        sig_h, sig_w = sig_mask.shape

        # масштаб подписи
        scale_range = self.config.get("scale_range", [0.8, 1.1])
        scale = float(random.uniform(float(scale_range[0]), float(scale_range[1])))
        new_w = max(8, int(sig_w * scale))
        new_h = max(6, int(sig_h * scale))

        sig_bgra = cv2.resize(sig_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sig_mask = cv2.resize(sig_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        sig_h, sig_w = sig_mask.shape

        # позиционирование
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

        sig_b = sig_bgra[:, :, 0].astype(float)
        sig_g = sig_bgra[:, :, 1].astype(float)
        sig_r = sig_bgra[:, :, 2].astype(float)
        sig_a = sig_bgra[:, :, 3].astype(float) / 255.0

        roi = result[y:y + sig_h, x:x + sig_w].astype(float)
        alpha_3c = np.stack([sig_a, sig_a, sig_a], axis=-1)
        src_rgb = np.stack([sig_b, sig_g, sig_r], axis=-1)

        blended = roi * (1 - alpha_3c) + src_rgb * alpha_3c
        result[y:y + sig_h, x:x + sig_w] = blended.clip(0, 255).astype(np.uint8)

        mask_out = np.zeros((img_h, img_w), dtype=np.uint8)
        binary_sig = (sig_mask > 127).astype(np.uint8) * 255
        mask_out[y:y + sig_h, x:x + sig_w] = binary_sig

        return result, mask_out

    def save_signature_png(self, path: str, width: Optional[int] = None, height: Optional[int] = None,
                           style_ranges_override: Optional[Dict[str, Any]] = None) -> None:
        sig_bgra, _ = self.generate_signature(width=width, height=height, style_override=style_ranges_override)
        rgba = np.dstack([sig_bgra[:, :, 2], sig_bgra[:, :, 1], sig_bgra[:, :, 0], sig_bgra[:, :, 3]])
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(path, format="PNG")


# -----------------------------
# USAGE EXAMPLES
# -----------------------------
# Пример конфига:
# cfg = {
#     "signature_width": 400,
#     "signature_height": 120,
#     "scale_range": [0.75, 1.05],
#     "style_ranges": {
#         "num_strokes_range": [2,3],
#         "stroke_thickness_range": [1,2],
#         "pressure_variation_prob": 0.0,
#         "antialias_scale": [1,1],
#         "smoothing": [0.0, 0.18],
#         "jitter_intensity": [0.3, 0.8],
#         "jitter_frequency": [0.35, 0.55],
#         "steps_per_stroke": [160, 200],
#         "hand_lift_prob": [0.05, 0.14],
#         "hand_lift_max_segments": [1,2],
#         "ink_color_range": [[60,100],[60,100],[60,100]],  # серый карандаш
#         "opacity": [0.78, 0.92],
#         "splat_prob": [0.0, 0.01],
#         "extra_small_strokes_prob": [0.0, 0.05],
#         "variable_sampling_prob": [0.8, 1.0]
#     }
# }
#
# gen = DigitalSignatureGenerator(cfg)
# img = cv2.imread("photo.jpg")
# res, mask = gen.apply_signature_to_image(img)  # каждый вызов — уникальная подпись
# gen.save_signature_png("example_sig.png")
