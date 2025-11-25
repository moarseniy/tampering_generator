import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
from typing import List, Union, Optional, Dict, Any
from fontTools.ttLib import TTFont

def font_supports_alphabet(font_path: str, alphabet: str) -> bool:
    """
    Проверяет, поддерживает ли TTF-файл все символы из alphabet.
    """
    try:
        font = TTFont(font_path, fontNumber=0)
    except Exception:
        return False  # битые шрифты пропускаем

    # собрать mapping unicode → glyph
    unicodes = set()

    for table in font["cmap"].tables:
        if table.isUnicode():
            unicodes.update(table.cmap.keys())

    # проверить каждый символ алфавита
    for ch in alphabet:
        if ord(ch) not in unicodes:
            return False

    return True

def random_text(alphabet: str, text_len=5) -> str:
    return "".join(random.choice(alphabet) for _ in range(text_len))

def _sample_scalar(val, dtype=int, default=None):
    if val is None:
        return default
    if isinstance(val, (list, tuple)) and len(val) == 2:
        a, b = val
        if dtype is int:
            return int(random.randint(int(a), int(b)))
        else:
            return float(random.uniform(float(a), float(b)))
    if dtype is int:
        return int(val)
    return float(val)

def _parse_hex_color(s: str):
    s2 = s.lstrip("#")
    if len(s2) == 3:
        s2 = "".join([c * 2 for c in s2])
    if len(s2) != 6:
        raise ValueError("Hex color must be 6 hex digits")
    return (int(s2[0:2], 16), int(s2[2:4], 16), int(s2[4:6], 16))

def _sample_color(spec, default=(0, 0, 0)):
    # spec may be tuple, hex string, "random", per-channel ranges, or gray range
    if spec is None:
        return default
    # fixed tuple
    if isinstance(spec, (list, tuple)) and len(spec) == 3 and all(isinstance(c, (int, float)) for c in spec):
        return tuple(int(c) for c in spec)
    # hex
    if isinstance(spec, str):
        if spec.lower() == "random":
            # случайный тёмный/насыщенный цвет (лучше читается на белом)
            pick = random.random()
            if pick < 0.7:
                return (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
            elif pick < 0.9:
                return (random.randint(0, 40), random.randint(0, 60), random.randint(30, 120))
            else:
                return (random.randint(30, 110), random.randint(20, 80), random.randint(0, 40))
        # try hex
        try:
            return _parse_hex_color(spec)
        except Exception:
            return default
    # per-channel ranges: [[rmin,rmax],[gmin,gmax],[bmin,bmax]]
    if isinstance(spec, (list, tuple)) and len(spec) == 3 and isinstance(spec[0], (list, tuple)):
        r = int(random.randint(int(spec[0][0]), int(spec[0][1])))
        g = int(random.randint(int(spec[1][0]), int(spec[1][1])))
        b = int(random.randint(int(spec[2][0]), int(spec[2][1])))
        return (r, g, b)
    # gray range [min,max]
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and isinstance(spec[0], (int, float)):
        v = int(random.randint(int(spec[0]), int(spec[1])))
        return (v, v, v)
    return default

def render_text_into_bbox(image: np.ndarray,
                          text: str,
                          bbox: Union[tuple, list, dict],
                          font_path: Optional[str] = None,
                          style: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Рендер текста в область bbox на изображении (OpenCV BGR → PIL → OpenCV BGR).
    Поддерживает style-опции (каждый может быть фиксированным значением или диапазоном [min,max]):
      - font_size или font_size_range: int или [min,max]
      - font_color или font_color_range:
            * кортеж (r,g,b)
            * hex "#RRGGBB"
            * "random" (случайный тёмный цвет)
            * [[rmin,rmax],[gmin,gmax],[bmin,bmax]] (диапазон по каналам)
            * [min,max] -> градация серого
      - opacity: float 0..1 или [min,max]
      - angle: угол поворота текста в градусах (фикс или [min,max])
      - align: "center"/"left"/"right" (по горизонтали)
      - valign: "middle"/"top"/"bottom" (по вертикали)
      - stroke_width: int (опционально)
      - stroke_fill: color spec (опционально)
    Возвращает BGR numpy.ndarray.
    """

    # --- распарсить bbox ---
    # допускаем форматы: [x, y, w, h] или (x1,y1,x2,y2) или dict {"x1":...,"y1":...,"x2":...,"y2":...}
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            bbox_w, bbox_h = x2 - x1, y2 - y1
            x, y = x1, y1
        else:
            # fallback: if provided as {"x":,"y":,"w":,"h":}
            x = int(bbox.get("x", bbox.get("left", 0)))
            y = int(bbox.get("y", bbox.get("top", 0)))
            bbox_w = int(bbox.get("w", bbox.get("width", 0)))
            bbox_h = int(bbox.get("h", bbox.get("height", 0)))
    else:
        # list/tuple
        x, y, bbox_w, bbox_h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # --- конфиг стиля (default) ---
    style = style or {}
    font_size_spec = style.get("font_size", style.get("font_size_range", [20, 36]))
    color_spec = style.get("font_color", style.get("font_color_range", None))
    opacity_spec = style.get("opacity", [1.0, 1.0])
    angle_spec = style.get("angle", 0)
    align = style.get("align", "center")       # center / left / right
    valign = style.get("valign", "middle")     # middle / top / bottom
    stroke_width = int(style.get("stroke_width", 0))
    stroke_fill_spec = style.get("stroke_fill", None)

    # sample values
    font_size = _sample_scalar(font_size_spec, dtype=int, default=font_size_spec if isinstance(font_size_spec, int) else 20)
    font_color = _sample_color(color_spec, default=(0, 0, 0))
    stroke_fill = _sample_color(stroke_fill_spec, default=None) if stroke_fill_spec is not None else None
    opacity = _sample_scalar(opacity_spec, dtype=float, default=1.0)
    angle = _sample_scalar(angle_spec, dtype=float, default=angle_spec if isinstance(angle_spec, (int, float)) else 0.0)

    # ensure valid ranges
    opacity = max(0.0, min(1.0, float(opacity)))
    font_size = max(8, int(font_size))

    # --- подготовка PIL изображений ---
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # --- загрузка шрифта ---
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        # fallback to default font
        font = ImageFont.load_default()

    # --- подобрать размер шрифта если нужно уменьшить чтобы влезло ---
    text_w, text_h = draw.textsize(text, font=font)
    # уменьшать только если слишком большой
    if (text_w > bbox_w or text_h > bbox_h) and font_path:
        # попробуем уменьшать до минимума
        fs = font_size
        while fs > 8:
            fnt = ImageFont.truetype(font_path, fs)
            tw, th = draw.textsize(text, font=fnt)
            if tw <= bbox_w and th <= bbox_h:
                font = fnt
                text_w, text_h = tw, th
                break
            fs -= 1
        # если не получилось вписать, оставляем последнюю попытку (будет частично за пределами)

    # --- позиционирование текста внутри bbox ---
    if align == "center":
        pos_x = x + (bbox_w - text_w) // 2
    elif align == "left":
        pos_x = x
    elif align == "right":
        pos_x = x + bbox_w - text_w
    else:
        pos_x = x + (bbox_w - text_w) // 2

    if valign == "middle":
        pos_y = y + (bbox_h - text_h) // 2
    elif valign == "top":
        pos_y = y
    elif valign == "bottom":
        pos_y = y + bbox_h - text_h
    else:
        pos_y = y + (bbox_h - text_h) // 2

    # --- рисуем текст на overlay с альфой ---
    rgba_fill = (int(font_color[0]), int(font_color[1]), int(font_color[2]), int(opacity * 255))
    if stroke_width > 0 and stroke_fill is not None:
        # Pillow >=8.0 поддерживает stroke_width/stroke_fill in text()
        draw.text((pos_x, pos_y), text, font=font, fill=rgba_fill,
                  stroke_width=stroke_width, stroke_fill=tuple(int(c) for c in stroke_fill) + (int(opacity * 255),))
    else:
        draw.text((pos_x, pos_y), text, font=font, fill=rgba_fill)

    # --- поворот overlay вокруг центра текста (если angle != 0) ---
    if abs(angle) > 0.001:
        # rotate overlay but keep full canvas size (expand=False), we'll composite later
        overlay = overlay.rotate(float(angle), resample=Image.BICUBIC, center=(pos_x + text_w/2, pos_y + text_h/2))

    # --- объединение слоя через альфу ---
    composed = Image.alpha_composite(pil_img, overlay).convert("RGB")

    # вернуть в OpenCV BGR
    return cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)
