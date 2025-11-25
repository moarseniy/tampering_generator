import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
from typing import List

def random_text(alphabet: str, text_len=5) -> str:
    return "".join(random.choice(alphabet) for _ in range(text_len))

def render_text_into_bbox(image: np.ndarray, text: str, bbox: dict, font_path: str, font_size: int = 30):
    """
    Рендер текста в область bbox на изображении (OpenCV BGR → PIL → OpenCV BGR).
    """
    x, y, bbox_w, bbox_h = bbox[0], bbox[1], bbox[2], bbox[3]

    # конвертируем в PIL
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # грузим шрифт
    font = ImageFont.truetype(font_path, font_size)

    # уменьшаем размер шрифта, чтобы влезло
    while True:
        text_w, text_h = draw.textsize(text, font=font)
        if text_w <= bbox_w and text_h <= bbox_h:
            break
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        if font_size < 10:
            break

    # рендерим по центру прямоугольника
    pos_x = x + (bbox_w - text_w) // 2
    pos_y = y + (bbox_h - text_h) // 2

    draw.text((pos_x, pos_y), text, fill="black", font=font)

    # обратно в OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
