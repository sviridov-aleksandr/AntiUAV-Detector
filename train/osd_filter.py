#!/usr/bin/env python3
"""
Умный OSD-фильтр: отсекает ложные детекции в краях кадра (HUD/телеметрия),
но пропускает реальные объекты у краёв.

Логика:
1. Большой bbox (> 5% площади кадра) — пропускаем (близкий дрон)
2. Центр bbox в безопасной зоне — пропускаем (дрон касается края, но центр в кадре)
3. Маленький bbox полностью у края — отсекаем (похоже на OSD-текст)
"""

import numpy as np


def is_osd_false_positive(x1, y1, x2, y2, img_w, img_h,
                           margin=60, large_bbox_ratio=0.05):
    """
    Возвращает True если детекция скорее всего OSD (ложная),
    False если это реальный объект.

    Args:
        x1, y1, x2, y2: координаты bbox
        img_w, img_h: размеры кадра
        margin: отступ от краёв (px) для OSD-зоны
        large_bbox_ratio: доля площади кадра, выше которой bbox считается крупным
    """
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_area = bbox_w * bbox_h
    frame_area = img_w * img_h

    # 1. Крупный объект — пропускаем (это не OSD)
    if bbox_area / frame_area > large_bbox_ratio:
        return False

    # 2. Центр bbox в безопасной зоне — пропускаем
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if margin < cx < img_w - margin and margin < cy < img_h - margin:
        return False

    # 3. Маленький объект у края — отсекаем
    return True
