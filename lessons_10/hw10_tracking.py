#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework 10 — Object Tracking with OpenCV

Compare two different trackers (KCF and CSRT) on a video.

Steps:
  1. Select a video and an object to track
  2. Initialize tracker #1 (KCF)
  3. Run it for ~15 frames, save bounding-box images
  4. Initialize tracker #2 (CSRT), repeat
  5. Compare results
"""

import os
import cv2
import numpy as np

# ========================== Step 1 ==========================
# Decide what video to use and select an object.
# Я использую видео с дорожным трафиком из материалов лекции.
# Буду отслеживать одну из машин на дороге.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_FILE = os.path.join(SCRIPT_DIR, "slow_traffic_small.mp4")
NUM_FRAMES = 15  # кол-во кадров для трекинга

if not os.path.exists(VIDEO_FILE):
    raise FileNotFoundError(f"Видео не найдено: {VIDEO_FILE}")

# Папки для сохранения результатов
DIR_KCF = os.path.join(SCRIPT_DIR, "results_kcf")
DIR_CSRT = os.path.join(SCRIPT_DIR, "results_csrt")
os.makedirs(DIR_KCF, exist_ok=True)
os.makedirs(DIR_CSRT, exist_ok=True)


def select_roi(video_path):
    """Читаем первый кадр и задаём ROI (область интереса) вручную."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Не удалось открыть видео!")
    return frame


def run_tracker(video_path, tracker, bbox, num_frames, output_dir, tracker_name):
    """
    Запускаем трекер на видео.
    Для каждого кадра рисуем bounding box и сохраняем изображение.
    Возвращаем список bounding box'ов.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Не удалось открыть видео!")

    # Инициализируем трекер первым кадром и начальным bbox
    tracker.init(frame, bbox)
    boxes = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Видео закончилось на кадре {i}")
            break

        # Обновляем трекер
        ok, new_bbox = tracker.update(frame)

        if ok:
            # Рисуем прямоугольник вокруг объекта
            x, y, w, h = [int(v) for v in new_bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{tracker_name} frame {i+1}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            boxes.append(new_bbox)
        else:
            # Трекер потерял объект
            cv2.putText(frame, f"{tracker_name}: tracking lost!",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            boxes.append(None)

        # Сохраняем кадр
        filename = os.path.join(output_dir, f"frame_{i+1:03d}.jpg")
        cv2.imwrite(filename, frame)

    cap.release()
    return boxes


# Читаем первый кадр, чтобы выбрать объект
first_frame = select_roi(VIDEO_FILE)
h_frame, w_frame = first_frame.shape[:2]

# Задаю bounding box вручную — выбираю белую машину в правой части кадра
# (x, y, width, height)
bbox = (470, 200, 80, 60)

# Рисуем начальный bbox на первом кадре для наглядности
preview = first_frame.copy()
x0, y0, w0, h0 = bbox
cv2.rectangle(preview, (x0, y0), (x0 + w0, y0 + h0), (255, 0, 0), 2)
cv2.putText(preview, "Selected object", (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.imwrite(os.path.join(DIR_KCF, "frame_000_initial.jpg"), preview)
cv2.imwrite(os.path.join(DIR_CSRT, "frame_000_initial.jpg"), preview)
print(f"Initial bounding box: {bbox}")
print(f"Frame size: {w_frame}x{h_frame}")


# ========================== Step 2 & 3 ==========================
# Initialize and run KCF tracker

print("\n--- Running KCF tracker ---")
tracker_kcf = cv2.TrackerKCF_create()
boxes_kcf = run_tracker(VIDEO_FILE, tracker_kcf, bbox, NUM_FRAMES,
                        DIR_KCF, "KCF")

for i, b in enumerate(boxes_kcf):
    if b is not None:
        print(f"  Frame {i+1}: bbox = ({int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])})")
    else:
        print(f"  Frame {i+1}: LOST")


# ========================== Step 4 & 5 ==========================
# Initialize and run CSRT tracker

print("\n--- Running CSRT tracker ---")
tracker_csrt = cv2.TrackerCSRT_create()
boxes_csrt = run_tracker(VIDEO_FILE, tracker_csrt, bbox, NUM_FRAMES,
                         DIR_CSRT, "CSRT")

for i, b in enumerate(boxes_csrt):
    if b is not None:
        print(f"  Frame {i+1}: bbox = ({int(b[0])}, {int(b[1])}, {int(b[2])}, {int(b[3])})")
    else:
        print(f"  Frame {i+1}: LOST")


# ========================== Step 6 ==========================
# Compare the results
#
# Считаю разницу между bbox'ами двух трекеров на каждом кадре

print("\n--- Comparison ---")
print(f"{'Frame':<8} {'KCF bbox':<30} {'CSRT bbox':<30} {'Diff (x,y)':<15}")
print("-" * 85)

for i in range(min(len(boxes_kcf), len(boxes_csrt))):
    bk = boxes_kcf[i]
    bc = boxes_csrt[i]

    if bk is not None and bc is not None:
        dx = int(bc[0]) - int(bk[0])
        dy = int(bc[1]) - int(bk[1])
        kcf_str = f"({int(bk[0])}, {int(bk[1])}, {int(bk[2])}, {int(bk[3])})"
        csrt_str = f"({int(bc[0])}, {int(bc[1])}, {int(bc[2])}, {int(bc[3])})"
        print(f"{i+1:<8} {kcf_str:<30} {csrt_str:<30} ({dx}, {dy})")
    else:
        status_k = "LOST" if bk is None else "OK"
        status_c = "LOST" if bc is None else "OK"
        print(f"{i+1:<8} {status_k:<30} {status_c:<30}")


# =================== Ответы на вопросы ====================
"""
Step 6 — Compare the results:

* Do you see any differences? If so, what are they?

  Да, разница есть. KCF работает быстрее, но bbox иногда немного
  «съезжает» с объекта. CSRT точнее удерживает объект в центре
  рамки, bbox плотнее прилегает к контуру машины.

* Does one tracker perform better than the other? In what way?

  CSRT в целом точнее — он лучше справляется с изменением масштаба
  и небольшими поворотами объекта. KCF быстрее по скорости, но
  менее устойчив: на сложных сценах может потерять объект раньше.
  Для данного видео оба трекера справляются, но CSRT даёт более
  стабильный результат.
"""

print("\nDone! Check 'results_kcf/' and 'results_csrt/' folders for saved frames.")
