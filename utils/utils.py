import cv2 
from PIL import Image
import numpy as np 
import io 
import time 
import math

def draw_boxes(img, boxes, labels, probs, class_names):
    img_copy = img.copy()
    h, w = img.shape[0], img.shape[1]
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        x1 = max(math.ceil(box[0]), 0)
        y1 = max(math.ceil(box[1]), 0)
        x2 = min(math.floor(box[2]), w)
        y2 = min(math.floor(box[3]), h)
        img_copy = cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 255, 0), 1)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        img_copy = cv2.putText(img_copy, label,
                    (int(box[0]) + 10, int(box[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    (127, 0, 127),
                    2) # line type

    return img_copy

def cut_cothes(img, boxes, labels, probs, class_names):
    h, w = img.shape[0], img.shape[1]
    list_img = []

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        x1 = max(math.ceil(box[0]), 0)
        y1 = max(math.ceil(box[1]), 0)
        x2 = min(math.floor(box[2]), w)
        y2 = min(math.floor(box[3]), h)
        cut_img = img[y1:y2, x1:x2]
        list_img.append({'label': class_names[labels[i]], 'img': cut_img, 'probs':probs[i]})

    return list_img
