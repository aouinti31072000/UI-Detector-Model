import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov3 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import *

TRAIN_MODEL_NAME += "_Tiny" if TRAIN_YOLO_TINY  else ""
LINES = open(TEST_ANNOT_PATH).readlines()
yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights

for image_info in LINES:
        
    image_path = image_info.split()[0]
    
    detect_image(yolo, image_path, "wireframe_test.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))


