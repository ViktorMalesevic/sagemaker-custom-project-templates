import sys
import numpy as np
from PIL import Image
import sys
import json
import os
import time
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper


__dir__ = os.path.dirname(os.path.realpath(__file__))

# variables
with open(f'{__dir__}/classes', 'r') as f:
    rawcat = f.read()
    class_dict = {idx_ : cat for idx_, cat in enumerate(rawcat.split('\n'))}



# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, COLORS):
    label = str(class_dict[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def postprocess(boxes, scores, indices):
    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        out_classes.append(idx_[1])
        out_scores.append(scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(boxes[idx_1])
    
    return out_boxes, out_scores, out_classes

def get_class_img(img_in, img_out, out_boxes, out_scores, out_classes):
    COLORS = np.random.uniform(0, 255, size=(len(class_dict.keys()), 3))

    image_data = cv2.imread(img_in)
    for box, score, cl in zip(out_boxes, out_scores, out_classes):
        x = box[1]
        y = box[0]
        w = box[3]
        h = box[2]
        
        draw_bounding_box(image_data, cl, score, round(x), round(y), round(w), round(h), COLORS)

    cv2.imwrite(img_out, image_data)
    
    return True


