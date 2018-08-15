#!/usr/bin/env python3

import re
import json
import keras.backend as K
import numpy as np
import time
import os
import uuid
from datetime import datetime
from functools import cmp_to_key
from keras.models import model_from_json
from PIL import Image, ImageDraw
import cv2

import server_settings # vehicle recognition server's shared variables between model and worker

class ModelParams:
    CROP_XMIN    = 200
    CROP_YMIN    = 300
    CROP_XMAX    = 700
    CROP_YMAX    = 1024
    CROP         = (CROP_XMIN, CROP_YMIN, CROP_XMAX, CROP_YMAX)
    SCALE        = 0.5
    GRID_WIDTH   = int(16 * SCALE)
    GRID_HEIGHT  = int(16 * SCALE)
    TRAIN_WIDTH  = int(3072 * SCALE)
    TRAIN_HEIGHT = int(2048 * SCALE)
    WIDTH        = int(CROP_XMAX - CROP_XMIN)
    HEIGHT       = int(CROP_YMAX - CROP_YMIN)
    GRID_X       = int(WIDTH // GRID_WIDTH)
    GRID_Y       = int(HEIGHT // GRID_HEIGHT)
    GRID_FACTOR  = GRID_WIDTH*10 # The object bounding box is a percentage of GRID FACTOR.
    CHANNEL      = 1
    TEXTS        = [i for i in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    PATTERN      = re.compile(r'^[A-Z]{1,3}\d{1,4}[A-Z]*$')

    RTSP_FEED = "" # RTSP Feed link goes here

    # File paths.
    MODEL  = 'model.json'
    WEIGHT = 'model_weights.h5'

    # Majority voting algorithm parameters.
    MEMORY    = 15 # 15
    THRESHOLD = 5 # 5
    
    # All validation methods return if a config is valid/not
    @staticmethod
    def valid_crop(xmin, ymin, xmax, ymax):
        xmin_ok = xmin >= 0
        ymin_ok = ymin >= 0
        xmax_ok = xmax <= 10000 and xmax >= 0
        ymax_ok = ymax <= 10000 and ymax >= 0
        is_proper_crop = (xmax-xmin > 0) and (ymax-ymin > 0)
        return (is_proper_crop and xmin_ok and ymin_ok and xmax_ok and xmin_ok)

    # All update methods return if an update was successful/not
    @staticmethod
    def update_scale(scale):
        if scale > 0:
            ModelParams.SCALE = scale
            ModelParams.GRID_WIDTH   = int(16 * scale)
            ModelParams.GRID_HEIGHT  = int(16 * scale)
            ModelParams.TRAIN_WIDTH  = int(3072 * scale)
            ModelParams.TRAIN_HEIGHT = int(2048 * scale)
            return True
        return False

    @staticmethod
    def update_memory(memory):
        if memory > 0: 
            ModelParams.MEMORY = memory
            return True
        return False

    @staticmethod
    def update_threshold(threshold):
        if threshold > 0:
            ModelParams.THRESHOLD = threshold
            return True
        return False

    @staticmethod
    def update_crop(xmin, ymin, xmax, ymax):
        if ModelParams.valid_crop(xmin, ymin, xmax, ymax):
            # Updates the crop variables as well as their dependencies
            ModelParams.CROP_XMIN    = xmin
            ModelParams.CROP_YMIN    = ymin
            ModelParams.CROP_XMAX    = xmax
            ModelParams.CROP_YMAX    = ymax
            ModelParams.CROP         = (xmin, ymin, xmax, ymax)
            ModelParams.WIDTH        = int(xmax - xmin)
            ModelParams.HEIGHT       = int(ymax - ymin)
            ModelParams.GRID_X       = int(ModelParams.WIDTH // ModelParams.GRID_WIDTH)
            ModelParams.GRID_Y       = int(ModelParams.HEIGHT // ModelParams.GRID_HEIGHT)
            return True
        return False

    # Loads configuration from config.json
    # for now it only loads configuration for crop region and memory, threshold for voting algorithm and rtsp feed link
    @staticmethod
    def load_configuration():
        with open('config.json') as f:
            config = json.load(f)
        try:
            xmin   = config['CROP_XMIN']
            ymin   = config['CROP_YMIN']
            xmax   = config['CROP_XMAX']
            ymax   = config['CROP_YMAX']
            ModelParams.update_crop(xmin, ymin, xmax, ymax)
            ModelParams.MEMORY      = config['MEMORY']
            ModelParams.THRESHOLD   = config['THRESHOLD']
            ModelParams.RTSP_FEED   = config['RTSP_FEED']
            return True
        except Exception as e:
            print('config file corrupted - loading defaults')
            print(e)
            
        return False


def inference():
    load_config_status = ModelParams.load_configuration()
    if load_config_status == False:
        print("Could not load configuration - Using defaults")
        
    model = model_from_json(open(ModelParams.MODEL).read())
    model.load_weights(ModelParams.WEIGHT)

    # Keep track for majority voting algorithm.
    current_plate = None
    current_count = 0
    
    cap = cv2.VideoCapture(ModelParams.RTSP_FEED)
    idx = 0
    while True:
        ret, frame = cap.read()
        idx += 1
        if ret == False:
            continue
        
        # Prepare image for inferencing.
        image = Image.fromarray(frame)
        image = image.convert(mode='L')
        image = image.resize((int(ModelParams.TRAIN_WIDTH), int(ModelParams.TRAIN_HEIGHT)))
        image = image.crop(box=ModelParams.CROP)
        img = np.array(image)
        img = img / 255
        img = np.reshape(img, [-1, ModelParams.HEIGHT, ModelParams.WIDTH, ModelParams.CHANNEL])

        # Keep start time for a single frame.
        start = datetime.now()

        # Inference.
        result = model.predict(img)[0]
        texts, bounds = calculate_render_parameters(result, thresh=0.5)
        # Cluster found texts to become a carplate.
        clusters = []
        for text, (x,y,w,h) in zip(texts, bounds):
            r1 = Rect(text, x, y, w, h)
            matched = False
            for r2 in clusters:
                # Overlap
                if r1.overlap(r2, padx=8, pady=4):
                    matched = True
                    r2.union(r1)
            if not matched:
                clusters.append(r1)

        clusters = sorted(clusters, key=lambda c: c.area, reverse=True)
        best = clusters[0] if len(clusters) > 0 else None
        potential = best.plate if best else '-'
            
        # Majority voting algorithm.
        if current_count == 0:
            current_text  = potential
            current_count = 1
        else:
            if current_text == potential:
                current_count = current_count + 1
            else:
                current_count = current_count - 1

        current_count = min(current_count, ModelParams.MEMORY) # Cap memory.

        send = ''
        found = ''
        if ModelParams.PATTERN.match(current_text): # Valid car plate pattern.
            if current_count > ModelParams.THRESHOLD: # Threshold.
                found = '*'
                if current_plate != current_text:
                    current_plate = current_text
                    send = 'SEND'
                    save_frame(frame, best)
        else:
            current_text = '-'

        # Print inference duration with result.
        # FORMAT:
        #   count duration | inferenced | votes potential_plate ('found' = exceed threshold) (send = send to server)
        #
        print('{:3d} Duration: {} | {:10} | {:2d}  {:10} {} {}'.format(idx, datetime.now()-start, potential, current_count, current_text, found, send))

def save_frame(img, carplate):
    ts = time.time()
    st = datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')
    x1 = carplate.minx + ModelParams.CROP_XMIN
    y1 = carplate.miny + ModelParams.CROP_YMIN
    x2 = x1 + carplate.w
    y2 = y1 + carplate.h
    bbox = ":".join([str(max(int(i), 0)) for i in [x1, y1, x2, y2]])
    mac = ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff) for i in range(0,8*6,8)][::-1])
    imgname = '{}_{}_{}_{}.png'.format(mac, st, carplate.plate, bbox)
    fullpath = os.path.join(server_settings.SAVE_DIR, imgname)
    cv2.imwrite(fullpath, img)

class Rect(object):
    """
    Represent a text with it's bounding box.
    """
    def __init__(self, text, x, y, w, h):
        self.text = text
        self.minx = x
        self.maxx = x+w
        self.miny = y
        self.maxy = y+h
        self.w = w
        self.h = h
        self.area = self.w * self.h
        self.rects = [self]
        self.plate = text

    def union(self, rect):
        self.minx = min(self.minx, rect.minx)
        self.maxx = max(self.maxx, rect.maxx)
        self.miny = min(self.miny, rect.miny)
        self.maxy = max(self.maxy, rect.maxy)
        self.w = self.maxx - self.minx
        self.h = self.maxy - self.miny
        self.area = self.w * self.h
        # Arrange texts into plate.
        self.rects.append(rect)
        self.rects = sorted(self.rects, key=cmp_to_key(Rect.arrange))
        self.plate = ''.join([r.text for r in self.rects])

    def valid(self):
        pattern = re.compile(r'^[A-Z]+\d{1,4}[A-Z]*$')
        return pattern.match(self.plate)

    def arrange(r1, r2):
        if r1.minx+r1.h/2 < r2.minx:
            return True
        if r1.minx < r2.minx:
            return True
        return False

    def overlap(self, rect, padx=0, pady=0):
        if (self.minx-padx > rect.maxx+padx) or (self.maxx+padx < rect.minx-padx):
            return False
        if (self.miny-pady > rect.maxy+pady) or (self.maxy+pady < rect.miny-pady):
            return False
        return True

def calculate_render_parameters(result, thresh=0.5):
    """
    Calculate the x,y,w,h from model inferenced representation.
    """
    texts  = []
    bounds = []
    # Calculate texts, bounds (based on confident)
    for row in range(ModelParams.GRID_Y):
        for col in range(ModelParams.GRID_X):
            if result[row, col, 0] >= thresh:
                x, y, w, h = box_to_bound(col, row, *result[row, col, 1:5])
                text = ModelParams.TEXTS[np.argmax(result[row, col, -len(ModelParams.TEXTS):])]
                # Collect
                texts.append(text)
                bounds.append((x,y,w,h))
    return texts, bounds

def box_to_bound(cx, cy, bx, by, bw, bh):
    w = bw * ModelParams.GRID_FACTOR
    h = bh * ModelParams.GRID_FACTOR
    x = (cx * ModelParams.GRID_WIDTH) + (bx * ModelParams.GRID_WIDTH) - w/2
    y = (cy * ModelParams.GRID_HEIGHT) + (by * ModelParams.GRID_HEIGHT) - h/2
    return x, y, w, h

if __name__ == '__main__':
    inference()
