#!/usr/bin/env python3
import glob, random, os, pickle, shutil, sys
import keras.backend as K
import numpy as np
import xmltodict
from datetime import datetime
from keras.models import model_from_json
from PIL import Image, ImageDraw

SCALE       = 1.0
GRID_WIDTH  = int(16 * SCALE)
GRID_HEIGHT = int(16 * SCALE)
WIDTH       = int(1920 * SCALE)
HEIGHT      = int(1080 * SCALE)
GRID_X      = WIDTH // GRID_WIDTH
GRID_Y      = HEIGHT // GRID_HEIGHT
GRID_FACTOR = GRID_WIDTH*10 # The object bounding box is a percentage of GRID FACTOR.
CHANNEL     = 1

# TEXTS = [i for i in '2378WBG']
TEXTS = [i for i in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ']

def calculate_render_parameters(img, result, thresh=0.5):
    texts  = []
    bounds = []
    # Calculate texts, bounds, and colors (based on confident)
    img = np.reshape(img, [HEIGHT, WIDTH])
    for row in range(GRID_Y):
        for col in range(GRID_X):
            if result[row, col, 0] >= thresh:
                x, y, w, h = box_to_bound(col, row, *result[row, col, 1:5])
                text = TEXTS[np.argmax(result[row, col, -len(TEXTS):])]
                # Collect
                texts.append(text)
                bounds.append((x,y,w,h))
    return Image.fromarray(np.uint8(img*255)), texts, bounds

def draw_texts_bounds(background, texts, bounds):
    background = background.copy()
    draw = ImageDraw.Draw(background)
    for atext, (x,y,w,h) in zip(texts, bounds):
        draw.rectangle([x,y,x+w,y+h], outline=255)
        draw.text((x,y+h), atext, fill=255)
    return background

def bound_to_box(x, y, w, h):
    cx = (x + w/2) // GRID_WIDTH # CELL x
    cy = (y + h/2) // GRID_HEIGHT # CELL y
    bx = (x + w/2) % GRID_WIDTH / GRID_WIDTH # CENTER of box relative box
    by = (y + h/2) % GRID_HEIGHT / GRID_HEIGHT # CENTER of box relative box
    bw = w / GRID_FACTOR # WIDTH of box relative to image
    bh = h / GRID_FACTOR # HEIGHT of box relative to image
    return cx, cy, bx, by, bw, bh

def box_to_bound(cx, cy, bx, by, bw, bh):
    w = bw * GRID_FACTOR
    h = bh * GRID_FACTOR
    x = (cx * GRID_WIDTH) + (bx * GRID_WIDTH) - w/2
    y = (cy * GRID_HEIGHT) + (by * GRID_HEIGHT) - h/2
    return x, y, w, h

def main():
    folder = 'models/{}'.format(sys.argv[1]) if len(sys.argv) > 1 else sorted(glob.glob('models/*'))[-1]

    model = model_from_json(open('{}/model.json'.format(folder)).read())
    model.load_weights('{}/model_weights.h5'.format(folder))

    # --- Load datas.
    with open('datas-test.pickle', 'rb') as f:
        datas = pickle.load(f)

    # --- Prepare data.
    x_tests = []
    for img in datas:
        height, width = img.shape
        # Image.
        x_data = img / 255
        x_data = np.reshape(x_data, [HEIGHT, WIDTH, CHANNEL])
        # Collect
        x_tests.append(x_data)

    # --- Test.
    results = model.predict(np.asarray(x_tests))

    print('------------------', results)

    # ofolder = '{}/tests'.format(folder)
    # os.makedirs(ofolder, exist_ok=True)
    # for f in glob.glob('{}/*'.format(ofolder)): # Delete all test files
    #     os.remove(f)
    #
    # for i, (x, r) in enumerate(zip(x_tests, results)):
    #     image, texts, bounds = calculate_render_parameters(x, r)
    #     image = draw_texts_bounds(image, texts, bounds)
    #     image.save('{}/image_{:02d}.png'.format(ofolder, i))

if __name__ == '__main__':
    main()
