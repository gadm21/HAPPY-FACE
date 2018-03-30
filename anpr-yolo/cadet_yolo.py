#!/usr/bin/env python3
import glob, json, random, sys
import keras, tensorflow as tf
import keras.backend as K
import numpy as np
from datetime import datetime
from keras.metrics import binary_accuracy, categorical_accuracy, sparse_categorical_accuracy
from keras.models import Sequential, Model
from keras.layers import Activation, Conv2D, Dense, Dropout, AveragePooling2D, MaxPooling2D, LeakyReLU, Input
from keras.layers.normalization import BatchNormalization
from cadet_image import *

TEXTS = [i for i in '027']

HEIGHT  = 128
WIDTH   = 128
CHANNEL = 1

GRID        = 8
GRID_WIDTH  = WIDTH / GRID
GRID_HEIGHT = HEIGHT / GRID

def bound_to_box(x, y, w, h):
    cx = (x + w/2) // GRID_WIDTH # CELL x
    cy = (y + h/2) // GRID_HEIGHT # CELL y
    bx = (x + w/2) % GRID_WIDTH / GRID_WIDTH # CENTER of box relative box
    by = (y + h/2) % GRID_HEIGHT / GRID_HEIGHT # CENTER of box relative box
    bw = w / WIDTH # WIDTH of box relative to image
    bh = h / HEIGHT # HEIGHT of box relative to image
    return cx, cy, bx, by, bw, bh

def box_to_bound(cx, cy, bx, by, bw, bh):
    w = bw * WIDTH
    h = bh * HEIGHT
    x = (cx * GRID_WIDTH) + (bx * GRID_WIDTH) - w/2
    y = (cy * GRID_HEIGHT) + (by * GRID_HEIGHT) - h/2
    return x, y, w, h

def calculate_render_parameters(img, result, thresh=0.5):
    texts  = []
    bounds = []
    # Calculate texts, bounds, and colors (based on confident)
    img = np.reshape(img, [HEIGHT, WIDTH])
    for row in range(GRID):
        for col in range(GRID):
            if result[row, col, 0] >= thresh:
                x, y, w, h = box_to_bound(col, row, *result[row, col, 1:5])
                text = TEXTS[np.argmax(result[row, col, -len(TEXTS):])]
                # Collect
                texts.append(text)
                bounds.append((x,y,w,h))
    return Image.fromarray(np.uint8(img*255)), texts, bounds

def loss(fact, pred):
    fact = K.reshape(fact, [-1, GRID*GRID, 5+len(TEXTS)])
    pred = K.reshape(pred, [-1, GRID*GRID, 5+len(TEXTS)])

    # Truth
    fact_conf = fact[:,:,0]
    fact_x    = fact[:,:,1]
    fact_y    = fact[:,:,2]
    fact_w    = fact[:,:,3]
    fact_h    = fact[:,:,4]
    fact_cat  = fact[:,:,5:]

    # Prediction
    pred_conf = pred[:,:,0]
    pred_x    = pred[:,:,1]
    pred_y    = pred[:,:,2]
    pred_w    = pred[:,:,3]
    pred_h    = pred[:,:,4]
    pred_cat  = pred[:,:,5:]

    # Mask
    mask_obj = fact_conf
    mask_noobj = 1 - mask_obj

    # --- Confident loss
    conf_loss = K.square(fact_conf - pred_conf)
    conf_loss = (mask_obj * conf_loss) + (mask_noobj * conf_loss)
    print('conf_loss.shape: ', conf_loss.shape)

    # --- Box loss
    xy_loss  = K.square(fact_x - pred_x) + K.square(fact_y - pred_y)
    wh_loss  = K.square(K.sqrt(fact_w) - K.sqrt(pred_w)) + K.square(K.sqrt(fact_h) - K.sqrt(pred_h))
    box_loss = mask_obj * (xy_loss + wh_loss)
    print('box_loss.shape: ', box_loss.shape)

    # --- Category loss
    cat_loss = mask_obj * K.sum(K.square(fact_cat - pred_cat), axis=-1)
    print('cat_loss.shape: ', cat_loss.shape)

    # --- Total loss
    return K.sum(conf_loss + box_loss + cat_loss, axis=-1)

def P_(fact, pred):
    fact = K.reshape(fact, [-1, GRID*GRID, 5+len(TEXTS)])
    pred = K.reshape(pred, [-1, GRID*GRID, 5+len(TEXTS)])
    # Truth
    fact_conf = fact[:,:,0]
    # Prediction
    pred_conf = pred[:,:,0]
    # PROBABILITY
    return binary_accuracy(fact_conf, pred_conf)

def XY_(fact, pred):
    fact = K.reshape(fact, [-1, GRID*GRID, 5+len(TEXTS)])
    pred = K.reshape(pred, [-1, GRID*GRID, 5+len(TEXTS)])
    # Truth
    fact_conf = fact[:,:,0]
    fw = fact[:,:,3] * WIDTH
    fh = fact[:,:,4] * HEIGHT
    fx = fact[:,:,0] * GRID_WIDTH - fw/2
    fy = fact[:,:,1] * GRID_HEIGHT - fh/2
    # Prediction
    pw = pred[:,:,3] * WIDTH
    ph = pred[:,:,4] * HEIGHT
    px = pred[:,:,0] * GRID_WIDTH - pw/2
    py = pred[:,:,1] * GRID_HEIGHT - ph/2
    # IOU
    intersect = (K.minimum(fx+fw, px+pw) - K.maximum(fx, px)) * (K.minimum(fy+fh, py+ph) - K.maximum(fy, py))
    union = (fw * fh) + (pw * ph) - intersect
    return K.sum((intersect / union) * fact_conf) / tf.count_nonzero(fact_conf, dtype=tf.float32)

def C_(fact, pred):
    fact = K.reshape(fact, [-1, GRID*GRID, 5+len(TEXTS)])
    pred = K.reshape(pred, [-1, GRID*GRID, 5+len(TEXTS)])
    # Truth
    fact_conf = fact[:,:,0]
    fact_cat = fact[:,:,5:]
    # Prediction
    pred_cat = pred[:,:,5:]
    # CLASSIFICATION
    return K.sum(categorical_accuracy(fact_cat, pred_cat) * fact_conf) / tf.count_nonzero(fact_conf, dtype=tf.float32)

def generator(batch):
    while True:
        x_boxes = []
        y_boxes = []
        for i in range(batch): # Batch
            image, texts, bounds = generate_image(size=(WIDTH, HEIGHT), texts=TEXTS*20, text_width=(7,9),
                                                  color=(32,128), fill=(168,255), rotation=(-5,5))
            # Image.
            x = np.array(image) / 255
            x = np.reshape(x, [HEIGHT, WIDTH, CHANNEL])
            # Truth.
            y = np.zeros((GRID, GRID, 5+len(TEXTS)))
            for atext, abound in zip(texts, bounds):
                cx, cy, bx, by, bw, bh = bound_to_box(*abound)
                row, col = int(cy), int(cx) # Swap to row and col.
                y[row, col, 0] = 1
                y[row, col, 1:5] = [bx, by, bw, bh]
                y[row, col, 5+TEXTS.index(atext)] = 1
            x_boxes.append(x)
            y_boxes.append(y)
        yield (np.asarray(x_boxes), np.asarray(y_boxes))

def get_model():
    input_layer = Input(shape=(HEIGHT, WIDTH, CHANNEL))

    x = Conv2D(8, 3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x) #64

    x = Conv2D(16, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x) #32

    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x) #16

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x) #8

    x = Conv2D(5+len(TEXTS), 1, padding='same')(x) # 1 x confident, 4 x coord, 5 x class
    x = Activation('sigmoid')(x)

    model = Model(input_layer, x)
    model.compile(optimizer='adam', loss=loss, metrics=[P_, XY_, C_])
    return model

def main():
    model = get_model()
    print(model.summary())
    print('')

    # Dev data from camera.
    x_devs = []
    y_devs = []
    for fp in sorted(glob.glob('cameras-outputs/*.json')):
        image = None
        texts = []
        bounds = []
        for data in json.load(open(fp)):
            t = data['name']
            x = float(data['xmin'])/2
            y = float(data['ymin'])/2
            w = float(data['xmax'])/2 - x
            h = float(data['ymax'])/2 - y
            if t in TEXTS:
                texts.append(t)
                bounds.append((x,y,w,h))
        if texts and bounds:
            imgfp = (fp.split('.')[0]).split('/')[-1]
            image = Image.open('cameras-outputs/{}.jpg'.format(imgfp))
            image = image.resize((WIDTH, HEIGHT))
        if image:
            # Image.
            x = np.array(image) / 255
            x = np.reshape(x, [HEIGHT, WIDTH, CHANNEL])
            # Truth.
            y = np.zeros((GRID, GRID, 5+len(TEXTS)))
            for atext, abound in zip(texts, bounds):
                cx, cy, bx, by, bw, bh = bound_to_box(*abound)
                row, col = int(cy), int(cx) # Swap to row and col.
                y[row, col, 0] = 1
                y[row, col, 1:5] = [bx, by, bw, bh]
                y[row, col, 5+TEXTS.index(atext)] = 1
            # Collect
            x_devs.append(x)
            y_devs.append(y)
    x_devs = np.asarray(x_devs)
    y_devs = np.asarray(y_devs)

    # Train.
    model.fit_generator(generator=generator(batch=8),
                        validation_data=(x_devs, y_devs),
                        steps_per_epoch=10, epochs=50)
    now = datetime.now()
    # Save
    model.save('models/aero_{:%Y%m%d-%H%M}.h5'.format(now))
    # Test
    results = model.predict(x_devs)
    for i, (x, r) in enumerate(zip(x_devs, results)):
        image, texts, bounds = calculate_render_parameters(x, r)
        image = draw_texts_bounds(image, texts, bounds)
        image.save('outputs/cadet_{:%Y%m%d-%H%M}_{:02d}.png'.format(now, i))

if __name__ == '__main__':
    main()
