#!/usr/bin/env python3
import json, os, pickle, random, sys
import keras, tensorflow as tf
import keras.backend as K
import numpy as np
from datetime import datetime
from keras.callbacks import ModelCheckpoint, Callback
from keras.metrics import binary_accuracy, categorical_accuracy
from keras.models import Model
from keras.layers import Activation, Conv2D, MaxPooling2D, LeakyReLU, Input
from keras.layers.normalization import BatchNormalization
from PIL import Image, ImageDraw


SCALE       = 1.0
GRID_WIDTH  = int(16 * SCALE)
GRID_HEIGHT = int(16 * SCALE)
WIDTH       = int(320 * SCALE)
HEIGHT      = int(128 * SCALE)
GRID_X      = WIDTH // GRID_WIDTH
GRID_Y      = HEIGHT // GRID_HEIGHT
GRID_FACTOR = GRID_WIDTH*10 # The object bounding box is a percentage of GRID_FACTOR.
CHANNEL     = 1

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

def loss(fact, pred):
    fact = K.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(TEXTS)])
    pred = K.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(TEXTS)])

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
    fact = K.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(TEXTS)])
    pred = K.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(TEXTS)])
    # Truth
    fact_conf = fact[:,:,0]
    # Prediction
    pred_conf = pred[:,:,0]
    # PROBABILITY
    return binary_accuracy(fact_conf, pred_conf)

def XY_(fact, pred):
    fact = K.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(TEXTS)])
    pred = K.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(TEXTS)])
    # Truth
    fact_conf = fact[:,:,0]
    fw = fact[:,:,3] * GRID_FACTOR
    fh = fact[:,:,4] * GRID_FACTOR
    fx = fact[:,:,0] * GRID_WIDTH - fw/2
    fy = fact[:,:,1] * GRID_HEIGHT - fh/2
    # Prediction
    pw = pred[:,:,3] * GRID_FACTOR
    ph = pred[:,:,4] * GRID_FACTOR
    px = pred[:,:,0] * GRID_WIDTH - pw/2
    py = pred[:,:,1] * GRID_HEIGHT - ph/2
    # IOU
    intersect = (K.minimum(fx+fw, px+pw) - K.maximum(fx, px)) * (K.minimum(fy+fh, py+ph) - K.maximum(fy, py))
    union = (fw * fh) + (pw * ph) - intersect
    nonzero_count = tf.count_nonzero(fact_conf, dtype=tf.float32)
    return K.switch(
        tf.equal(nonzero_count, 0),
        1.0,
        K.sum((intersect / union) * fact_conf) / nonzero_count
    )

def C_(fact, pred):
    fact = K.reshape(fact, [-1, GRID_Y*GRID_X, 5+len(TEXTS)])
    pred = K.reshape(pred, [-1, GRID_Y*GRID_X, 5+len(TEXTS)])
    # Truth
    fact_conf = fact[:,:,0]
    fact_cat = fact[:,:,5:]
    # Prediction
    pred_cat = pred[:,:,5:]
    # CLASSIFICATION
    nonzero_count = tf.count_nonzero(fact_conf, dtype=tf.float32)
    return K.switch(
        tf.equal(nonzero_count, 0),
        1.0,
        K.sum(categorical_accuracy(fact_cat, pred_cat) * fact_conf) / nonzero_count
    )

class HistoryCheckpoint(keras.callbacks.Callback):
    def __init__(self, folder):
        self.folder = folder
    def on_train_begin(self, logs={}):
        with open('{}/model.json'.format(self.folder), 'w') as f:
            json.dump(json.loads(self.model.to_json()), f)
        with open('{}/history.txt'.format(self.folder), 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    def on_epoch_end(self, epoch, logs={}):
        keys = ['loss', 'P_', 'XY_', 'C_']
        h = ' - '. join(['{}: {:.4f}'.format(k, logs[k]) for k in keys])
        h = h + ' // ' + ' - '. join(['val_{}: {:.4f}'.format(k, logs['val_'+k]) for k in keys])
        h = '{:03d} : '.format(epoch) + h
        with open('{}/history.txt'.format(self.folder), 'a') as f:
            f.write(h + '\n')

def get_model():
    input_layer = Input(shape=(None, None, CHANNEL))

    SEED = 32
    x = Conv2D(SEED, 3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    for i in range(1):
        x = Conv2D(SEED // 2, 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(SEED , 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

    x = MaxPooling2D()(x) # ONE

    SEED = 64
    x = Conv2D(SEED, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    for i in range(2): # ----- SQUEEZE
        x = Conv2D(SEED // 2, 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(SEED , 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

    x = MaxPooling2D()(x) # TWO

    SEED = 128
    x = Conv2D(SEED, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    for i in range(3): # ----- SQUEEZE
        x = Conv2D(SEED // 2, 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(SEED , 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

    x = MaxPooling2D()(x)

    SEED = 256
    x = Conv2D(SEED, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    for i in range(4): # ----- SQUEEZE
        x = Conv2D(SEED // 2, 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Conv2D(SEED , 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

    x = MaxPooling2D()(x)

    x = Conv2D(5+len(TEXTS), 1, padding='same')(x) # 1 x confident, 4 x coord, 5 x len(TEXTS)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(input_layer, x)
    model.compile(optimizer='adam', loss=loss, metrics=[P_, XY_, C_])
    return model

def main():
    model = get_model()
    print(model.summary())
    print('')

    # --- Load datas.
    with open('datas.pickle', 'rb') as f:
        datas = pickle.load(f)
        random.shuffle(datas)

    # --- Prepare data.
    x_trains = []
    y_trains = []
    for texts, img in datas:
        # Image.
        x_data = img / 255
        x_data = np.reshape(x_data, [HEIGHT, WIDTH, CHANNEL])
        # Truth.
        y_data = np.zeros((GRID_Y, GRID_X, 5+len(TEXTS)))
        for name, x,y,w,h in texts:
            cx, cy, bx, by, bw, bh = bound_to_box(x,y,w,h)
            # print(name, ':', x, y, w, h, '--->', cx, cy, bx, by, bw, bh)
            row, col = int(cy), int(cx) # Swap to row and col.
            y_data[row, col, 0] = 1
            y_data[row, col, 1:5] = [bx, by, bw, bh]
            y_data[row, col, 5+TEXTS.index(name)] = 1
        # Collect
        x_trains.append(x_data)
        y_trains.append(y_data)

    # --- Split test vs validation.
    datas = list(zip(x_trains, y_trains))
    random.shuffle(datas)
    x_trains, y_trains = zip(*datas)

    VALIDATION = 50
    # Convert to np array.
    x_trains   = np.asarray(x_trains[:-VALIDATION])
    y_trains   = np.asarray(y_trains[:-VALIDATION])
    x_vals     = np.asarray(x_trains[-VALIDATION:])
    y_vals     = np.asarray(y_trains[-VALIDATION:])

    # --- Setup.

    now = datetime.now()
    folder = 'models/{:%Y%m%d-%H%M}'.format(now)
    os.makedirs(folder)
    # Callbacks
    history_checkpoint = HistoryCheckpoint(folder=folder)
    model_checkpoint = ModelCheckpoint('{}/model_weights.h5'.format(folder), save_weights_only=True)

    # --- Train.

    print(x_trains.shape)
    print(y_trains.shape)

    model.fit(x=x_trains, y=y_trains,
              batch_size=16, epochs=100, # ------------------------------ EPOCHS
              validation_data=(x_vals, y_vals),
              shuffle=True, callbacks=[model_checkpoint, history_checkpoint])

    # Render validation output.
    results = model.predict(x_vals)

    ofolder = '{}/outputs'.format(folder)
    os.makedirs(ofolder)
    for i, (x, r) in enumerate(zip(x_vals, results)):
        image, texts, bounds = calculate_render_parameters(x, r)
        image = draw_texts_bounds(image, texts, bounds)
        image.save('{}/image_{:02d}.png'.format(ofolder, i))

if __name__ == '__main__':
    main()
