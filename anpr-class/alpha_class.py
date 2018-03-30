#!/usr/bin/env python3
import glob, math, random
import keras, tensorflow as tf
import keras.backend as K
import numpy as np
from keras.callbacks import Callback, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from PIL import Image, ImageDraw, ImageFont
from skimage import io, color, transform, util

print('Keras:', keras.__version__)
print('Tensorflow:', tf.__version__)
print('')

LETTERS = ['0','5','7']
WIDTH   = 48
HEIGHT  = 48
CHANNEL = 1

FONT = ImageFont.truetype("Arial Bold.ttf", 48)

def get_val_images():
    images, labels, names = [], [], []
    for letter in LETTERS:
        for fp in glob.glob('val_images/{}-*.png'.format(letter)):
            img = io.imread(fp, as_grey=True)
            images.append(img)
            labels.append(letter)
            names.append(fp)
    return images, labels, names

def get_model():
    model = Sequential()
    model.add(Conv2D(4, 3, padding='same', activation='relu', input_shape=(HEIGHT, WIDTH, CHANNEL)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(8, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(len(LETTERS), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

def generate_image(text):
    # --- Text
    prefix, postfix = str(random.randint(10,99))
    text = '{}{}{}'.format(prefix, text, postfix)
    # --- Image.
    image = Image.new(mode='L', color=random.randint(0,32), size=(WIDTH,HEIGHT))
    draw  = ImageDraw.Draw(image)
    w, h = draw.textsize(text, font=FONT)
    draw.text(((WIDTH-w)/2, -2), text, fill=random.randint(128,255), font=FONT)
    # --- Augmentation.
    img              = np.array(image)
    noise            = random.uniform(0.001, 0.01)
    rotation         = np.deg2rad(random.uniform(-5,5))
    x, y             = random.uniform(-5,5), random.uniform(-5,5)
    shift_y, shift_x = WIDTH/2, HEIGHT/2

    tx_rotate    = transform.SimilarityTransform(rotation=rotation)
    tx_shift     = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    tx_shift_inv = transform.SimilarityTransform(translation=[shift_x + x, shift_y + y])
    tx           = tx_shift + (tx_rotate + tx_shift_inv)
    img          = transform.warp(img, tx.inverse, order=3, output_shape=(HEIGHT,WIDTH))
    img          = util.random_noise(img, var=noise)
    return img

def generator():
    batch_size = 16
    while True:
        x_trains = []
        y_trains = []
        for l in (LETTERS * batch_size):
            img = generate_image(l)
            x_trains.append(generate_image(l))
            y_trains.append(LETTERS.index(l))

        x_trains = np.reshape(x_trains, [-1, HEIGHT, WIDTH, CHANNEL])
        y_trains = keras.utils.to_categorical(y_trains, num_classes=len(LETTERS))
        yield x_trains, y_trains

def main():
    model = get_model()
    print(model.summary())
    print('')

    # --- Validation data
    images, labels, _ = get_val_images()
    # Images
    val_datas = np.asarray(images)
    val_datas = np.reshape(val_datas, [-1, HEIGHT, WIDTH, CHANNEL])
    # Labels
    labels = [LETTERS.index(l) for l in labels]
    val_labels = keras.utils.to_categorical(labels, num_classes=len(LETTERS))

    # --- Training
    model.fit_generator(generator=generator(),
                        steps_per_epoch=10,
                        epochs=30,
                        validation_data=(val_datas, val_labels),
                        callbacks=[BestStopping(val_acc=0.99)])

    # --- Prediction
    images, labels, names = get_val_images()
    for aimage, alabel, aname in zip(images, labels, names):
        result = model.predict(np.reshape(aimage, [-1, HEIGHT, WIDTH, CHANNEL]))
        index = np.argmax(result)
        correct = '' if alabel == LETTERS[index] else '[WRONG]'
        print('   {} --> {}   {}'.format(aname, LETTERS[index], correct))

class BestStopping(Callback):
    def __init__(self, val_acc):
        super(Callback, self).__init__()
        self.val_acc = val_acc
    def on_epoch_end(self, epoch, logs={}):
        if logs['val_acc'] >= self.val_acc:
            self.model.stop_training = True

if __name__ == '__main__':
    main()
    # # Output generated images.
    # for i in range(50):
    #     image = generate_image(random.choice(LETTERS))
    #     io.imsave('samples/{:03d}.png'.format(i), image)
