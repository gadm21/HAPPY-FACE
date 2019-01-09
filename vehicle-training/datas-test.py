#!/usr/bin/env python3
import json, glob, math, os, pickle, random, sys
import numpy as np
import xmltodict
from collections import Counter
from PIL import Image, ImageDraw, ImageFile

SCALE  = 1.0
WIDTH  = int(1920 * SCALE)
HEIGHT = int(1080 * SCALE)

ImageFile.LOAD_TRUNCATED_IMAGES = True

datas   = []

for fp in sorted(glob.glob('datastests/*')):
    print(fp, '...')
    # Load image.
    image = Image.open(fp)
    image = image.convert(mode='L')
    image = image.resize((WIDTH, HEIGHT), resample=Image.LANCZOS)
    datas.append(np.array(image))

# Pickle datas.
with open('datas-test.pickle', 'wb') as f:
    pickle.dump(datas, f, pickle.HIGHEST_PROTOCOL)

# Count.
print('')
print('')
print('TOTAL        : {}'.format(len(datas)))
print('')
