#!/usr/bin/env python3
import json, glob, math, os, pickle, random, sys
import numpy as np
import xmltodict
from collections import Counter
from PIL import Image, ImageDraw, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

TEXTS = [i for i in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ']

SCALE   = 1.0
WIDTH   = int(320 * SCALE)
HEIGHT  = int(128 * SCALE)
BLANK   = 1
FOLDERS = ['datas{:03d}'.format(i) for i in range(21)]

datas   = []
letters = []

skip_not_founds     = []
skip_no_annotations = []
skip_single_objects = []
skip_no_letters     = []
skip_big_letters    = []

for afolder in FOLDERS:
    for idx, fp in enumerate(sorted(glob.glob('datas/{}/labels/*.xml'.format(afolder)))):
        print(idx, fp, '...')
        # Load image.
        fname = os.path.split(fp)[-1]
        fname = os.path.splitext(fname)[0]
        ip = 'datas/{}/images/{}.jpg'.format(afolder, fname)
        if not os.path.exists(ip):
            skip_not_founds.append(fp)
            print(ip, '... NOT FOUND')
            continue
        image = Image.open(ip)
        image = image.convert(mode='L')
        image = image.resize((int(image.size[0]*SCALE), int(image.size[1]*SCALE)), resample=Image.LANCZOS)
        # Load XML.
        with open(fp) as f:
            annotation = xmltodict.parse(f.read())['annotations']
            texts = []
            # --- Get coordinates.
            if 'object' not in annotation:
                skip_no_annotations.append(fp + ' = ' + ip)
                print(fp, '... NO ANNOTATION')
                continue
            if type(annotation['object']) != list:
                skip_single_objects.append(fp)
                print(fp, '... SINGLE OBJECT')
                continue
            for obj in annotation['object']:
                if len(obj['name']) == 1 and obj['name'] in TEXTS:
                    # Keep text and coordinates.
                    name = obj['name'].upper()
                    x = int(int(obj['bndbox']['xmin']) * SCALE)
                    y = int(int(obj['bndbox']['ymin']) * SCALE)
                    w = int(int(obj['bndbox']['xmax']) * SCALE) - x
                    h = int(int(obj['bndbox']['ymax']) * SCALE) - y
                    texts.append((name,x,y,w,h))
            # Clean up
            top = 0
            votes = [0 for _ in texts]
            for idx, (a,x,y,_,_) in enumerate(texts):
                for b,p,q,_,_ in texts:
                    if abs(x-p) < 300 and abs(y-q) < 200:
                        votes[idx] = votes[idx] + 1
                        top = votes[idx]
            texts = [texts[idx] for idx, v in enumerate(votes) if v == top]
            # Calculte biggest bounding box.
            l,t,r,b = sys.maxsize, sys.maxsize, 0, 0
            for _,x,y,w,h in texts:
                l,t,r,b = min(l,x), min(t,y), max(r,x+w), max(b,y+h)
            # No carplate found or too big.
            if not texts:
                skip_no_letters.append(fp)
                print(fp, '... NO CAR LETTERS')
                continue
            if not texts or (r-l) >= WIDTH or (b-t) >= HEIGHT:
                skip_big_letters.append(fp)
                print(fp, '... CAR LETTERS TOO BIG')
                continue
            # --- Crop.
            rx = random.randint(l - (WIDTH-(r-l)), l)
            ry = random.randint(t - (HEIGHT-(b-t)), t)
            rtexts = [(name, x-rx, y-ry, w, h) for name,x,y,w,h in texts]
            cropped = image.crop(box=(rx, ry, rx+WIDTH, ry+HEIGHT))
            # --- Collect datas.
            datas.append((fname, rtexts, cropped))
            letters.extend([name for name, _, _, _, _ in rtexts])
        # Find a few emtpy crop images.
        for i in range(BLANK):
            hoverlaps = True
            voverlaps = True
            while(hoverlaps or voverlaps):
                x = random.randint(0, image.size[0]-WIDTH)
                y = random.randint(0, image.size[1]-HEIGHT)
                hoverlaps = False if (x > rx+WIDTH) or (x+WIDTH < rx) else True
                voverlaps = False if (y > ry+HEIGHT) or (y+HEIGHT < ry) else True
            cropped = image.crop(box=(x, y, x+WIDTH, y+HEIGHT))
            datas.append((fname, [], cropped))

# --- DEBUG: Check data.
os.makedirs('temps', exist_ok=True)
for idx, (fname, rtexts, image) in enumerate(random.sample(datas, min(50,len(datas)))):
    background = image.copy()
    draw = ImageDraw.Draw(background)
    for name, x, y, w, h in rtexts:
        draw.rectangle([x,y,x+w,y+h], outline=255)
        draw.text((x,y+h), name, fill=255)
    background.save('temps/{}_{}.jpg'.format(fname, idx % (BLANK+1)))

# Pickle datas.
datas = [(rtexts, np.array(image)) for fname, rtexts, image in datas]
with open('datas.pickle', 'wb') as f:
    pickle.dump(datas, f, pickle.HIGHEST_PROTOCOL)

# Count.
print('')
print('')
print('TOTAL        : {}'.format(len(datas)))
print('')
print('No found     : {}'.format(len(skip_not_founds)))
print('No annotation: {}'.format(len(skip_no_annotations)))
print('Single object: {}'.format(len(skip_single_objects)))
print('No letter    : {}'.format(len(skip_no_letters)))
print('Big letter   : {}'.format(len(skip_big_letters)))
print('')
c = Counter(letters)
for t, n in c.most_common():
    print(t, n)
print('')

logs = {
    'NOT FOUND': skip_not_founds,
    'NO ANNOTATION': skip_no_annotations,
    'SINGLE OBJECT': skip_single_objects,
    'NO LETTERS': skip_no_letters,
    'BIG LETTERS': skip_big_letters,
}
with open('datas.log', 'w') as f:
    json.dump(logs, f, indent=4)
