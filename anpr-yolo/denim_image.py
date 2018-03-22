#!/usr/bin/env python3
import json, glob, random, sys
import numpy as np
import xmltodict
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from skimage import io, draw, color, transform, util

HEIGHT     = 256
WIDTH      = 256
TEXTS      = ['0','1','2','3','4','5','6','7','8','9']

# ----- CAMERA

image_negatives = []
for fp in glob.glob('cameras-negatives/*.png'):
    image = Image.open(fp).convert('L')
    image_negatives.append(image)

# ----- SYNTHETIC

def create_image(size, color):
    image = random.choice(image_negatives)
    image = image.resize(size)
    return image

def generate_text_images(texts, text_width, fill, rotation, fontname='Arial Bold.ttf'):
    images = []
    for atext in texts:
        # -- Get text image size.
        font = ImageFont.truetype(fontname, 100)
        x_offset, y_offset = font.getoffset(atext)
        w, h = font.getsize(atext)
        w, h = w-x_offset, h-y_offset
        # -- Create text image.
        aimage = Image.new(mode='RGBA', size=(w,h), color=0)
        adraw  = ImageDraw.Draw(aimage)
        adraw.text((-x_offset,-y_offset), atext, font=font, fill=tuple([fill]*3))
        # -- Rotation
        aimage = aimage.rotate(rotation, resample=Image.BICUBIC, expand=True)
        # -- Resize
        tw = random.randint(*text_width)
        aimage = aimage.resize((tw, round(h/w*tw)), resample=Image.BICUBIC)
        # -- Collect
        images.append(aimage)
    return images

def paste_text_images(background, images):
    width, height = background.size
    bounds = []
    for aimage in images:
        w, h = aimage.size
        ok = False # Ensure no intersect.
        while not ok:
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            # Ensure within bounds and no intersect.
            ok = x+w < width and y+h < height
            for bx,by,bw,bh in bounds:
                dx = min(x+w, bx+bw) - max(x, bx)
                dy = min(y+h, by+bh) - max(y, by)
                ok = ok and (dx <= 0 or dy <= 0)
        background.paste(aimage, (x, y), mask=aimage)
        # Collect IMAGE coordinates and dimensions.
        ew, eh = int(w * 0.1), int(h * 0.1) # Expand bounds
        bounds.append((x-ew, y-eh, w+ew*2, h+eh*2))
    return bounds

def generate_image(size, texts, text_width, color=(0,0), fill=(255,255), rotation=(0,0)):
    # Parameters.
    color    = random.randint(*color)
    fill     = random.randint(*fill)
    rotation = random.uniform(*rotation)
    # Background image.
    background = create_image(size=size, color=color)
    # Text images.
    images = generate_text_images(texts, text_width=text_width, fill=fill, rotation=rotation)
    # Paste text images.
    bounds = paste_text_images(background, images)
    # Return final image.
    return background, texts, bounds

def draw_texts_bounds(background, texts, bounds):
    background = background.copy()
    draw = ImageDraw.Draw(background)
    for atext, (x,y,w,h) in zip(texts, bounds):
        draw.rectangle([x,y,x+w,y+h], outline=255)
        draw.text((x,y+h), atext, fill=255)
    return background

# ----- MAIN

def main_text_samples():
    images = generate_text_images(TEXTS, text_width=(100,100), fill=255, rotation=0)
    for i, aimage in enumerate(images):
        aimage.save('cameras-trains/text_{:02}.png'.format(i))

def main_generated_samples():
    for i in range(5):
        image, texts, bounds = generate_image(size=(WIDTH, HEIGHT), texts=TEXTS*3, text_width=(12,16),
                                              color=(32,128), fill=(168,255), rotation=(-5,5))
        image = draw_texts_bounds(image, texts, bounds)
        image.save('cameras-trains/generated_{:02}.png'.format(i))

def main_dev_samples():
    CROP_W, CROP_H = 512, 512
    SCALE = 256 / 512
    for fp in sorted(glob.glob('cameras-metas/*.xml')):
        with open(fp) as f:
            print(fp, '...')
            dev_meta = xmltodict.parse(f.read())
            annotation = dev_meta['annotation']
            texts = []
            # --- Get coordinates.
            for obj in annotation['object']:
                if obj['name'] == 'carplate':
                    carplate = {}
                    carplate['xmin'] = int(obj['bndbox']['xmin'])
                    carplate['ymin'] = int(obj['bndbox']['ymin'])
                    carplate['xmax'] = int(obj['bndbox']['xmax'])
                    carplate['ymax'] = int(obj['bndbox']['ymax'])
                    crop_r = random.randint(carplate['ymax']-CROP_H, carplate['ymin'])
                    crop_c = random.randint(carplate['xmax']-CROP_W, carplate['xmin'])
                else:
                    q = {}
                    q['name'] = obj['name']
                    q['xmin'] = (int(obj['bndbox']['xmin'])-crop_c) * SCALE
                    q['ymin'] = (int(obj['bndbox']['ymin'])-crop_r) * SCALE
                    q['xmax'] = (int(obj['bndbox']['xmax'])-crop_c) * SCALE
                    q['ymax'] = (int(obj['bndbox']['ymax'])-crop_r) * SCALE
                    texts.append(q)
            # --- Save scaled coordinaes.
            with open('cameras-outputs/{}.json'.format(dev_meta['annotation']['filename']), 'w') as f:
                json.dump(texts, f, indent=4)
            # --- Crop based on coordinates.
            img = io.imread(dev_meta['annotation']['path'], as_grey=True)
            img = img[crop_r:crop_r+CROP_H, crop_c:crop_c+CROP_W]
            img = transform.rescale(img, SCALE)
            for q in texts:
                r = (q['ymin'], q['ymin'], q['ymax'], q['ymax'])
                c = (q['xmin'], q['xmax'], q['xmax'], q['xmin'])
                # img[draw.polygon_perimeter(r, c)] = 1
            io.imsave('cameras-outputs/{}'.format(dev_meta['annotation']['filename']), img)

if __name__ == '__main__':
    param = sys.argv[-1]
    if param == '--text-samples':
        main_text_samples()
    elif param == '--generate-samples':
        main_generated_samples()
    elif param == '--dev-samples':
        main_dev_samples()
    else:
        print('----- Doing Nothing')

