#!/usr/bin/env python3
import glob, json, os
import xmltodict
from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.truetype('__font.ttf', 20)

def extract(results):
    outputs = []
    for response in results['responses']:
        if 'fullTextAnnotation' not in response:
            continue

        for pages in response['fullTextAnnotation']['pages']:
            for block in pages['blocks']:
                for paragraph in block['paragraphs']:
                    for word in paragraph['words']:
                        for symbol in word['symbols']:
                            text = symbol['text']
                            bounds = [(v['x'], v['y']) for v in symbol['boundingBox']['vertices']]

                            # print(text, bounds)

                            # Ignore CCTV data and make bounds into rectangle.
                            valid = True
                            minx, maxx, miny, maxy = 9999, 0, 9999, 0
                            for x,y in bounds:
                                minx = min(minx, x)
                                maxx = max(maxx, x)
                                miny = min(miny, y)
                                maxy = max(maxy, y)

                                valid = valid and (y < 1000)
                                valid = valid and (y > 100)
                            # Only add if valid.
                            if valid:
                                bounds = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
                                outputs.append((text, bounds))
    return outputs

PATHS = ['datas{:03d}'.format(i) for i in range(7, 21)]

for apath in PATHS:

    # Loads the image into memory
    IMAGES = 'datas/{}/images'.format(apath)
    VISION = 'datas/{}/vision'.format(apath)
    RENDER = 'datas/{}/render'.format(apath)
    LABELS = 'datas/{}/labels'.format(apath)

    os.makedirs(RENDER, exist_ok=True)
    os.makedirs(LABELS, exist_ok=True)

    for idx, fp in enumerate(sorted(glob.glob('{}/*'.format(IMAGES)))):
        print(fp, idx)
        filename = fp.split('/')[-1]

        # Load
        try:
            datas = json.load(open('{}/{}.json'.format(VISION, filename)))
            datas = extract(datas)
            image = Image.open(fp)
        except:
            continue

        # # --- Draw
        # draw = ImageDraw.Draw(image)
        # for atext, coords in datas:
        #     draw.polygon(coords, outline=(255, 255, 255))
        #     x, y = coords[0]
        #     draw.text((x,y-100), atext, fill=(255, 255, 255), font=FONT)
        # # Save
        # image.save('{}/{}'.format(RENDER, filename))

        # --- Label
        objs = []

        for atext, coords in datas:
            bndbox = {}
            bndbox['xmin'] = coords[0][0]
            bndbox['ymin'] = coords[0][1]
            bndbox['xmax'] = coords[2][0]
            bndbox['ymax'] = coords[2][1]
            objs.append({'name': atext, 'bndbox': bndbox})

        # Save
        label = {}
        label['annotations'] = {'filename': filename, 'object': objs, }

        with open('{}/{}.xml'.format(LABELS, filename.split('.')[0]), 'w') as fo:
            fo.write(xmltodict.unparse(label, pretty=True))


