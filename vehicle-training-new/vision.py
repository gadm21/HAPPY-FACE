#!/usr/bin/env python3
import base64, glob, json, os
import requests
from google.cloud import vision
from google.auth.credentials import Credentials

APIKEY = '---INPUT API KEY HERE---'

PATH =  'datas/datas020'

# Loads the image into memory
IMAGES = '{}/images'.format(PATH)
VISION = '{}/vision'.format(PATH)
os.makedirs(VISION, exist_ok=True)

for idx, fp in enumerate(sorted(glob.glob('{}/*'.format(IMAGES)))):
    print(fp, idx)
    filename = fp.split('/')[-1]

    with open(fp, 'rb') as f:
        content = base64.b64encode(f.read()).decode('UTF-8')
        # Parameters
        params = {}
        params['requests'] = [
            {
                'image': {'content': content},
                'features': [
                    {
                        'type': 'TEXT_DETECTION',
                    },
                ]
            },
        ]
        # Request
        data = json.dumps(params)
        url = 'https://vision.googleapis.com/v1/images:annotate?key={}'.format(APIKEY)
        response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})
        # Save
        with open('{}/{}.json'.format(VISION, filename), 'w') as f:
            f.write(response.text)
