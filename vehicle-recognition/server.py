#!/usr/bin/python3

import os
import json
import model
import worker

from flask import Flask
from flask import request # Important for loading json

app = Flask(__name__)

def error_message(msg):
    return "ERROR: {}".format(msg)

def ok_message(msg):
    return "OK: {}".format(msg)

def save_configuration(xmin, ymin, xmax, ymax):
    print('Saving configuration')
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            config['CROP_XMIN'] = max(xmin, 0)
            config['CROP_YMIN'] = max(ymin, 0)
            config['CROP_XMAX'] = max(xmax, 0)
            config['CROP_YMAX'] = max(ymax, 0)
        with open('config.json', 'w') as outfile:
            json.dump(config, outfile)
        return ok_message('Saved model configuration')
    except Exception as e:
        print('Error opening config.json')
        print(e)
        return error_message('Could not configure model - region might be invalid')

def start():
    stop()
    print('Creating model.py process')
    os.system('python3 model.py &')
    return ok_message('model started')

def stop(): 
    print('Stopping all instances of model.py')
    processes = os.popen("ps aux | grep model.py | awk '{print $2}'")
    for pid in processes.readlines():
        os.system('kill '+pid)
    return ok_message('model stopped')

def configure_crop(dataDict):
    try:
        if 'x' in dataDict:
            x = dataDict['x']
        if 'y' in dataDict:
            y = dataDict['y']
        if 'w' in dataDict:
            w = dataDict['w']
        if 'h' in dataDict:
            h = dataDict['h']

        save_configuration(int(x), int(y), int(x+w), int(y+h))
        start()
        return ok_message('configured')
    except Exception as e:
        print(e)
        return error_message('not configured')

def valid_region(region):
    try:
        xmax = max([int(float(region[i]['x'])) for i in range(len(region))])
        xmin = min([int(float(region[i]['x'])) for i in range(len(region))])
        ymax = max([int(float(region[i]['y'])) for i in range(len(region))])
        ymin = min([int(float(region[i]['y'])) for i in range(len(region))])
    except Exception as e:
        print(e)
        return False, 'could not parse region'
    
    if xmax-xmin < 15*5 or ymax-ymin < 20:
        return False, 'region dimensions are too small'
        
    return True, 'region valid'

def get_region_values(region):
    xmax = max([int(float(region[i]['x'])) for i in range(len(region))])
    xmin = min([int(float(region[i]['x'])) for i in range(len(region))])
    ymax = max([int(float(region[i]['y'])) for i in range(len(region))])
    ymin = min([int(float(region[i]['y'])) for i in range(len(region))])
    return xmin, ymin, xmax, ymax
    
@app.route('/server/update', methods=['POST'])
def dispatcher():
    data = request.json
    dataDict = data
    print(dataDict)
    if 'status' in dataDict:
        status = dataDict['status']
        if status == 'start':
            return start()
        elif status == 'stop':
            return stop()
        elif status == 'configure':
            
            try:
                region = dataDict["region"]
            except:
                return error_message("no region specified")
            
            region = dataDict["region"]# Assumes region is a rectangle
            region_valid_status, msg = valid_region(region)
            if region_valid_status == True:
                 xmin, ymin, xmax, ymax = get_region_values(region)
                 try:
                     config_status = save_configuration(xmin, ymin, xmax, ymax)
                     start()
                     return config_status
                 except:
                     return error_message('could not save configuration')

            else:
                return error_message(msg)
        else:
            return error_message('invalid status')
    else:
        return error_message('invalid command')
