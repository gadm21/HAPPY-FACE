# README

## Installation Dependencies
`pip3 install -r requirements.txt`

## Running the Edge Server services
`./run.sh` to start the server, model and heartbeat api. 

## ./run.sh
The `run.sh` script does the following in sequence:
* Add the `hearbeat.py` script to cron for every 10 minutes.
* Start the `worker.py` script (This monitors the filesystem for any new files that are created in the folder specified in `server_settings.py` (`SAVE_DIR` variable). It then pushes frames to AWS S3 ('vehicle-recognition' bucket by default) as soon as it is notified of the file creation.
* Start the `model.py` script where the `inference()` method is called.



## Configuration
**heartbeat.py** \
To configure the URL that `heartbeat.py` POSTs to: change the `api_url` in the main method.
The interval time at which heartbeat.py sends POSTs the ip address to `api_url` is in heartbeat_cron.txt, the line `*/10` currently specifies that the cron should schedule it at every 10 minutes, this can be made to `*/5` for every 5 minutes instead.

**server_settings.py** \
This file contains `SAVE_DIR` that stores the full path to where the frames are locally stored. This is used by `model.py` to store frames and `worker.py` to look for frames to push to AWS S3.


**config.json** \
All configurations should be made here. When the Vehicle Edge Server API (for updating start/stop/region) is used, this file is updated, and then reloaded with the updated values when `inference()` is called again.
Used to store model parameters - paths to weights, RTSP link (`RTSP_FEED`), voting algorithm parameters (`THRESHOLD` and `MEMORY`), CROP values (`CROP_XMIN`, `CROP_YMIN`, `CROP_XMAX`, `CROP_YMAX`), `TEXTS` for valid carplate character values. 

For each deployment here are the configuration changes that _must_ be made:
1. heartbeat.py `mac` and `api_url` 
2. heartbeat_cron.txt must contain **full path** to heartbeat.py
3. server_settings.py's `SAVE_DIR` must be set to the **full path** of the folder which `model.py` saves frames to and `worker.py` monitors.
4. config.json's `RTSP_FEED` must be set to the right link
5. Make sure `awscli` is installed on the machine, and `aws configure` (to set Access Key ID and Secret Key Access) has been setup. If `awscli` is not installed you may pip install it (one of the ways). 
	* `pip3 install awscli`


## Edge Server Create/Update Settings API
The `main.py` script is to run `server.py` (a Flask server) using Gevent - a more secure way of hosting a Flask server. It is hardcoded to server on port 5000.

POST Requests are sent to `http://[JETSON_IP]:5000/server/update/` \
Requests are sent in JSON, How to use each command:

**start** - starts the model.py and inferencing \
`{"status": "start"}`

**stop** - stops all instances of model.py \
`{"status": "stop"}`


[Example Region](doc/region.png)\
**configure** - enables configuration of crop region

`{"status": "configure", "region": [{"x":200, "y":300 }, {"x"1024:, "y":300 }, {"x":200, "y":700 }, {"x":1024, "y":700 }]}`

## What each script does
#### **heartbeat.py**
Sends a heartbeat message to a node (node link must be configured by setting the `mac` and `api_url` variables in the main method)
#### **server.py**
Starts a Flask HTTP Server, that can respond to POST requests such as the ones mentioned above (start/stop/configure)
#### **main.py**
Gevent server that runs the Flask server. Note that it is not recommended to run the Flask development server on it's own directly.
#### **model.py**
Loads weights from a model to recognize car number plates. Reads frame by frame from an RTSP feed, passes it through the model to recognize the characters, while executing a voting algorithm to pick the most frequent numberplate in a given window (`MEMORY` and `THRESHOLD` config parameters are used to tweak the behaviour of this voting algorithm. Once the model has recognized the carplate, the frame is saved to the folder (`SAVE_DIR` in `server_settings.py`). The frame is an image of the car, and the name of the image is in the following format:
`[SERVER_MAC]_[TIMESTAMP]_[CARPLATE]_x:y:w:h.png`, where _x, y, w, h_ are the values for cropping the number plate (_w_ = width, _h_ = height)

#### **worker.py**
Uses the `watchdog` library to get notifications on filesystem changes. `worker.py` is used to monitor the folder where the frames are saved in, when each new frame is created, this worker.py script is alerted, and the image created is pushed to AWS S3. to change the bucket which it is pushed to, change the `bucket` variable in the `worker.py` script's `upload_aws` method.
