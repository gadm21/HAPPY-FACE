#!/usr/bin/python3

import time
import logging
import requests
import boto3

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
from watchdog.events import FileSystemEventHandler
from watchdog.events import FileCreatedEvent

import server_settings

s3 = boto3.client('s3')

def upload_aws(filepath, key):
    bucket = 'vehicle-recognition'
    s3.upload_file(filepath, bucket, key, ExtraArgs={'ACL': 'public-read'})
    print('Uploaded {}'.format(key))
    
def send_server(event):
    filepath = event.src_path
    filename = event.src_path[len(server_settings.SAVE_DIR):]
    print("SENDING SERVER: ", filename)
    upload_aws(filepath, filename)
        
if __name__ == "__main__":
    print("Worker started")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    event_handler = FileSystemEventHandler()
    event_handler.on_created = send_server

    observer = Observer()
    observer.schedule(event_handler, server_settings.SAVE_DIR, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

