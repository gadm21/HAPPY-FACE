#!/usr/bin/python3

import socket
import requests
import time

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return '127.0.0.1'

def send_heartbeat(api_url, ip):
    timestamp = str(int(time.time()))
    r = requests.post(api_url, data={'ts':timestamp, 'public_ip_address': ip})

if __name__ == "__main__":
    myip = get_ip()
    mac = '123456789ABC' # Please configure this mac address manually, for every new node
    api_url = 'http://192.168.0.4:8000/receiver/vehicle/heartbeat/{}'.format(mac)
    send_heartbeat(api_url, myip)
