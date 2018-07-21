#!/usr/bin/python3

import socket
import requests
import time

def get_ip(portnumber):
    try:
        ip = requests.get('https://api.ipify.org').text
        return ip+':'+str(portnumber)
    except:
        return '127.0.0.1'+':'+str(portnumber)

def send_heartbeat(api_url, ip):
    timestamp = str(int(time.time()))
    r = requests.post(api_url, data={'ts':timestamp, 'public_ip_address': ip})

if __name__ == "__main__":
    myip = get_ip(portnumber=98)
    mac = '123456789ABC' # Please configure this mac address manually, for every new node
    api_url = 'http://192.168.0.4:8000/receiver/vehicle/heartbeat/{}'.format(mac) # Please configure the API URL also
    send_heartbeat(api_url, myip)
