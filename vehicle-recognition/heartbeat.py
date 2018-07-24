#!/usr/bin/python3

import socket
import requests
import time

def get_ip(portnumber=80):
    try:
        ip = requests.get('https://api.ipify.org').text
        return ip+':'+str(portnumber)
    except:
        return '127.0.0.1'+':'+str(portnumber)

def send_heartbeat(api_url, ip):
    timestamp = str(int(time.time()))
    r = requests.post(api_url, data={'ts':timestamp, 'public_ip_address': ip})

if __name__ == "__main__":
    myip = get_ip() # Specify parameter portnumber, default is 80
    mac = '14A78BA221FD' # Please configure this mac address manually, for every new node
    api_url = 'http://data.gotapway.com/receiver/vehicle/heartbeat/{}'.format(mac) # Please configure the API URL also
    send_heartbeat(api_url, myip)
