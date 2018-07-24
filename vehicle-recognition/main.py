#!/usr/bin/python3

from gevent.pywsgi import WSGIServer
import server
import os
from server import app

os.system('fuser 5000/tcp -k')
server.start()
http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
