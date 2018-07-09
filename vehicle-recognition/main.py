#!/usr/bin/python3

from gevent.pywsgi import WSGIServer
import server
from server import app

server.start()
http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
