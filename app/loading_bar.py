import socketio
import eventlet
import sys
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

# call FastAPI app
async_mode = 'gevent'
sio = socketio.Server(cors_allowed_origins='*', logger=True, async_mode=async_mode, always_connect=True)
app = socketio.WSGIApp(sio)
pywsgi.WSGIServer(('', 8041), app,
                  handler_class=WebSocketHandler, log=sys.stdout).serve_forever()

@sio.event
def connect(sid, environ):
  print(sid, 'connect')

@sio.event
def disconnect(sid):
  print(sid, 'disconnect')

@sio.event
async def status(sid, data):
  print("status received with ", data)
  sio.emit('uploadProgress', data)