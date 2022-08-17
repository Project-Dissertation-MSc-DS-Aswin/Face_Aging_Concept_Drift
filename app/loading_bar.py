import socketio
import eventlet

# call FastAPI app
async_mode = 'eventlet'
sio = socketio.Server(cors_allowed_origins='*', logger=True, async_mode=async_mode, always_connect=True)
app = socketio.WSGIApp(sio)
eventlet.wsgi.server(eventlet.listen(('', 8000)), app)

@sio.event
def connect(sid, environ):
  print(sid, 'connected')

@sio.event
def disconnect(sid):
  print(sid, 'disconnect')

@sio.event
def status(sid, data):
  print("status received with ", data)
  sio.emit('uploadProgress', data)