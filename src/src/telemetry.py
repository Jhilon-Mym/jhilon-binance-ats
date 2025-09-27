import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from urllib.parse import urlparse

_status = {
    'model_ok': False,
    'model_msg': '',
    'last_mean_conf': 0.0,
}

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(_status).encode('utf-8'))
        elif p.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(_status).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def start_server(port=8765):
    def _run():
        srv = HTTPServer(('0.0.0.0', port), _Handler)
        srv.serve_forever()
    t = threading.Thread(target=_run, daemon=True)
    t.start()

def update_status(model_ok: bool, model_msg: str, mean_conf: float = 0.0):
    _status['model_ok'] = model_ok
    _status['model_msg'] = model_msg
    _status['last_mean_conf'] = mean_conf

__all__ = ['start_server', 'update_status']
