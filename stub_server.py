# stub_server.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        answer = {"text": f"[stub] got: {body['prompt']}"}
        body_bytes = json.dumps(answer).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()

        self.wfile.write(body_bytes)

if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8601), Handler).serve_forever()
