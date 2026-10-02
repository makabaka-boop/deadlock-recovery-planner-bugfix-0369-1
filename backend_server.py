"""Stdlib HTTP backend exposing the deadlock simulator as a JSON API.

    POST /simulate   JSON body {"jobs": [...], "resources": [...]}
                     -> 200 with the replay produced by ``solve``,
                        400 with {"error": ...} on invalid input.
    GET  /health     -> {"status": "ok"}

Run directly to serve on 0.0.0.0:8000:

    python3 backend_server.py [port]
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from deadlock_simulator import solve
from checkpoint_recovery import solve_checkpoints


class SimulatorHandler(BaseHTTPRequestHandler):
    server_version = "DeadlockSimulator/1.0"

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/simulate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return
        try:
            use_checkpoints = isinstance(payload, dict) and payload.get('mode') == 'checkpoint'
            result = solve_checkpoints(payload) if use_checkpoints else solve(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, result)

    def log_message(self, *args):  # keep test output clean
        pass


def create_server(host="127.0.0.1", port=8000):
    return ThreadingHTTPServer((host, port), SimulatorHandler)


def main(argv):
    port = int(argv[1]) if len(argv) > 1 else 8000
    server = create_server("0.0.0.0", port)
    print(f"deadlock simulator backend listening on 0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv)
