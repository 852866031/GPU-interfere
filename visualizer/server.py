#!/usr/bin/env python3
"""Replay GUI backend — stdlib only (no deps).

  GET /api/runs          -> ["dummy_prefill", ...]
  GET /api/runs/<name>   -> full run JSON (see dummy_run.py for the contract)
  GET /...               -> static files from web/

Run:  python3 server.py [port]   (default 8000)

Later, extract.py drops real runs as JSON files into runs/ — they are picked up
automatically alongside the generated dummy.
"""
import http.server, json, os, sys, functools
from dummy_run import make_dummy_run

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
RUNS_DIR = os.path.join(HERE, "runs")          # real extracted runs (JSON), optional


@functools.lru_cache(maxsize=None)
def get_run(name):
    if name == "dummy_prefill":
        return json.dumps(make_dummy_run()).encode()
    path = os.path.join(RUNS_DIR, os.path.basename(name) + ".json")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    return None


def list_runs():
    runs = ["dummy_prefill"]
    if os.path.isdir(RUNS_DIR):
        runs += sorted(f[:-5] for f in os.listdir(RUNS_DIR) if f.endswith(".json"))
    return runs


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def do_GET(self):
        if self.path == "/api/runs":
            return self._json(json.dumps(list_runs()).encode())
        if self.path.startswith("/api/runs/"):
            body = get_run(self.path[len("/api/runs/"):])
            if body is None:
                self.send_error(404, "unknown run")
                return
            return self._json(body)
        super().do_GET()

    def _json(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):           # quieter log
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"replay GUI: http://localhost:{port}  (runs: {', '.join(list_runs())})")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
