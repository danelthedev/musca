"""fair_backgammon wire client. stdlib only."""
import base64, hashlib, http.client, json, os, socket, struct, threading, urllib.parse

BASE = os.environ.get("BG_URL", "http://localhost:8080")

def _parts():
    u = urllib.parse.urlparse(BASE)
    return u.hostname or "localhost", u.port or 80, u.scheme

def api(method, path, body=None, cookie=""):
    host, port, scheme = _parts()
    cls = http.client.HTTPConnection if scheme != "https" else http.client.HTTPSConnection
    c = cls(host, port, timeout=10)
    hdr = {"Content-Type": "application/json"}
    if cookie:
        hdr["Cookie"] = cookie
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers=hdr)
    r = c.getresponse()
    raw = r.read()
    try:
        j = json.loads(raw) if raw else {}
    except Exception:
        j = {}
    setck = r.getheader("Set-Cookie", "")
    return r.status, j, setck

def session(username, cookie=""):
    st, j, setck = api("POST", "/api/session", {"username": username}, cookie)
    assert st == 200, f"session {st} {j}"
    ck = setck.split(";")[0] if setck else f"user={username}"
    return ck

def create_lobby(cookie):
    st, j, _ = api("POST", "/api/lobby", {}, cookie)
    assert st == 200, f"create {st} {j}"
    return j["code"]

def join_lobby(code, cookie):
    st, j, _ = api("POST", f"/api/lobby/{code}/join", {}, cookie)
    assert st == 200, f"join {st} {j}"

class WS:
    """Minimal RFC6455 client, text frames only."""
    def __init__(self, code, cookie):
        host, port, scheme = _parts()
        assert scheme in ("http", "ws"), "use http:// URL for ws too"
        self.sock = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET /ws?code={code} HTTP/1.1\r\nHost: {host}:{port}\r\n"
               "Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
               f"Cookie: {cookie}\r\n\r\n")
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            head += self.sock.recv(4096)
        assert b"101" in head.split(b"\r\n")[0], head[:200]
        self.lock = threading.Lock()
        self.buf = b""

    def send(self, obj):
        p = json.dumps(obj).encode()
        mask = os.urandom(4)
        if len(p) < 126:
            hdr = struct.pack("!BB", 0x81, 0x80 | len(p))
        elif len(p) < 65536:
            hdr = struct.pack("!BBH", 0x81, 0x80 | 126, len(p))
        else:
            hdr = struct.pack("!BBQ", 0x81, 0x80 | 127, len(p))
        mp = bytes(b ^ mask[i % 4] for i, b in enumerate(p))
        with self.lock:
            self.sock.sendall(hdr + mask + mp)

    def _fill(self, n):
        while len(self.buf) < n:
            d = self.sock.recv(65536)
            if not d:
                raise ConnectionError("ws closed")
            self.buf += d

    def recv(self, timeout=30):
        self.sock.settimeout(timeout)
        self._fill(2)
        b1, b2 = self.buf[0], self.buf[1]
        ln = b2 & 0x7F
        off = 2
        if ln == 126:
            self._fill(4)
            ln = struct.unpack("!H", self.buf[2:4])[0]
            off = 4
        elif ln == 127:
            self._fill(10)
            ln = struct.unpack("!Q", self.buf[2:10])[0]
            off = 10
        if b1 & 0x0F == 0x9:  # ping -> pong
            self._fill(off + ln)
            self.buf = self.buf[off + ln:]
            return {"t": "ping"}
        if b1 & 0x0F == 0x8:
            raise ConnectionError("ws close")
        masked = (b2 & 0x80) != 0
        if masked:
            self._fill(off + 4)
            mask = self.buf[off:off + 4]
            off += 4
        self._fill(off + ln)
        p = self.buf[off:off + ln]
        self.buf = self.buf[off + ln:]
        if masked:
            p = bytes(b ^ mask[i % 4] for i, b in enumerate(p))
        op = b1 & 0x0F
        if op == 0x1:
            return json.loads(p.decode() or "{}")
        return {"t": "bin"}

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
