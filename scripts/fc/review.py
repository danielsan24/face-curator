"""Local review page (stdlib http.server on 127.0.0.1). Images come from local files only (the curator's XL
cache, or an imported local copy); nothing leaves the machine.

Every request must carry this session's token (random, made at start-up, part of the URL that is opened):
- the page and its script/style and the images (`/crop`, `/photo`): query parameter `t` or the cookie the page
  sets (HttpOnly, SameSite=Strict, Path=/);
- `/api/*`: the header `X-FC-Token` (a header a foreign page cannot send without a CORS preflight).
The Host header must be 127.0.0.1:<port> or localhost:<port> (DNS rebinding), an Origin header must be the
server's own, and a POST must be `application/json` of at most 1 MB. Nothing is cached; a strict
Content-Security-Policy (no inline script) and `nosniff` are sent with every answer.

Endpoints: GET / /app.js /app.css ; GET /api/status ; GET /api/gallery?person=&bin= ; POST /api/gallery ;
GET /api/cards?kind= ; POST /api/card ; POST /api/card/undo ; GET /api/compare ; POST /api/compare ;
POST /api/compare/undo ; GET /crop/<face>?s= ; GET /photo/<face>?w=
"""

from __future__ import annotations

import hmac
import io
import json
import secrets
import sqlite3
import threading
import urllib.parse
from contextlib import contextmanager
from datetime import date
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fc import ages, cards
from fc import db as D
from fc.config import Config
from fc.library import image_path

STATIC_DIR = Path(__file__).resolve().parent.parent / "review"
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/app.css": ("app.css", "text/css; charset=utf-8")}
MAX_BODY = 1 << 20
CSP = ("default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; connect-src 'self'; "
       "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
BUSY = "the ledger is busy (a run is writing to it); nothing was saved, try again in a moment"


class App:
    def __init__(self, cfg: Config, db_path: Path):
        self.cfg = cfg
        self.db_path = db_path
        self.crop_dir = D.private_dir(cfg.data_dir / "crops")
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)        # this session only; never stored
        self.write_timeout = 5.0                      # seconds a save waits for a run that holds the write lock
        D.connect(db_path).close()                    # schema checked once, here, not on every request

    @contextmanager
    def db(self):
        """A connection that writes: short wait for the lock, so a busy ledger is an error, not a stalled page."""
        c = D.connect(self.db_path, timeout=self.write_timeout)
        try:
            yield c
            c.commit()
        finally:
            c.close()

    @contextmanager
    def ro(self):
        """A connection that only reads: it never waits for a run that is writing."""
        c = D.connect(self.db_path, timeout=5, readonly=True)
        try:
            yield c
        finally:
            c.close()

    # images
    def _image(self, db: sqlite3.Connection, item: int):
        from PIL import Image, ImageOps

        p = image_path(db, self.cfg, item, None)
        if p is None:
            return None
        try:
            im = Image.open(p)
            return ImageOps.exif_transpose(im).convert("RGB")
        except Image.DecompressionBombError as e:  # not an OSError: report it like any unreadable image
            raise OSError(str(e)) from None

    def has_image(self, db: sqlite3.Connection, item: int) -> bool:
        return image_path(db, self.cfg, item, None) is not None

    def crop(self, face: int, size: int) -> bytes | None:
        size = max(64, min(size, 512))
        cached = self.crop_dir / f"{face // 1000:04d}" / f"{face}_{size}.jpg"
        if cached.is_file():
            data = cached.read_bytes()
            if data[:3] == b"\xff\xd8\xff":
                return data
        with self.ro() as db:
            r = db.execute("SELECT item, x1, y1, x2, y2 FROM faces WHERE id=?", (face,)).fetchone()
            if r is None:
                return None
            im = self._image(db, r["item"])
        if im is None:
            return None
        w, h = im.size
        x1, y1, x2, y2 = r["x1"] * w, r["y1"] * h, r["x2"] * w, r["y2"] * h
        mx, my = (x2 - x1) * 0.35, (y2 - y1) * 0.35
        c = im.crop((max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)))
        c.thumbnail((size, size))
        buf = io.BytesIO()
        c.save(buf, "JPEG", quality=88)
        D.atomic_write(cached, buf.getvalue())         # two requests for one crop: each renames a whole file
        return buf.getvalue()

    def photo(self, face: int, width: int) -> bytes | None:
        from PIL import ImageDraw

        width = max(200, min(width, 2000))
        with self.ro() as db:
            r = db.execute("SELECT item, x1, y1, x2, y2 FROM faces WHERE id=?", (face,)).fetchone()
            if r is None:
                return None
            im = self._image(db, r["item"])
        if im is None:
            return None
        im.thumbnail((width, width))
        w, h = im.size
        ImageDraw.Draw(im).rectangle((r["x1"] * w, r["y1"] * h, r["x2"] * w, r["y2"] * h), outline=(255, 200, 0),
                                     width=max(2, w // 300))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()

    # data
    def status(self) -> dict:
        cfg, edges = self.cfg, self.cfg["age_bins_months"]
        with self.ro() as db:
            labels = D.latest_labels(db)
            taken = {r[0]: (r[1] or "")[:10] for r in db.execute("SELECT f.id, i.taken FROM faces f JOIN items i ON i.id=f.item "
                                                                 "WHERE f.id IN (SELECT face FROM labels)")}
            people = []
            for p in cfg.close_people:
                mine = [f for f, v in labels.items() if v == p.key]
                if p.child:
                    bins = []
                    for i in range(len(edges)):
                        if ages.months(p.born, date.today().isoformat()) < edges[i]:
                            continue           # that age has not been reached yet
                        n = sum(1 for f in mine if ages.bin_index(ages.months(p.born, taken.get(f)), edges) == i)
                        bins.append({"bin": i, "label": ages.bin_label(i, edges), "n": n})
                    people.append({"key": p.key, "name": p.name, "child": True, "bins": bins})
                else:
                    people.append({"key": p.key, "name": p.name, "child": False,
                                   "bins": [{"bin": None, "label": "all", "n": len(mine)}]})
            waiting = {r[0]: r[1] for r in db.execute("SELECT kind, COUNT(*) FROM cards WHERE answer IS NULL GROUP BY kind")}
        g = cfg["gallery"]
        return {"people": people, "target": g["target"], "max": g["max"], "min_active": g["min_active"],
                "cards_waiting": waiting}

    def gallery(self, person: str, bin_i: int | None) -> list[dict]:
        if person not in self.cfg.people:
            raise ValueError("unknown person")
        with self.ro() as db:
            return cards.gallery_candidates(db, self.cfg, person, bin_i, self.cfg["gallery"]["batch"],
                                            self.cfg["models"]["reference"], lambda it: self.has_image(db, it))

    def save_gallery(self, body: dict) -> dict:
        person = body["person"]
        if not isinstance(person, str) or person not in self.cfg.people:
            raise ValueError("unknown person")
        todo = []
        for field, value in (("yes", person), ("no", f"not:{person}"), ("not_face", "not_face")):
            faces = body.get(field, [])
            if not isinstance(faces, list) or any(isinstance(f, bool) or not isinstance(f, int) for f in faces):
                raise ValueError(f"{field} must be a list of face ids")
            todo += [(f, value) for f in faces]
        if len(todo) > 500:
            raise ValueError("too many faces in one batch")
        with self.lock, self.db() as db:
            known = {r[0] for r in db.execute("SELECT id FROM faces WHERE id IN (%s)" % ",".join("?" * len(todo)),
                                              [f for f, _ in todo])} if todo else set()
            if len(known) != len({f for f, _ in todo}):
                raise ValueError("unknown face id")
            for f, value in todo:
                D.add_label(db, f, value, via="gallery")
            db.commit()
        return {"saved": len(todo)}

    def open_cards(self, kind: str | None) -> list[dict]:
        out = []
        with self.ro() as db:
            q = ("SELECT c.*, i.taken FROM cards c JOIN faces f ON f.id=c.face JOIN items i ON i.id=f.item "
                 "WHERE c.answer IS NULL AND c.kind NOT IN ('h2h','study','edge')")   # comparison cards are blind: never shown here
            params: list = []
            if kind:
                q += " AND c.kind=?"
                params.append(kind)
            for c in db.execute(q + " ORDER BY c.id LIMIT 40", params).fetchall():
                out.append({"id": c["id"], "kind": c["kind"], "face": c["face"], "day": (c["taken"] or "")[:10],
                            "bin": c["bin"], "a": self._side(db, c["person"], c["bin"], c["face"]),
                            "b": self._side(db, c["alt"], c["bin"], c["face"])})
        return out

    def _side(self, db: sqlite3.Connection, who: str | None, binl: str | None, exclude: int) -> dict:
        if not who:
            return {"key": None, "name": "(none)", "refs": []}
        if who.startswith("syno:"):
            gid = int(who[5:])
            refs = [r[0] for r in db.execute("SELECT f.id FROM faces f JOIN syno_faces s ON s.face_id=f.syno_face_id "
                                              "WHERE s.person_id=? AND f.id != ? LIMIT 60", (gid, exclude))]
            name = (db.execute("SELECT name FROM syno_people WHERE id=?", (gid,)).fetchone() or [""])[0] or f"group #{gid}"
        else:
            p = self.cfg.people.get(who)
            name = p.name if p else who
            labels = D.latest_labels(db)
            refs = [f for f, v in labels.items() if v == who and f != exclude]
            if p and p.child and binl and binl not in ("-", "all"):
                edges = self.cfg["age_bins_months"]
                taken = {r[0]: (r[1] or "")[:10] for r in db.execute(
                    "SELECT f.id, i.taken FROM faces f JOIN items i ON i.id=f.item WHERE f.id IN (%s)" % ",".join("?" * len(refs)),
                    refs)} if refs else {}
                same = [f for f in refs if ages.bin_label(ages.bin_index(ages.months(p.born, taken.get(f)), edges), edges) == binl]
                refs = same or refs
        refs = [f for f in refs if self.has_image(db, db.execute("SELECT item FROM faces WHERE id=?", (f,)).fetchone()[0])][:6]
        return {"key": who, "name": name, "refs": refs}

    # blind comparison: no system's answer ever leaves the server
    def compare_cards(self) -> dict:
        from fc import compare
        with self.ro() as db:
            rows = db.execute("SELECT c.id, c.face, i.taken, c.payload FROM cards c JOIN faces f ON f.id=c.face JOIN items i "
                              "ON i.id=f.item WHERE c.kind IN ('h2h','study','edge') AND c.answer IS NULL ORDER BY c.id").fetchall()
            done = db.execute("SELECT COUNT(*) FROM cards WHERE kind IN ('h2h','study','edge') AND answer IS NOT NULL").fetchone()[0]
        return {"people": [{"key": p.key, "name": p.name} for p in self.cfg.close_people], "answered": done,
                "cards": [{"id": r[0], "face": r[1], "day": (r[2] or "")[:10],
                           **({"options": [{"key": k, "name": self.cfg.people[k].name}   # shuffled at build: no pick
                                           for k in opts if k in self.cfg.people]}
                              if (opts := compare.card_options({"payload": r[3]})) else {})}
                          for r in rows]}

    def compare_answer(self, body: dict) -> dict:
        from fc import compare
        with self.lock, self.db() as db:
            return compare.answer(db, self.cfg, body["id"], body["answer"], replace=body.get("replace") is True)

    def compare_undo(self, body: dict) -> dict:
        from fc import compare
        cid = body["id"]
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise ValueError("card id must be an integer")
        with self.lock, self.db() as db:
            c = db.execute("SELECT * FROM cards WHERE id=? AND kind IN ('h2h','study','edge')", (cid,)).fetchone()
            if c is None:
                return {"undone": False}
            return {"undone": compare.retract(db, c)}

    def answer(self, body: dict) -> dict:
        """An answered card is refused (409) unless the page sends `replace: true` on purpose."""
        with self.lock, self.db() as db:
            return cards.answer(db, self.cfg, body["id"], body["answer"], replace=body.get("replace") is True)

    def undo(self, body: dict) -> dict:
        cid = body["id"]
        if isinstance(cid, bool) or not isinstance(cid, int):
            raise ValueError("card id must be an integer")
        with self.lock, self.db() as db:
            c = db.execute("SELECT * FROM cards WHERE id=? AND kind NOT IN ('h2h','study','edge')", (cid,)).fetchone()
            if c is None:
                return {"undone": False}
            return {"undone": cards.retract(db, self.cfg, c)}


class Refuse(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


def handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, cookie: str | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        # --- who may talk to this server ---------------------------------------------------------------
        def _port(self) -> int:
            return self.server.server_address[1]

        def _cookie_name(self) -> str:
            return f"fc_token_{self._port()}"

        def _guard(self, path: str, query: dict) -> None:
            port = self._port()
            if self.headers.get("Host", "") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                raise Refuse(403, "this page answers only as 127.0.0.1 or localhost")
            origin = self.headers.get("Origin")
            if origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                raise Refuse(403, "requests from another origin are refused")
            if path.startswith("/api/"):
                given = [self.headers.get("X-FC-Token")]
            else:
                jar = SimpleCookie()
                try:
                    jar.load(self.headers.get("Cookie", ""))
                except Exception:  # noqa: BLE001 - a malformed Cookie header is simply no cookie
                    jar = SimpleCookie()
                given = [query.get("t"), jar[self._cookie_name()].value if self._cookie_name() in jar else None]
            if not any(isinstance(g, str) and hmac.compare_digest(g.encode(), app.token.encode()) for g in given):
                raise Refuse(403, "missing or wrong session token: open the URL that `facecurator review` printed")

        def _fail(self, e: Exception) -> None:
            if isinstance(e, Refuse):
                return self._json({"error": e.msg}, e.code)
            if isinstance(e, cards.AlreadyAnswered):
                return self._json({"error": str(e), "answered": True}, 409)
            if isinstance(e, sqlite3.OperationalError):
                return self._json({"error": BUSY}, 503)
            self._json({"error": str(e) or type(e).__name__}, 400)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            try:
                self._guard(u.path, q)
                if u.path in STATIC:
                    name, ctype = STATIC[u.path]
                    cookie = f"{self._cookie_name()}={app.token}; HttpOnly; SameSite=Strict; Path=/" if u.path == "/" else None
                    return self._send(200, (STATIC_DIR / name).read_bytes(), ctype, cookie)
                if u.path == "/api/status":
                    return self._json(app.status())
                if u.path == "/api/gallery":
                    b = q.get("bin")
                    return self._json(app.gallery(q["person"], int(b) if b not in (None, "", "null") else None))
                if u.path == "/api/cards":
                    return self._json(app.open_cards(q.get("kind") or None))
                if u.path == "/api/compare":
                    return self._json(app.compare_cards())
                if u.path.startswith("/crop/"):
                    data = app.crop(int(u.path[6:]), int(q.get("s", 160)))
                    return self._send(200, data, "image/jpeg") if data else self._send(404, b"", "text/plain")
                if u.path.startswith("/photo/"):
                    data = app.photo(int(u.path[7:]), int(q.get("w", 900)))
                    return self._send(200, data, "image/jpeg") if data else self._send(404, b"", "text/plain")
                self._send(404, b"not found", "text/plain")
            except OSError:  # an unreadable or damaged local image (PIL errors are OSErrors)
                self._json({"error": "image not available"}, 404)
            except (Refuse, KeyError, ValueError, TypeError, sqlite3.OperationalError) as e:
                self._fail(e)

        def _body(self) -> dict:
            if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
                raise Refuse(415, "Content-Type must be application/json")
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise Refuse(400, "Content-Length is required")
            if n < 0:
                raise Refuse(400, "bad Content-Length")
            if n > MAX_BODY:
                raise Refuse(413, "request body too large")
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            return body

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            routes = {"/api/gallery": app.save_gallery, "/api/card": app.answer, "/api/card/undo": app.undo,
                      "/api/compare": app.compare_answer, "/api/compare/undo": app.compare_undo}
            try:
                self._guard(u.path, {})
                if u.path not in routes:
                    return self._send(404, b"not found", "text/plain")
                self._json(routes[u.path](self._body()))
            except (Refuse, KeyError, ValueError, TypeError, sqlite3.OperationalError) as e:
                self.close_connection = True          # the body may not have been read
                self._fail(e)

    return H


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), handler(app))


def serve(cfg: Config, db_path: Path, port: int | None = None, open_browser: bool = True) -> None:
    app = App(cfg, db_path)
    srv = make_server(app, port if port is not None else cfg["review"]["port"])
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t={app.token}"
    print(f"review page: {url}\n(the token in the URL is this session's key: do not share it; keep this running "
          "while you label; Ctrl-C stops it)")
    if open_browser:
        import subprocess
        subprocess.run(["open", url], check=False)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
