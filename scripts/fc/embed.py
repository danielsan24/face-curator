"""Face detection (SCRFD) and embedding (ArcFace-style models) on onnxruntime, without the insightface package.

Preprocessing reproduces insightface 0.7 (FaceAnalysis buffalo_l, det_size 640, det_thresh 0.5, norm_crop with
the arcface template), so new buffalo_l embeddings are comparable with insightface's own (and with an imported
face cache). `selftest` checks that on real photos (cosine with the stored embedding).

Model registry: `MODELS`. A model file is loaded only if its sha256 matches `models.lock.json` (written when the
owner registers it with `facecurator models add-buffalo` / `models add`); the hash is checked every time a
model is loaded, once per process. The two buffalo_l files must also match the hashes pinned here (`PINNED`).
Licences: InsightFace models (buffalo_l, antelopev2) are for non-commercial research use; AdaFace weights are
MIT. The weights are not part of this repository.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    file: str               # file name inside the models dir (or an absolute path recorded in the lock)
    rgb: bool               # True: feed RGB (insightface models); False: BGR (AdaFace's original code)
    mean: float = 127.5
    std: float = 127.5
    size: int = 112
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "buffalo_l": ModelSpec("buffalo_l", "buffalo_l/w600k_r50.onnx", rgb=True,
                           note="InsightFace buffalo_l recogniser (ResNet-50, WebFace600K). Reference model."),
    "antelopev2": ModelSpec("antelopev2", "antelopev2/glintr100.onnx", rgb=True,
                            note="InsightFace antelopev2 recogniser (ResNet-100, Glint360K)."),
    "adaface_ir101_webface12m": ModelSpec("adaface_ir101_webface12m", "adaface/adaface_ir101_webface12m.onnx",
                                          rgb=False, note="AdaFace IR-101 WebFace12M, ONNX export. BGR input."),
}
DETECTOR = "buffalo_l/det_10g.onnx"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# sha256 of the two buffalo_l files this code was written and measured against (InsightFace model zoo,
# buffalo_l.zip). A file registered under these names with another hash is refused.
PINNED: dict[str, str] = {
    "buffalo_l/det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "buffalo_l/w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
_VERIFIED: dict[tuple, str] = {}     # (path, size, mtime_ns) -> sha256, per process


class Registry:
    """models.lock.json: {relative file: {sha256, path, size}}. The pinned buffalo_l files must match `PINNED`;
    any other model is trusted when the owner registers it by name, and verified against that hash after."""

    def __init__(self, models_dir: Path):
        self.dir = models_dir
        self.lock_path = models_dir / "models.lock.json"
        self.lock: dict[str, dict] = json.loads(self.lock_path.read_text()) if self.lock_path.is_file() else {}

    def register(self, rel: str, src: Path) -> dict:
        from fc.db import atomic_write

        src = src.expanduser().resolve()
        if not src.is_file():
            raise SystemExit(f"model file not found: {src}")
        digest = sha256(src)
        if rel in PINNED and digest != PINNED[rel]:
            raise SystemExit(f"{src} is not the {rel} this skill expects (sha256 {digest[:16]}…, expected "
                             f"{PINNED[rel][:16]}…): not registered. Download buffalo_l again from the InsightFace "
                             "model zoo (references/models.md); to use another recogniser, register it under its "
                             "own name with `models add NAME --file PATH`.")
        entry = {"path": str(src), "sha256": digest, "size": src.stat().st_size}
        self.lock[rel] = entry
        atomic_write(self.lock_path, json.dumps(self.lock, indent=2))
        return entry

    def path(self, rel: str) -> Path:
        """The registered file, after checking that it is still the file that was registered."""
        e = self.lock.get(rel)
        if not e:
            raise SystemExit(f"model {rel} not registered: run `facecurator models add-buffalo` or `models add` "
                             "(see references/models.md)")
        p = Path(e["path"])
        if not p.is_file() or p.stat().st_size != e["size"]:
            raise SystemExit(f"model {rel} changed or missing at {p}; re-register it after checking its source")
        st = p.stat()
        key = (str(p), st.st_size, st.st_mtime_ns)
        if key not in _VERIFIED:
            _VERIFIED[key] = sha256(p)
        want = PINNED.get(rel, e["sha256"])
        if _VERIFIED[key] != e["sha256"] or _VERIFIED[key] != want:
            raise SystemExit(f"model {rel} at {p} does not match its registered sha256 ({_VERIFIED[key][:16]}… "
                             f"instead of {want[:16]}…): refused. Re-register it after checking its source.")
        return p

    def verify(self, rel: str) -> bool:
        try:
            self.path(rel)
        except SystemExit:
            return False
        return True


def _session(path: Path) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


# --- geometry --------------------------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similarity transform src->dst (2x3), same as skimage SimilarityTransform.estimate (used by insightface)."""
    num, dim = src.shape
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_d, dst_d = src - src_mean, dst - dst_mean
    A = dst_d.T @ src_d / num
    d = np.ones(dim)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.full((2, 3), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    scale = 1.0 / src_d.var(0).sum() * (S @ d)
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T[:2]


def norm_crop(img: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    import cv2

    M = umeyama(np.asarray(kps, dtype=np.float32), ARCFACE_DST * (size / 112.0))
    return cv2.warpAffine(img, M, (size, size), borderValue=0.0)


def distance2bbox(points: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.stack([points[:, 0] - d[:, 0], points[:, 1] - d[:, 1], points[:, 0] + d[:, 2], points[:, 1] + d[:, 3]], -1)


def distance2kps(points: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = []
    for i in range(0, d.shape[1], 2):
        out += [points[:, i % 2] + d[:, i], points[:, i % 2 + 1] + d[:, i + 1]]
    return np.stack(out, -1)


def nms(dets: np.ndarray, thresh: float = 0.4) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


# --- detector --------------------------------------------------------------------------------------------
class Detector:
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, path: Path, size: int = 640, thresh: float = 0.5):
        self.sess = _session(path)
        self.input = self.sess.get_inputs()[0].name
        self.outputs = [o.name for o in self.sess.get_outputs()]
        self.size, self.thresh = size, thresh
        self._centers: dict[tuple, np.ndarray] = {}

    def _anchors(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape((-1, 2))
            self._centers[key] = np.stack([c] * self.NUM_ANCHORS, axis=1).reshape((-1, 2))
        return self._centers[key]

    def detect(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """img: BGR uint8. Returns (N x 5 [x1,y1,x2,y2,score] in pixels, N x 5 x 2 keypoints)."""
        import cv2

        S = self.size
        im_ratio = img.shape[0] / img.shape[1]
        if im_ratio > 1.0:
            new_h, new_w = S, int(S / im_ratio)
        else:
            new_w, new_h = S, int(S * im_ratio)
        det_scale = new_h / img.shape[0]
        det_img = np.zeros((S, S, 3), dtype=np.uint8)
        det_img[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, (S, S), (127.5, 127.5, 127.5), swapRB=True)
        outs = [o[0] if o.ndim == 3 else o for o in self.sess.run(self.outputs, {self.input: blob})]
        return decode(outs, S, det_scale, self.thresh, self.STRIDES, self._anchors)


def decode(outs: list[np.ndarray], S: int, det_scale: float, thresh: float, strides=(8, 16, 32),
           anchors=None) -> tuple[np.ndarray, np.ndarray]:
    fmc = len(strides)
    scores_l, boxes_l, kps_l = [], [], []
    for idx, stride in enumerate(strides):
        scores = outs[idx].reshape(-1)
        bbox = outs[idx + fmc] * stride
        kps = outs[idx + fmc * 2] * stride
        h = w = S // stride
        if anchors is not None:
            centers = anchors(h, w, stride)
        else:
            c = (np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32) * stride).reshape((-1, 2))
            centers = np.stack([c] * 2, axis=1).reshape((-1, 2))
        pos = np.where(scores >= thresh)[0]
        scores_l.append(scores[pos])
        boxes_l.append(distance2bbox(centers, bbox)[pos])
        kps_l.append(distance2kps(centers, kps).reshape((len(centers), -1, 2))[pos])
    scores = np.concatenate(scores_l)
    if scores.size == 0:
        return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
    order = scores.argsort()[::-1]
    boxes = np.concatenate(boxes_l) / det_scale
    kpss = np.concatenate(kps_l) / det_scale
    pre = np.hstack((boxes, scores[:, None])).astype(np.float32)[order]
    keep = nms(pre)
    return pre[keep], kpss[order][keep]


# --- recogniser ------------------------------------------------------------------------------------------
class Recognizer:
    def __init__(self, spec: ModelSpec, path: Path):
        self.spec = spec
        self.sess = _session(path)
        self.input = self.sess.get_inputs()[0].name

    def embed(self, img: np.ndarray, kps_list: list[np.ndarray]) -> np.ndarray:
        """img: BGR uint8; returns L2-normalised embeddings (N x D)."""
        import cv2

        if not kps_list:
            return np.zeros((0, 512), np.float32)
        crops = [norm_crop(img, k, self.spec.size) for k in kps_list]
        blob = cv2.dnn.blobFromImages(crops, 1.0 / self.spec.std, (self.spec.size,) * 2, (self.spec.mean,) * 3,
                                      swapRB=self.spec.rgb)
        out = self.sess.run(None, {self.input: blob})[0]
        out = out.reshape(len(crops), -1).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-6)


def load_image(path: Path, max_side: int) -> np.ndarray | None:
    """BGR, upright (cv2.imread applies EXIF orientation), longest side <= max_side (1600 by default)."""
    import cv2

    im = cv2.imread(str(path))
    if im is None:
        return None
    if max(im.shape[:2]) > max_side:
        s = max_side / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return im


class Engine:
    """Detector + the active recognisers, loaded lazily."""

    def __init__(self, models_dir: Path, models: list[str]):
        self.reg = Registry(models_dir)
        self.models = models
        self._det: Detector | None = None
        self._rec: dict[str, Recognizer] = {}

    @property
    def det(self) -> Detector:
        if self._det is None:
            self._det = Detector(self.reg.path(DETECTOR))
        return self._det

    def rec(self, name: str) -> Recognizer:
        if name not in self._rec:
            spec = MODELS[name]
            self._rec[name] = Recognizer(spec, self.reg.path(spec.file))
        return self._rec[name]

    def faces(self, img: np.ndarray) -> list[dict]:
        """[{box (fractions), det, kps (pixels)}]."""
        h, w = img.shape[:2]
        dets, kpss = self.det.detect(img)
        out = []
        for d, k in zip(dets, kpss):
            x0, y0, x1, y1, s = (float(v) for v in d)
            out.append({"box": (max(x0 / w, 0.0), max(y0 / h, 0.0), min(x1 / w, 1.0), min(y1 / h, 1.0)),
                        "det": round(s, 3), "kps": k.astype(np.float32)})
        return out
