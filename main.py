"""
Real-time webcam pain detection - fully TFLite (no TensorFlow / Keras needed).

Pipeline per frame:
    webcam frame (BGR)
        -> SCRFD tflite face detector (tflite_common.detect_faces)
        -> crop each face (+ margin), resize to 250x250
        -> quantized int8 tflite pain classifier
        -> per-face EMA smoothing
        -> draw box + "Pain xx%" / "No Pain xx%"

Usage:
    python live_pain_detection_tflite.py
    python live_pain_detection_tflite.py --npu
    python live_pain_detection_tflite.py --every 2
"""

import argparse
import sys
import time

import cv2
import numpy as np

from tflite_common import load_interpreter, detect_faces


# -------------------------------------------------------
# Config
# -------------------------------------------------------
DET_MODEL_DEFAULT = "models/det_500g_640x640_full_quantized.tflite"
PAIN_MODEL_DEFAULT = "models/pain_detection_full_integer_quant_v2.tflite"

PAIN_IMG_SIZE = 250
FACE_MARGIN = 0.20
EMA_ALPHA = 0.4
PAIN_THRESHOLD = 0.5
MAX_MISSED = 10

GREEN = (0, 200, 0)
RED = (0, 0, 255)


# -------------------------------------------------------
# TFLite pain classifier
# -------------------------------------------------------
class PainClassifier:
    """Wraps the quantized pain model.

    Input : 1x250x250x3, int8, quant (scale=1.0, zp=-128)
            -> real value = (q - zp) * scale = q + 128
            -> so q = pixel - 128, where pixel is the raw 0-255 BGR value
               the original Keras model was trained on.
    Output: 1x2, int8, quant (scale=1/256, zp=-128)
            -> prob = (q + 128) / 256 ; index 0 = pain
    """

    def __init__(self, model_path, use_npu=False, num_threads=4):
        self.interp = load_interpreter(model_path, use_npu=use_npu, num_threads=num_threads)
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

        self.in_dtype = self.inp["dtype"]
        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

        h, w = self.inp["shape"][1:3]
        self.size = (int(w), int(h))
        print(f"[PainClassifier] input {self.inp['shape']} {self.in_dtype.__name__}, "
              f"output {self.out['shape']} {self.out['dtype'].__name__}")

    def _quantize(self, x_float):
        """x_float: raw 0-255 float pixels -> model input dtype."""
        if self.in_dtype in (np.int8, np.uint8) and self.in_scale != 0:
            info = np.iinfo(self.in_dtype)
            q = np.round(x_float / self.in_scale + self.in_zp)
            return np.clip(q, info.min, info.max).astype(self.in_dtype)
        return x_float.astype(self.in_dtype)

    def _dequantize(self, q):
        if self.out_scale == 0:
            return q.astype(np.float32)
        return (q.astype(np.float32) - self.out_zp) * self.out_scale

    def predict(self, face_bgr):
        """face_bgr: cropped face (any size). Returns pain probability 0..1."""
        img = cv2.resize(face_bgr, self.size).astype(np.float32)   # BGR, 0-255
        batch = self._quantize(img)[None, ...]

        self.interp.set_tensor(self.inp["index"], batch)
        self.interp.invoke()
        out = self._dequantize(self.interp.get_tensor(self.out["index"]))[0]

        if out.size == 1:                  # single sigmoid-style output
            return float(out[0])
        # normalise in case the two outputs don't sum exactly to 1 after quantization
        s = float(out.sum())
        return float(out[0] / s) if s > 0 else float(out[0])


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def crop_face(frame, bbox, margin=FACE_MARGIN):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * margin))
    y1 = int(max(0, y1 - bh * margin))
    x2 = int(min(w, x2 + bw * margin))
    y2 = int(min(h, y2 + bh * margin))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return frame[y1:y2, x1:x2]


class FaceTrack:
    def __init__(self, center, prob):
        self.center = center
        self.prob = prob
        self.missed = 0


def update_tracks(tracks, bboxes, probs):
    assigned = [None] * len(bboxes)
    used = set()

    for i, (bbox, p) in enumerate(zip(bboxes, probs)):
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        diag = np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])

        best, best_d = None, diag
        for t_idx, t in enumerate(tracks):
            if t_idx in used:
                continue
            d = np.hypot(cx - t.center[0], cy - t.center[1])
            if d < best_d:
                best, best_d = t_idx, d

        if best is None:
            t = FaceTrack((cx, cy), p)
            tracks.append(t)
            used.add(len(tracks) - 1)
        else:
            t = tracks[best]
            t.prob = EMA_ALPHA * p + (1 - EMA_ALPHA) * t.prob
            t.center = (cx, cy)
            used.add(best)
        t.missed = 0
        assigned[i] = t

    for t_idx, t in enumerate(tracks):
        if t_idx not in used:
            t.missed += 1
    tracks[:] = [t for t in tracks if t.missed <= MAX_MISSED]
    return assigned


def draw_result(frame, bbox, pain_prob):
    x1, y1, x2, y2, _ = bbox
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    in_pain = pain_prob >= PAIN_THRESHOLD
    color = RED if in_pain else GREEN
    label = f"Pain ({pain_prob * 100:.0f}%)" if in_pain else f"No Pain ({pain_prob * 100:.0f}%)"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = max(y1 - 10, th + 6)
    cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), color, -1)
    cv2.putText(frame, label, (x1 + 3, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-model", default=DET_MODEL_DEFAULT)
    ap.add_argument("--pain-model", default=PAIN_MODEL_DEFAULT)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--npu", action="store_true", help="use VX NPU delegate for both models")
    ap.add_argument("--det-thresh", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=1,
                    help="run the pain classifier every N frames")
    args = ap.parse_args()

    det_interpreter = load_interpreter(args.det_model, use_npu=args.npu)
    pain = PainClassifier(args.pain_model, use_npu=args.npu)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform == "win32" \
        else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Pain detection started. Press 'q' to quit.")

    tracks = []
    last_probs = []
    frame_idx = 0
    fps, t_prev = 0.0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bboxes, _ = detect_faces(det_interpreter, frame, score_thresh=args.det_thresh)
        bboxes = [] if bboxes is None else list(bboxes)

        if bboxes:
            if frame_idx % args.every == 0 or len(last_probs) != len(bboxes):
                probs = []
                for b in bboxes:
                    crop = crop_face(frame, b)
                    probs.append(pain.predict(crop) if crop is not None else 0.0)
                last_probs = probs

            assigned = update_tracks(tracks, bboxes, last_probs)
            for b, t in zip(bboxes, assigned):
                draw_result(frame, b, t.prob)
        else:
            tracks.clear()
            last_probs = []

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Real-time Pain Detection (TFLite)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
