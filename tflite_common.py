"""
Shared utilities for running the quantized (uint8) InsightFace models:
  - det_10g_full_quantized.tflite   (SCRFD-10g face detector)
  - w600k_r50_full_quantized.tflite (ArcFace R50 recognizer)

These replace insightface.model_zoo / insightface.app.common.Face, which only
know how to drive the original ONNX graphs. With a fully-quantized tflite
model we have to do the pre/post-processing ourselves:
  - feed uint8 pixels directly (the model's input quant params already encode
    the img/255 normalization that was baked in at conversion time)
  - manually dequantize every output tensor using its own (scale, zero_point)
  - manually decode SCRFD's per-stride anchors -> boxes/scores/keypoints
  - manually align faces to 112x112 before feeding the recognizer
"""

import numpy as np
import cv2

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python import interpreter as tflite


# ============================================================
# Interpreter loading
# ============================================================
NPU_DELEGATE_PATH = "/usr/lib/libvx_delegate.so"


def load_interpreter(model_path, use_npu=False, num_threads=4):
    """If use_npu=True, tries to load the VX (NPU) delegate from
    /usr/lib/libvx_delegate.so and run the model on it. Falls back to the
    default CPU interpreter if the delegate can't be loaded (e.g. on a PC,
    where this delegate doesn't exist at all - it's for NXP i.MX/VeriSilicon
    NPUs specifically).

    num_threads controls XNNPACK/CPU thread count used when running on CPU.
    Ignored when an NPU delegate is actually active.
    """
    delegates = []

    if use_npu:
        try:
            vx_delegate = tflite.load_delegate(NPU_DELEGATE_PATH)
            delegates.append(vx_delegate)
            print(f"[tflite_common] NPU delegate loaded: {NPU_DELEGATE_PATH}")
        except (ValueError, OSError) as e:
            print(f"[tflite_common] Failed to load NPU delegate ({e}), falling back to CPU.")

    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=delegates,
        num_threads=num_threads if not delegates else None,
    )
    interpreter.allocate_tensors()
    return interpreter


def _dequantize(arr, quant_params):
    """quant_params = (scale, zero_point) as returned by tflite's
    get_output_details()[i]['quantization']."""
    scale, zero_point = quant_params
    if scale == 0:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - zero_point) * scale


def run_interpreter(interpreter, input_uint8):
    """input_uint8: HxWx3 uint8 array (already resized to the model's
    expected input size). Returns a dict: {output_index: dequantized_array}
    keyed by the order of get_output_details()."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = np.expand_dims(input_uint8, axis=0).astype(np.uint8)  # NHWC
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()

    outputs = []
    for od in output_details:
        raw = interpreter.get_tensor(od['index'])
        scale, zero_point = od['quantization']
        deq = _dequantize(raw, (scale, zero_point))
        outputs.append(deq)

    return outputs, output_details


# ============================================================
# SCRFD (det_10g) decoding
# ============================================================
INPUT_SIZE = 640
FEAT_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2

_ANCHOR_CACHE = {}


def _get_anchor_centers(stride):
    """Anchor centers only depend on stride/input size, which never change
    at runtime - compute once and reuse instead of rebuilding every frame."""
    if stride in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[stride]

    height = INPUT_SIZE // stride
    width = INPUT_SIZE // stride

    anchor_centers = np.stack(
        np.mgrid[:height, :width][::-1], axis=-1
    ).astype(np.float32)
    anchor_centers = (anchor_centers * stride).reshape(-1, 2)
    if NUM_ANCHORS > 1:
        anchor_centers = np.repeat(anchor_centers, NUM_ANCHORS, axis=0)

    _ANCHOR_CACHE[stride] = anchor_centers
    return anchor_centers


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets, thresh=0.4):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def letterbox(img, target_size=INPUT_SIZE):
    """Resize keeping aspect ratio, pad to target_size x target_size.
    Returns padded uint8 RGB image and the scale factor to map detections
    back to the original image."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(img, (new_w, new_h))
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    return padded, scale


def detect_faces(interpreter, frame_bgr, score_thresh=0.5, nms_thresh=0.4):
    """Runs det_10g_full_quantized.tflite on a BGR frame (as read by cv2).
    Returns (bboxes, kpss) matching the shape/semantics of the old
    det_model.detect() call:
        bboxes: Nx5 (x1,y1,x2,y2,score)
        kpss:   Nx5x2
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    padded, scale = letterbox(img_rgb, INPUT_SIZE)

    raw_outputs, output_details = run_interpreter(interpreter, padded)

    # Group outputs by their (rows) dimension -> which stride they belong to,
    # and by their last dim -> whether they are scores(1) / bbox(4) / kps(10).
    by_rows = {}
    for out, od in zip(raw_outputs, output_details):
        arr = out.reshape(od['shape'])
        rows = arr.shape[0]
        cols = arr.shape[1]
        by_rows.setdefault(rows, {})[cols] = arr

    all_scores = []
    all_bboxes = []
    all_kpss = []

    for stride in FEAT_STRIDES:
        height = INPUT_SIZE // stride
        width = INPUT_SIZE // stride
        rows = height * width * NUM_ANCHORS

        if rows not in by_rows:
            continue

        scores = by_rows[rows][1].reshape(-1)
        bbox_preds = by_rows[rows][4] * stride
        kps_preds = by_rows[rows][10] * stride

        anchor_centers = _get_anchor_centers(stride)

        pos_inds = np.where(scores >= score_thresh)[0]
        if pos_inds.size == 0:
            continue

        bboxes = distance2bbox(anchor_centers, bbox_preds)
        kpss = distance2kps(anchor_centers, kps_preds)
        kpss = kpss.reshape(kpss.shape[0], -1, 2)

        all_scores.append(scores[pos_inds])
        all_bboxes.append(bboxes[pos_inds])
        all_kpss.append(kpss[pos_inds])

    if len(all_scores) == 0:
        return None, None

    scores = np.concatenate(all_scores)
    bboxes = np.concatenate(all_bboxes)
    kpss = np.concatenate(all_kpss)

    # undo letterbox scale to map back to the original frame
    bboxes = bboxes / scale
    kpss = kpss / scale

    pre_det = np.hstack([bboxes, scores[:, None]]).astype(np.float32)
    keep = nms(pre_det, nms_thresh)

    det_out = pre_det[keep]
    kps_out = kpss[keep]

    return det_out, kps_out


# ============================================================
# ArcFace (w600k_r50) alignment + embedding
# ============================================================
REC_INPUT_SIZE = 112

# Standard 5-point ArcFace reference template for a 112x112 crop
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def norm_crop(img_bgr, kps, image_size=REC_INPUT_SIZE):
    """Align a face crop to the canonical 112x112 ArcFace pose using the
    5 facial keypoints, mirroring insightface.utils.face_align.norm_crop."""
    kps = kps.astype(np.float32)
    tform = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(img_bgr, tform, (image_size, image_size), borderValue=0.0)
    return aligned


def get_embedding(interpreter, frame_bgr, kps):
    """Aligns the face using kps, runs w600k_r50_full_quantized.tflite,
    dequantizes, and returns a unit-normalized 512-d embedding."""
    aligned_bgr = norm_crop(frame_bgr, kps, REC_INPUT_SIZE)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    raw_outputs, _ = run_interpreter(interpreter, aligned_rgb)
    embedding = raw_outputs[0].reshape(-1).astype(np.float32)

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding
