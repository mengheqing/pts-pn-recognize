import os
import re
import json
import math
import time

import numpy as np
import cv2
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

try:
    from pyzbar.pyzbar import decode as zbar_decode
    PYZBAR_OK = True
except Exception:
    PYZBAR_OK = False
    print("pyzbar 不可用，将仅使用 OpenCV QRCodeDetector。")


# =========================================================
# 配置
# =========================================================
CLS_QR = 0
CLS_PTS = 1

PARALLEL_TH = 10.0
DMAX_RATIO = 0.30
ALPHA = 0.72
BETA = 0.28
MAX_COST = 1.35

CONF_TH = 0.6
IMGSZ = 1536

CLASS_SIZES = {
    CLS_QR: (900, 900),
    CLS_PTS: (200, 600),
}

QR_MARGIN_RATIO = 0.22

TEMP_CROPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_crops")
TEMP_FILE_MAX_AGE = 3600

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "yolo-script-and-model", "train-yolo11n-obb", "weights", "best.pt"
)


# =========================================================
# 模型管理
# =========================================================
_yolo_model = None


def init_yolo_model(model_path=None):
    global _yolo_model
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if _yolo_model is None:
        _yolo_model = YOLO(model_path)
        os.makedirs(TEMP_CROPS_DIR, exist_ok=True)


def get_yolo_model():
    if _yolo_model is None:
        raise RuntimeError("YOLO 模型未初始化，请先调用 init_yolo_model()")
    return _yolo_model


# =========================================================
# 工具函数
# =========================================================
def norm180(angle_deg):
    a = angle_deg % 180.0
    if a < 0:
        a += 180.0
    return a


def parallel_diff_deg(a_deg, b_deg):
    a = norm180(a_deg)
    b = norm180(b_deg)
    d = abs(a - b)
    return min(d, 180.0 - d)


def long_edge_angle_from_rect(rect):
    (_, _), (w, h), angle = rect
    if w > h:
        theta = angle + 90.0
    else:
        theta = angle
    return norm180(theta)


def order_quad_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def expand_quad_from_center(quad, ratio=0.2):
    q = np.array(quad, dtype=np.float32)
    c = q.mean(axis=0, keepdims=True)
    q_exp = c + (q - c) * (1.0 + ratio)
    return q_exp.astype(np.float32)


def perspective_crop(img, quad_pts, out_size=(900, 900), margin_ratio=0.0):
    if quad_pts is None or len(quad_pts) != 4:
        return None

    q = order_quad_points(quad_pts)
    if margin_ratio > 1e-6:
        q = expand_quad_from_center(q, margin_ratio)

    h_img, w_img = img.shape[:2]
    q[:, 0] = np.clip(q[:, 0], -w_img, 2 * w_img)
    q[:, 1] = np.clip(q[:, 1], -h_img, 2 * h_img)

    ow, oh = out_size
    dst = np.array([
        [0, 0],
        [ow - 1, 0],
        [ow - 1, oh - 1],
        [0, oh - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(img, M, (ow, oh), flags=cv2.INTER_CUBIC)
    return warped


def crop_by_rect_rotated(img, rect):
    (cx, cy), (w, h), angle = rect

    if w > h:
        w, h = h, w
        angle += 90

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = rotated[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def sanitize_filename(s, max_len=80):
    s = s.strip()
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s if s else "EMPTY"


def decode_qr_opencv(img):
    detector = cv2.QRCodeDetector()
    txt, pts, _ = detector.detectAndDecode(img)
    if txt is not None and txt.strip() != "":
        return txt.strip()
    return ""


def decode_qr_pyzbar(img):
    if not PYZBAR_OK:
        return ""
    try:
        rs = zbar_decode(img)
        for r in rs:
            t = r.data.decode("utf-8", errors="ignore").strip()
            if t:
                return t
        return ""
    except Exception:
        return ""


def decode_qr_robust(qr_img):
    if qr_img is None or qr_img.size == 0:
        return ""

    attempts = []
    attempts.append(("orig", qr_img))

    gray = cv2.cvtColor(qr_img, cv2.COLOR_BGR2GRAY)
    attempts.append(("gray", gray))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    attempts.append(("clahe", gray_clahe))

    _, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    attempts.append(("otsu", bw_otsu))

    bw_adp = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
    attempts.append(("adp", bw_adp))

    scaled_attempts = []
    for name, im in attempts:
        up2 = cv2.resize(im, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        up3 = cv2.resize(im, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        scaled_attempts.append((name + "_x2", up2))
        scaled_attempts.append((name + "_x3", up3))

    attempts.extend(scaled_attempts)

    for _, im in attempts:
        im_bgr = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        txt = decode_qr_opencv(im_bgr)
        if txt:
            return txt
        txt2 = decode_qr_pyzbar(im_bgr)
        if txt2:
            return txt2

    return ""


def pair_qr_pts(qr_list, pts_list, parallel_th=PARALLEL_TH, dmax_ratio=DMAX_RATIO,
                alpha=ALPHA, beta=BETA, max_cost=MAX_COST):
    if len(qr_list) == 0 or len(pts_list) == 0:
        return []

    all_centers = np.array([q["center"] for q in qr_list] + [p["center"] for p in pts_list], dtype=np.float32)
    min_xy = all_centers.min(axis=0)
    max_xy = all_centers.max(axis=0)
    diag = np.linalg.norm(max_xy - min_xy) + 1e-6
    dmax = dmax_ratio * diag

    n, m = len(qr_list), len(pts_list)
    BIG = 1e6
    C = np.full((n, m), BIG, dtype=np.float32)

    for i, q in enumerate(qr_list):
        qx, qy = q["center"]
        for j, p in enumerate(pts_list):
            px, py = p["center"]
            dist = math.hypot(qx - px, qy - py)
            ad = parallel_diff_deg(q["theta"], p["theta"])

            if ad > parallel_th:
                continue
            if dist > dmax:
                continue

            d_norm = dist / dmax
            a_norm = ad / parallel_th
            C[i, j] = alpha * d_norm + beta * a_norm

    rows, cols = linear_sum_assignment(C)
    pairs = []
    for r, c in zip(rows, cols):
        cost = float(C[r, c])
        if cost >= BIG:
            continue
        if cost > max_cost:
            continue
        pairs.append((r, c, cost))
    return pairs


# =========================================================
# 主业务函数
# =========================================================
def yolo_recognize(image_path, temp_dir=None):
    if not os.path.isfile(image_path):
        return json.dumps({'success': False, 'error': f'图片文件不存在: {image_path}'}, ensure_ascii=False)

    if temp_dir is None:
        temp_dir = TEMP_CROPS_DIR
    os.makedirs(temp_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        return json.dumps({'success': False, 'error': '图片读取失败'}, ensure_ascii=False)

    model = get_yolo_model()
    results = model.predict(source=img, conf=CONF_TH, imgsz=IMGSZ, verbose=False)

    result = results[0]

    if result.obb is None or len(result.obb) == 0:
        return json.dumps({
            'success': True,
            'total_detections': 0,
            'qr_count': 0,
            'pts_count': 0,
            'pairs': [],
            'unpaired_qr': [],
            'unpaired_pts': [],
            'message': '未检测到任何目标'
        }, ensure_ascii=False)

    obb_infos = []
    for idx, obb in enumerate(result.obb):
        quad = obb.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(quad)
        (cx, cy), _, _ = rect
        theta = long_edge_angle_from_rect(rect)
        cls_id = int(obb.cls[0].item())
        conf = float(obb.conf[0].item())

        obb_infos.append({
            "idx": idx,
            "cls_id": cls_id,
            "conf": conf,
            "quad": quad,
            "rect": rect,
            "center": (cx, cy),
            "theta": theta
        })

    qr_list = [x for x in obb_infos if x["cls_id"] == CLS_QR]
    pts_list = [x for x in obb_infos if x["cls_id"] == CLS_PTS]

    if len(qr_list) == 0 or len(pts_list) == 0:
        return json.dumps({
            'success': True,
            'total_detections': len(obb_infos),
            'qr_count': len(qr_list),
            'pts_count': len(pts_list),
            'pairs': [],
            'unpaired_qr': list(range(len(qr_list))),
            'unpaired_pts': list(range(len(pts_list))),
            'message': 'QR或PTS缺失，无法匹配'
        }, ensure_ascii=False)

    pairs = pair_qr_pts(qr_list, pts_list)

    paired_qr_indices = set()
    paired_pts_indices = set()
    pair_results = []
    now_ts = time.strftime("%Y%m%d%H%M%S")

    for k, (qi, pi, cost) in enumerate(pairs, start=1):
        q = qr_list[qi]
        p = pts_list[pi]
        paired_qr_indices.add(qi)
        paired_pts_indices.add(pi)

        # QR 透视裁剪 + 外扩
        qr_w, qr_h = CLASS_SIZES.get(CLS_QR, (900, 900))
        qr_crop = perspective_crop(img, q["quad"], out_size=(qr_w, qr_h), margin_ratio=QR_MARGIN_RATIO)
        if qr_crop is None or qr_crop.size == 0:
            continue

        # PTS 旋转裁剪
        pts_crop = crop_by_rect_rotated(img, p["rect"])
        if pts_crop is None:
            continue

        pts_w, pts_h = CLASS_SIZES.get(CLS_PTS, (200, 600))
        pts_final = cv2.resize(pts_crop, (pts_w, pts_h), interpolation=cv2.INTER_CUBIC)

        # QR 解码
        qr_text = decode_qr_robust(qr_crop)
        qr_decode_success = qr_text != ""
        if not qr_decode_success:
            qr_text = "DECODE_FAIL"

        safe_text = sanitize_filename(qr_text, max_len=60)

        # 保存 PTS 裁剪图到临时目录
        pts_filename = f"PTS_pair{k}_{safe_text}_{now_ts}.jpg"
        pts_save_path = os.path.join(temp_dir, pts_filename)
        cv2.imwrite(pts_save_path, pts_final)

        pair_results.append({
            'pair_id': k,
            'qr_text': qr_text,
            'qr_decode_success': qr_decode_success,
            'qr_confidence': round(q["conf"], 4),
            'pts_confidence': round(p["conf"], 4),
            'match_cost': round(cost, 4),
            'pts_image_path': pts_save_path,
            'pts_image_url': f"/api/crops/{pts_filename}",
            'qr_center': [round(q["center"][0], 2), round(q["center"][1], 2)],
            'pts_center': [round(p["center"][0], 2), round(p["center"][1], 2)]
        })

    unpaired_qr = [i for i in range(len(qr_list)) if i not in paired_qr_indices]
    unpaired_pts = [i for i in range(len(pts_list)) if i not in paired_pts_indices]

    message = ""
    if len(pairs) == 0:
        message = "未得到有效匹配"
    elif len(pair_results) == 0:
        message = "配对存在但裁剪失败"

    return json.dumps({
        'success': True,
        'total_detections': len(obb_infos),
        'qr_count': len(qr_list),
        'pts_count': len(pts_list),
        'pairs': pair_results,
        'unpaired_qr': unpaired_qr,
        'unpaired_pts': unpaired_pts,
        'message': message
    }, ensure_ascii=False)


# =========================================================
# 临时文件清理
# =========================================================
def cleanup_temp_crops(temp_dir=None):
    if temp_dir is None:
        temp_dir = TEMP_CROPS_DIR
    if not os.path.exists(temp_dir):
        return
    now = time.time()
    for f in os.listdir(temp_dir):
        fp = os.path.join(temp_dir, f)
        if os.path.isfile(fp) and (now - os.path.getmtime(fp)) > TEMP_FILE_MAX_AGE:
            try:
                os.unlink(fp)
            except OSError:
                pass
