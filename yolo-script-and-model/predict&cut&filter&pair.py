import os
import re
import time
import math
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# pyzbar 可选兜底
try:
    from pyzbar.pyzbar import decode as zbar_decode
    PYZBAR_OK = True
except Exception:
    PYZBAR_OK = False
    print("⚠️ pyzbar 不可用，将仅使用 OpenCV QRCodeDetector。")


# =========================================================
# 0) 配置区
# =========================================================
MODEL_PATH = r"C:\machine_learning\ultralytics\ROItraining-obb-case\runs\obb\ultralytics-8.4-results\train-yolo11n-obb\weights\best.pt"
SOURCE_PATH = r"C:\machine_learning\ultralytics\ROItraining-obb-case\data\images\test"

# 类别ID（与训练一致）
CLS_QR = 0
CLS_PTS = 1

# 角度/匹配参数
PARALLEL_TH = 10.0
DMAX_RATIO = 0.30
ALPHA = 0.72
BETA = 0.28
MAX_COST = 1.35

# 推理参数
CONF_TH = 0.6
IMGSZ = 1536

# 裁剪输出尺寸
CLASS_SIZES = {
    CLS_QR: (900, 900),
    CLS_PTS: (200, 600),
}

# QR ROI 外扩比例
QR_MARGIN_RATIO = 0.22

# 是否保存调试图
SAVE_DEBUG = False


# =========================================================
# 1) 工具函数
# =========================================================
def norm180(angle_deg: float) -> float:
    a = angle_deg % 180.0
    if a < 0:
        a += 180.0
    return a


def parallel_diff_deg(a_deg: float, b_deg: float) -> float:
    a = norm180(a_deg)
    b = norm180(b_deg)
    d = abs(a - b)
    return min(d, 180.0 - d)


def long_edge_angle_from_rect(rect) -> float:
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


def sanitize_filename(s: str, max_len=80):
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


def decode_qr_robust(qr_img, debug_dir=None, debug_prefix=""):
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

    for i, (name, im) in enumerate(attempts):
        im_bgr = im if len(im.shape) == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        txt = decode_qr_opencv(im_bgr)
        if txt:
            return txt

        txt2 = decode_qr_pyzbar(im_bgr)
        if txt2:
            return txt2

        if SAVE_DEBUG and debug_dir is not None:
            os.makedirs(debug_dir, exist_ok=True)
            fp = os.path.join(debug_dir, f"{debug_prefix}_{i:02d}_{name}.png")
            cv2.imwrite(fp, im)

    return ""


def pair_qr_pts(qr_list, pts_list, parallel_th=10.0, dmax_ratio=0.3, alpha=0.72, beta=0.28, max_cost=1.35):
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
# 2) 运行推理
# =========================================================
model = YOLO(MODEL_PATH)

results = model.predict(
    source=SOURCE_PATH,
    conf=CONF_TH,
    imgsz=IMGSZ,
    show=False,
    save=True,
    save_txt=True,
    save_conf=True
)

predict_save_dir = results[0].save_dir
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
roi_save_path = os.path.join(predict_save_dir, f"ROI_results_{run_timestamp}")
os.makedirs(roi_save_path, exist_ok=True)

debug_dir = os.path.join(roi_save_path, "debug_decode")
if SAVE_DEBUG:
    os.makedirs(debug_dir, exist_ok=True)

mapping_csv_path = os.path.join(roi_save_path, "qr_pts_mapping.csv")
csv_lines = []
csv_lines.append("image,pair_id,qr_obb_idx,pts_obb_idx,match_cost,qr_text,qr_file,pts_file\n")


# =========================================================
# 3) 主循环：配对 + 裁剪 + 解码 + 输出
# =========================================================
for result in results:
    img = result.orig_img.copy()
    img_name = Path(result.path).stem

    if result.obb is None or len(result.obb) == 0:
        print(f"⚠️ {img_name}: 未检测到目标，跳过")
        continue

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
        print(f"⚠️ {img_name}: QR或PTS缺失，无法匹配（QR={len(qr_list)}, PTS={len(pts_list)}）")
        continue

    pairs = pair_qr_pts(
        qr_list, pts_list,
        parallel_th=PARALLEL_TH,
        dmax_ratio=DMAX_RATIO,
        alpha=ALPHA,
        beta=BETA,
        max_cost=MAX_COST
    )

    if len(pairs) == 0:
        print(f"⚠️ {img_name}: 未得到有效匹配")
        continue

    print(f"\n📌 {img_name}: 匹配到 {len(pairs)} 对")

    for k, (qi, pi, cost) in enumerate(pairs, start=1):
        q = qr_list[qi]
        p = pts_list[pi]

        # QR：透视裁剪 + 外扩
        qr_w, qr_h = CLASS_SIZES.get(CLS_QR, (900, 900))
        qr_crop = perspective_crop(
            img,
            q["quad"],
            out_size=(qr_w, qr_h),
            margin_ratio=QR_MARGIN_RATIO
        )
        if qr_crop is None or qr_crop.size == 0:
            print(f"⚠️ {img_name}: QR idx={q['idx']} 透视裁剪失败")
            continue

        # PTS：旋转裁剪
        pts_crop = crop_by_rect_rotated(img, p["rect"])
        if pts_crop is None:
            print(f"⚠️ {img_name}: PTS idx={p['idx']} 裁剪失败")
            continue

        pts_w, pts_h = CLASS_SIZES.get(CLS_PTS, (200, 600))
        pts_final = cv2.resize(pts_crop, (pts_w, pts_h), interpolation=cv2.INTER_CUBIC)

        # 鲁棒解码
        qr_text = decode_qr_robust(
            qr_crop,
            debug_dir=debug_dir if SAVE_DEBUG else None,
            debug_prefix=f"{img_name}_pair{k}_qrIdx{q['idx']}"
        )
        if not qr_text:
            qr_text = "DECODE_FAIL"

        safe_text = sanitize_filename(qr_text, max_len=60)
        now_ts = time.strftime("%Y%m%d%H%M%S")  # 当前时间戳（到秒）

        # 文件命名
        # QR 文件名（保留较详细信息）
        qr_file = f"{img_name}_pair{k}_QR_idx{q['idx']}_{safe_text}.jpg"

        # PTS 文件名（按你的要求）
        # 原图片名_PTS_QRcode_扫描出来二维码的信息_当前时间戳
        pts_file = f"{img_name}_PTS_QRcode_{safe_text}_{now_ts}.jpg"

        qr_save = os.path.join(roi_save_path, qr_file)
        pts_save = os.path.join(roi_save_path, pts_file)

        cv2.imwrite(qr_save, qr_crop)
        cv2.imwrite(pts_save, pts_final)

        print(
            f"✅ Pair{k}: QR(idx={q['idx']}) <-> PTS(idx={p['idx']}), "
            f"cost={cost:.3f}, qr_text={qr_text}"
        )

        csv_lines.append(
            f"{img_name},{k},{q['idx']},{p['idx']},{cost:.6f},"
            f"{qr_text},{qr_file},{pts_file}\n"
        )

# 写CSV
with open(mapping_csv_path, "w", encoding="utf-8") as f:
    f.writelines(csv_lines)

print(f"\n========== 所有 ROI 已保存至：{roi_save_path} ==========")
print(f"========== 映射CSV已输出：{mapping_csv_path} ==========")