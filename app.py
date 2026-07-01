import os
import json
import base64
import tempfile
import threading
import time

from flask import Flask, request, jsonify, send_from_directory
from recognize import recognize
from yolo_recognize import init_yolo_model, yolo_recognize, cleanup_temp_crops, TEMP_CROPS_DIR

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': '缺少 image 字段'}), 400

    image_b64 = data['image']

    if ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({'success': False, 'error': 'base64 解码失败'}), 400

    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    try:
        tmp.write(image_bytes)
        tmp.close()
        result_json = recognize(tmp.name)
    finally:
        os.unlink(tmp.name)

    return jsonify(json.loads(result_json))


@app.route('/api/yolo-recognize', methods=['POST'])
def api_yolo_recognize():
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': '缺少 image 字段'}), 400

    image_b64 = data['image']

    if ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({'success': False, 'error': 'base64 解码失败'}), 400

    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    try:
        tmp.write(image_bytes)
        tmp.close()
        result_json = yolo_recognize(tmp.name)
    finally:
        os.unlink(tmp.name)

    return jsonify(json.loads(result_json))


@app.route('/api/crops/<path:filename>')
def serve_crop(filename):
    return send_from_directory(TEMP_CROPS_DIR, filename)


def _cleanup_loop():
    while True:
        time.sleep(600)
        cleanup_temp_crops()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))

    yolo_model_path = os.path.join(
        base_dir, 'yolo-script-and-model', 'train-yolo11n-obb', 'weights', 'best.pt'
    )
    init_yolo_model(yolo_model_path)

    cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
    cleanup_thread.start()

    ssl_context = (
        os.path.join(base_dir, 'cert.pem'),
        os.path.join(base_dir, 'key.pem'),
    )
    app.run(host='0.0.0.0', port=10086, debug=False, ssl_context=ssl_context)
