import os
import json
import base64
import tempfile
import threading
import time
import urllib.request
import urllib.parse
import ssl

from flask import Flask, request, jsonify, send_from_directory, redirect
from recognize import recognize
from yolo_recognize import init_yolo_model, yolo_recognize, cleanup_temp_crops, TEMP_CROPS_DIR

app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/')
def index():
    return redirect('/cn/')


@app.route('/cn/')
def index_cn():
    return send_from_directory('static', 'index.html')


@app.route('/en/')
def index_en():
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


GEO_APPCODE = '64269fbca2f643c5bd390e45a66dd075'
GEO_API_URL = 'https://dizhi.market.alicloudapi.com/location_address'

_geo_ssl_context = None


def _get_geo_ssl_context():
    global _geo_ssl_context
    if _geo_ssl_context is not None:
        return _geo_ssl_context
    try:
        import certifi
        _geo_ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        _geo_ssl_context = ssl.create_default_context()
    return _geo_ssl_context


@app.route('/api/reverse-geocode')
def api_reverse_geocode():
    lat = request.args.get('lat', '').strip()
    lng = request.args.get('lng', '').strip()
    if not lat or not lng:
        return jsonify({'success': False, 'error': '缺少 lat 或 lng 参数'}), 400

    qs = urllib.parse.urlencode({'lng': lng, 'lat': lat, 'from': '1'})
    url = GEO_API_URL + '?' + qs

    req = urllib.request.Request(url, headers={'Authorization': 'APPCODE ' + GEO_APPCODE})

    try:
        with urllib.request.urlopen(req, timeout=10, context=_get_geo_ssl_context()) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return jsonify({'success': False, 'error': '逆地理编码请求失败: ' + str(e)})

    body = data.get('showapi_res_body', {})
    if body.get('ret_code') != 0:
        return jsonify({'success': False, 'error': body.get('msg', '查询失败')})

    addr = body.get('addressComponent', {})
    province = addr.get('province', '')
    city = addr.get('city', '')
    address = (province + city).strip() or 'Unknown'

    return jsonify({'success': True, 'address': address})


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
