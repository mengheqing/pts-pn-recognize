import os
import base64
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from recognize import recognize

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

    import json
    return jsonify(json.loads(result_json))


if __name__ == '__main__':
    # 使用10086端口
    app.run(host='0.0.0.0', port=10086, debug=False)
