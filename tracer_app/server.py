"""
Tracer — Invisible Watermark Engine
Backend API for DigiKit pitch

Endpoints:
  POST /api/embed   — embed watermark into uploaded image
  POST /api/decode  — decode watermark from uploaded image
  GET  /            — serve the frontend
"""

import os
import sys
import io
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template

import cv2
import numpy as np

from imwatermark import WatermarkEncoder, WatermarkDecoder

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload

# ─── Watermark Format ─────────────────────────────────────────────────────────
# Fixed 128-byte payload (1024 bits) embedded using dwtDct
#
# Layout (pipe-delimited, fixed-width fields, null-padded to 128 bytes):
#   TRC1|<company:20>|<client:20>|<campaign:20>|<date:8>|<time:6>|<note:31>
#   Total: 4+1+20+1+20+1+20+1+8+1+6+1+31 = 115 bytes + 13 null padding = 128
#
PAYLOAD_BYTES = 128
MAGIC = b'TRC1'


def _pad(s: str, length: int) -> bytes:
    b = s.encode('utf-8')[:length]
    return b.ljust(length, b' ')


def _strip(b: bytes) -> str:
    return b.decode('utf-8', errors='replace').rstrip(' \x00')


def build_payload(company: str, client: str, campaign: str,
                  date: str, time: str, note: str) -> bytes:
    """Pack metadata into exactly PAYLOAD_BYTES bytes."""
    payload = (
        MAGIC
        + b'|' + _pad(company, 20)
        + b'|' + _pad(client, 20)
        + b'|' + _pad(campaign, 20)
        + b'|' + _pad(date, 8)
        + b'|' + _pad(time, 6)
        + b'|' + _pad(note, 31)
    )
    # Should be 115 bytes; pad to 128
    return payload.ljust(PAYLOAD_BYTES, b'\x00')


def parse_payload(raw: bytes) -> dict:
    """Parse a PAYLOAD_BYTES payload back into a metadata dict."""
    if len(raw) < PAYLOAD_BYTES:
        raw = raw.ljust(PAYLOAD_BYTES, b'\x00')

    if not raw.startswith(MAGIC):
        return {'error': 'No Tracer watermark detected in this image.'}

    # Expected layout after magic:
    # |company(20)|client(20)|campaign(20)|date(8)|time(6)|note(31)
    #  ^0         ^21        ^42          ^63     ^72     ^79
    try:
        rest = raw[5:]  # skip "TRC1|"
        company  = _strip(rest[0:20])
        client   = _strip(rest[21:41])
        campaign = _strip(rest[42:62])
        date     = _strip(rest[63:71])
        time_str = _strip(rest[72:78])
        note     = _strip(rest[79:110])
        return {
            'company':  company,
            'client':   client,
            'campaign': campaign,
            'date':     date,
            'time':     time_str,
            'note':     note,
        }
    except Exception as e:
        return {'error': f'Payload parse error: {str(e)}'}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_image_from_request(file_field: str):
    """Read an uploaded image into a BGR numpy array."""
    f = request.files.get(file_field)
    if f is None:
        return None, 'No image file provided'
    data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 'Could not decode image'
    h, w = img.shape[:2]
    if h * w < 256 * 256:
        return None, f'Image too small ({w}x{h}). Must be at least 256×256 pixels.'
    return img, None


def image_to_png_bytes(img) -> bytes:
    """Encode a BGR numpy array to PNG bytes."""
    success, buf = cv2.imencode('.png', img)
    if not success:
        raise RuntimeError('PNG encode failed')
    return buf.tobytes()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/embed', methods=['POST'])
def embed():
    img, err = load_image_from_request('image')
    if err:
        return jsonify({'error': err}), 400

    company  = request.form.get('company', 'DigiKit')[:20]
    client   = request.form.get('client', '')[:20]
    campaign = request.form.get('campaign', '')[:20]
    date     = request.form.get('date', datetime.now().strftime('%Y%m%d'))[:8]
    time_val = request.form.get('time', datetime.now().strftime('%H%M%S'))[:6]
    note     = request.form.get('note', '')[:31]

    payload = build_payload(company, client, campaign, date, time_val, note)

    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', payload)

    try:
        encoded_img = encoder.encode(img, 'dwtDctSvd')
    except Exception as e:
        return jsonify({'error': f'Encoding failed: {str(e)}'}), 500

    png_bytes = image_to_png_bytes(encoded_img)
    return send_file(
        io.BytesIO(png_bytes),
        mimetype='image/png',
        as_attachment=True,
        download_name='tracer_watermarked.png'
    )


@app.route('/api/decode', methods=['POST'])
def decode():
    img, err = load_image_from_request('image')
    if err:
        return jsonify({'error': err}), 400

    wm_bits = PAYLOAD_BYTES * 8  # 1024 bits

    decoder = WatermarkDecoder('bytes', wm_bits)
    try:
        raw = decoder.decode(img, 'dwtDctSvd')
    except Exception as e:
        return jsonify({'error': f'Decoding failed: {str(e)}'}), 500

    metadata = parse_payload(raw)
    return jsonify(metadata)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'service': 'Tracer by DigiKit'})


if __name__ == '__main__':
    app.run(debug=True, port=5050)
