from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)

MODEL_PATH = 'models/serpensgate_yolov8.pt'  # pretrained
mtx, dist = np.eye(3), np.zeros((1,5))

history = {'fish': defaultdict(list), 'plant': defaultdict(list)}

def calc_growth_rate(current, hist_key, id_key):
    # 이전 로직

def calc_variability(df_col):
    return df_col.std() / df_col.mean() if df_col.mean() > 0 else 0.0

@app.route('/upload_event', methods=['POST'])
def handle_event():
    data = request.json
    frame_data = np.frombuffer(bytes.fromhex(data['frame_hex']), np.uint8)
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    frame = cv2.undistort(frame, mtx, dist)
    type_key = data['type']

    model = YOLO(MODEL_PATH)
    results = model.track(source=frame, persist=True)[0]
    boxes = results.boxes
    records = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        id_key = int(box.id[0]) if box.id else 0

        if type_key == 'fish':
            length_cm = np.linalg.norm([x2-x1, y2-y1]) * 0.1
            growth_rate = calc_growth_rate(length_cm, 'fish', id_key)
            records.append({'id': id_key, 'length_cm': length_cm, 'growth_rate': growth_rate})

        else:
            height_cm = (y2 - y1) * 0.1
            hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            avg_hue = np.mean(hsv[:,:,0])
            growth_rate = calc_growth_rate(height_cm, 'plant', id_key)
            records.append({'id': id_key, 'height_cm': height_cm, 'growth_rate': growth_rate})

    df_new = pd.DataFrame(records)
    df_path = f"server_{type_key}_records.xlsx"
    df = pd.read_excel(df_path) if os.path.exists(df_path) else df_new
    df = pd.concat([df, df_new])
    df.to_excel(df_path, index=False)

    variability = calc_variability(df['length_cm' if type_key == 'fish' else 'height_cm'])
    print(f"변동성: {variability:.2f}")

    return jsonify({'status': 'processed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
