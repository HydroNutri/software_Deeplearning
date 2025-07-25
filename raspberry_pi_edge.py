# raspberry_pi_edge.py - 라즈베리파이 최적화 엣지 코드
import cv2
import numpy as np
import json
import time
import threading
import requests
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from collections import deque, defaultdict
import psutil
import gc
from datetime import datetime
import os

class RaspberryPiEdgeMonitor:
    def __init__(self):
        # 하드웨어 최적화 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU만 사용
        
        # 경량화된 모델 로드
        self.model = YOLO('yolov5n.pt')  # nano 버전 사용 (가장 빠름)
        self.model.to('cpu')
        
        # 스트림 설정 (저해상도)
        self.rtsp_url = "rtsp://192.168.1.100:554/stream2"  # 서브 스트림 사용
        self.target_fps = 10  # 10fps로 제한
        self.frame_skip = 2   # 2프레임마다 처리
        
        # 메모리 관리
        self.max_history = 20  # 히스토리 제한
        self.detection_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.frame_buffer = deque(maxlen=5)
        
        # 변화량 감지 (더 관대한 임계값)
        self.change_threshold = 0.20  # 20% 변화 시 이벤트
        self.previous_detections = {}
        self.last_server_send = 0
        self.server_send_interval = 5.0  # 5초마다 최대 1회 전송
        
        # MQTT 설정 (경량화)
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = ""  # 서버 IP
        self.mqtt_port = 1883
        self.setup_mqtt()
        
        # 서버 연동
        self.server_url = ""
        
        # 성능 모니터링
        self.performance_stats = {
            'fps': 0,
            'cpu_usage': 0,
            'memory_usage': 0,
            'processing_time': 0
        }
        
        # 캘리브레이션 (간단한 고정값)
        self.pixel_to_cm_ratio = 0.08  # 저해상도에 맞게 조정
        
    def setup_mqtt(self):
        """경량화된 MQTT 설정"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"MQTT 연결 성공")
            else:
                print(f"MQTT 연결 실패: {rc}")
                
        self.mqtt_client.on_connect = on_connect
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"MQTT 연결 오류: {e}")
            
    def optimize_frame(self, frame):
        """프레임 최적화 (해상도 감소, 전처리)"""
        # 해상도 조정 (처리 속도 향상)
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
        return frame
        
    def lightweight_detection(self, frame):
        """경량화된 객체 탐지"""
        start_time = time.time()
        
        # YOLO 추론 (conf 임계값 높임)
        results = self.model(frame, conf=0.6, verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                
                # 간단한 크기 계산
                length_pixels = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                length_cm = length_pixels * self.pixel_to_cm_ratio
                
                # 기본 클래스만 처리 (COCO 기준)
                if class_id in [0, 14, 15, 16]:  # person, bird, cat, dog (물고기 대신)
                    obj_type = 'fish'
                    weight_g = length_cm ** 2 * 0.8  # 간단한 무게 추정
                    
                    detections.append({
                        'type': obj_type,
                        'bbox': bbox.tolist(),
                        'length': length_cm,
                        'weight': weight_g,
                        'confidence': confidence
                    })
                    
                elif class_id in [56, 57, 58]:  # plant-like objects
                    obj_type = 'plant'
                    height_cm = (bbox[3] - bbox[1]) * self.pixel_to_cm_ratio
                    
                    detections.append({
                        'type': obj_type,
                        'bbox': bbox.tolist(),
                        'height': height_cm,
                        'confidence': confidence
                    })
        
        processing_time = time.time() - start_time
        self.performance_stats['processing_time'] = processing_time
        
        return detections
        
    def detect_significant_change(self, detections):
        """변화량 기반 이벤트 감지 (최적화)"""
        significant_events = []
        current_objects = {}
        
        # 현재 탐지된 객체들 정리
        for i, detection in enumerate(detections):
            obj_key = f"{detection['type']}_{i}"
            current_objects[obj_key] = detection
            
        # 변화량 체크
        for obj_key, current in current_objects.items():
            if obj_key in self.previous_detections:
                prev = self.previous_detections[obj_key]
                
                if current['type'] == 'fish':
                    length_change = abs(current['length'] - prev['length']) / max(prev['length'], 1)
                    if length_change > self.change_threshold:
                        significant_events.append({
                            'type': 'fish_growth',
                            'id': obj_key,
                            'change': length_change,
                            'data': current
                        })
                        
                elif current['type'] == 'plant':
                    height_change = abs(current['height'] - prev['height']) / max(prev['height'], 1)
                    if height_change > self.change_threshold:
                        significant_events.append({
                            'type': 'plant_growth',
                            'id': obj_key,
                            'change': height_change,
                            'data': current
                        })
                        
        # 새로운 객체
        for obj_key, current in current_objects.items():
            if obj_key not in self.previous_detections:
                significant_events.append({
                    'type': 'new_object',
                    'id': obj_key,
                    'data': current
                })
                
        # 현재 상태 업데이트
        self.previous_detections = current_objects.copy()
        
        return significant_events
        
    def send_mqtt_event(self, event):
        """MQTT 이벤트 전송 (경량화)"""
        try:
            topic = f"aquaponics/edge/events"
            payload = {
                'timestamp': datetime.now().isoformat(),
                'device': 'raspberry_pi',
                'event': event
            }
            
            self.mqtt_client.publish(topic, json.dumps(payload), qos=0)  # QoS 0으로 성능 우선
            
        except Exception as e:
            print(f"MQTT 전송 실패: {e}")
            
    def send_to_server(self, frame, events):
        """서버 전송 (제한적)"""
        current_time = time.time()
        
        # 전송 빈도 제한
        if current_time - self.last_server_send < self.server_send_interval:
            return
            
        try:
            # 프레임 압축
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            
            files = {'frame': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
            data = {'events': json.dumps(events)}
            
            # 짧은 타임아웃
            response = requests.post(f"{self.server_url}/analyze", 
                                   files=files, data=data, timeout=3)
            
            if response.status_code == 200:
                self.last_server_send = current_time
                print("서버 전송 성공")
                
        except Exception as e:
            print(f"서버 전송 실패: {e}")
            
    def monitor_performance(self):
        """성능 모니터링"""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.performance_stats.update({
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent
        })
        
        # 성능 이슈 감지
        if cpu_percent > 80:
            print(f"⚠️  CPU 사용률 높음: {cpu_percent}%")
            self.target_fps = max(5, self.target_fps - 1)  # FPS 조절
            
        if memory_percent > 85:
            print(f"⚠️  메모리 사용률 높음: {memory_percent}%")
            gc.collect()  # 가비지 컬렉션
            
        # MQTT로 성능 상태 전송
        perf_topic = "aquaponics/edge/performance"
        self.mqtt_client.publish(perf_topic, json.dumps(self.performance_stats))
        
    def run(self):
        """메인 실행 루프 (최적화)"""
        cap = cv2.VideoCapture(self.rtsp_url)
        
        # 카메라 설정 최적화
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        frame_count = 0
        fps_counter = 0
        start_time = time.time()
        
        print("라즈베리파이 엣지 모니터링 시작")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("프레임 읽기 실패")
                    break
                    
                frame_count += 1
                
                # 프레임 스키핑
                if frame_count % self.frame_skip != 0:
                    continue
                    
                # 프레임 최적화
                frame = self.optimize_frame(frame)
                
                # 객체 탐지
                detections = self.lightweight_detection(frame)
                
                # 변화량 감지
                events = self.detect_significant_change(detections)
                
                # 이벤트가 있을 때만 처리
                if events:
                    print(f"{len(events)}개 이벤트 감지")
                    
                    # MQTT 전송
                    for event in events:
                        self.send_mqtt_event(event)
                        
                    # 서버 전송 (중요한 이벤트만)
                    critical_events = [e for e in events if e['type'] in ['new_object', 'fish_growth']]
                    if critical_events:
                        self.send_to_server(frame, critical_events)
                
                # FPS 계산
                fps_counter += 1
                if fps_counter >= 30:
                    elapsed = time.time() - start_time
                    self.performance_stats['fps'] = fps_counter / elapsed
                    fps_counter = 0
                    start_time = time.time()
                    
                    # 성능 모니터링
                    self.monitor_performance()
                
                # FPS 제어
                time.sleep(1.0 / self.target_fps)
                
                # 메모리 정리 (주기적)
                if frame_count % 100 == 0:
                    gc.collect()
                    
        except KeyboardInterrupt:
            print("시스템 종료")
            
        finally:
            cap.release()
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

if __name__ == "__main__":
    # 라즈베리파이 최적화 설정
    os.nice(10)  # 프로세스 우선순위 낮춤
    
    monitor = RaspberryPiEdgeMonitor()
    monitor.run()
