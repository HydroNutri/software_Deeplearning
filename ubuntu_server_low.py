# ubuntu_server_low.py - 저사양 우분투 서버용 24시간 구동 코드
import cv2
import numpy as np
import json
import sqlite3
import time
import threading
import queue
from flask import Flask, request, jsonify
from ultralytics import YOLO
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta
import pandas as pd
import smtplib
import gc
import psutil
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aquaponics_server.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)

class LowSpecServerSystem:
    def __init__(self):
        # CPU 전용 모델 (경량화)
        self.model = None  # 필요시에만 로드
        self.model_loaded = False
        self.last_model_use = 0
        self.model_timeout = 300  # 5분 후 모델 언로드
        
        # 데이터베이스 최적화
        self.db_path = 'aquaponics_lite.db'
        self.init_database()
        
        # MQTT 브로커 설정
        self.mqtt_client = mqtt.Client()
        self.setup_mqtt_broker()
        
        # 배치 처리 큐
        self.analysis_queue = queue.Queue(maxsize=50)
        self.alert_queue = queue.Queue(maxsize=20)
        
        # 메모리 관리
        self.max_cache_size = 100
        self.detection_cache = {}
        self.performance_monitor = {
            'cpu_threshold': 70,
            'memory_threshold': 80
        }
        
        # 백그라운드 스레드들
        self.start_background_workers()
        
        # 이메일 설정 (간단한 SMTP)
        self.smtp_config = {
            'server': 'smtp.gmail.com',
            'port': 587,
            'email': 'your_email@gmail.com',
            'password': 'your_app_password'
        }
        
    def init_database(self):
        """경량화된 데이터베이스 초기화"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # 테이블 생성 (최소한의 인덱스)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                device_id TEXT,
                event_type TEXT,
                object_id TEXT,
                data TEXT,
                processed INTEGER DEFAULT 0
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date DATE PRIMARY KEY,
                fish_count INTEGER DEFAULT 0,
                plant_count INTEGER DEFAULT 0,
                alerts_count INTEGER DEFAULT 0,
                avg_growth_rate REAL DEFAULT 0,
                system_health REAL DEFAULT 1.0
            )
        ''')
        
        # 필수 인덱스만 생성
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_processed ON events(processed)')
        
        self.conn.commit()
        
    def setup_mqtt_broker(self):
        """MQTT 브로커 설정"""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logging.info("MQTT 브로커 연결 성공")
                client.subscribe("aquaponics/edge/+")
                client.subscribe("aquaponics/+/+/+")
            else:
                logging.error(f"MQTT 연결 실패: {rc}")
                
        def on_message(client, userdata, msg):
            try:
                topic = msg.topic
                payload = json.loads(msg.payload.decode())
                
                # 큐에 메시지 추가 (논블로킹)
                if not self.analysis_queue.full():
                    self.analysis_queue.put({
                        'topic': topic,
                        'payload': payload,
                        'timestamp': datetime.now()
                    })
                    
            except Exception as e:
                logging.error(f"MQTT 메시지 처리 오류: {e}")
                
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
        
        try:
            self.mqtt_client.connect("localhost", 1883, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            logging.error(f"MQTT 연결 오류: {e}")
            
    def load_model_if_needed(self):
        """필요시에만 모델 로드 (메모리 절약)"""
        current_time = time.time()
        
        if not self.model_loaded:
            try:
                self.model = YOLO('yolov8n.pt')  # nano 버전 (가장 가벼움)
                self.model.to('cpu')
                self.model_loaded = True
                logging.info("모델 로드 완료")
            except Exception as e:
                logging.error(f"모델 로드 실패: {e}")
                return False
                
        self.last_model_use = current_time
        return True
        
    def unload_model_if_idle(self):
        """모델 유휴 시 언로드"""
        if (self.model_loaded and 
            time.time() - self.last_model_use > self.model_timeout):
            
            del self.model
            self.model = None
            self.model_loaded = False
            gc.collect()
            logging.info("모델 언로드 (메모리 절약)")
            
    def start_background_workers(self):
        """백그라운드 워커 스레드 시작"""
        # 분석 워커
        analysis_worker = threading.Thread(target=self.analysis_worker, daemon=True)
        analysis_worker.start()
        
        # 알림 워커
        alert_worker = threading.Thread(target=self.alert_worker, daemon=True)
        alert_worker.start()
        
        # 정리 워커
        cleanup_worker = threading.Thread(target=self.cleanup_worker, daemon=True)
        cleanup_worker.start()
        
        # 성능 모니터링 워커
        perf_worker = threading.Thread(target=self.performance_worker, daemon=True)
        perf_worker.start()
        
    def analysis_worker(self):
        """배치 분석 워커"""
        batch_size = 5
        batch = []
        
        while True:
            try:
                # 큐에서 메시지 수집
                try:
                    message = self.analysis_queue.get(timeout=5)
                    batch.append(message)
                except queue.Empty:
                    pass
                    
                # 배치 처리
                if len(batch) >= batch_size or (batch and time.time() % 30 < 1):
                    self.process_batch(batch)
                    batch = []
                    
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"분석 워커 오류: {e}")
                time.sleep(5)
                
    def process_batch(self, batch):
        """배치 데이터 처리"""
        try:
            for message in batch:
                topic = message['topic']
                payload = message['payload']
                
                # DB에 이벤트 저장
                if 'events' in topic:
                    event_data = payload.get('event', {})
                    
                    self.conn.execute('''
                        INSERT INTO events (device_id, event_type, object_id, data)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        payload.get('device', 'unknown'),
                        event_data.get('type', 'unknown'),
                        event_data.get('id', ''),
                        json.dumps(event_data)
                    ))
                    
                    # 알림 필요 여부 체크
                    if self.needs_alert(event_data):
                        if not self.alert_queue.full():
                            self.alert_queue.put(event_data)
                            
            self.conn.commit()
            logging.info(f"배치 처리 완료: {len(batch)}개 메시지")
            
        except Exception as e:
            logging.error(f"배치 처리 오류: {e}")
            
    def needs_alert(self, event_data):
        """알림 필요 여부 판단"""
        event_type = event_data.get('type', '')
        
        # 새 객체 감지
        if event_type == 'new_object':
            return True
            
        # 급격한 성장 변화
        if 'growth' in event_type:
            change = event_data.get('change', 0)
            if change > 0.5:  # 50% 이상 변화
                return True
                
        return False
        
    def alert_worker(self):
        """알림 처리 워커"""
        while True:
            try:
                alert_data = self.alert_queue.get(timeout=10)
                self.send_simple_alert(alert_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"알림 워커 오류: {e}")
                time.sleep(5)
                
    def send_simple_alert(self, alert_data):
        """간단한 알림 전송"""
        try:
            subject = f"아쿠아포닉스 알림: {alert_data.get('type', '이벤트')}"
            message = f"""
            이벤트 타입: {alert_data.get('type')}
            객체 ID: {alert_data.get('id')}
            변화량: {alert_data.get('change', 'N/A')}
            시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # 이메일 전송 (간단한 버전)
            # self.send_email(subject, message)
            
            # MQTT로 알림 발행
            alert_topic = "aquaponics/alerts/system"
            self.mqtt_client.publish(alert_topic, json.dumps({
                'timestamp': datetime.now().isoformat(),
                'alert': alert_data
            }))
            
            logging.info(f"알림 전송: {subject}")
            
        except Exception as e:
            logging.error(f"알림 전송 실패: {e}")
            
    def cleanup_worker(self):
        """정리 작업 워커"""
        while True:
            try:
                # 1시간마다 실행
                time.sleep(3600)
                
                # 오래된 이벤트 삭제 (7일 이상)
                cutoff_date = datetime.now() - timedelta(days=7)
                cursor = self.conn.execute(
                    'DELETE FROM events WHERE timestamp < ?', 
                    (cutoff_date,)
                )
                deleted_count = cursor.rowcount
                self.conn.commit()
                
                if deleted_count > 0:
                    logging.info(f"오래된 이벤트 {deleted_count}개 삭제")
                
                # 메모리 정리
                self.unload_model_if_idle()
                gc.collect()
                
                # 캐시 정리
                if len(self.detection_cache) > self.max_cache_size:
                    # 오래된 항목부터 삭제
                    sorted_items = sorted(self.detection_cache.items(), 
                                        key=lambda x: x[1].get('timestamp', 0))
                    for key, _ in sorted_items[:self.max_cache_size//2]:
                        del self.detection_cache[key]
                        
            except Exception as e:
                logging.error(f"정리 작업 오류: {e}")
                
    def performance_worker(self):
        """성능 모니터링 워커"""
        while True:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                # 성능 상태 MQTT 발행
                perf_data = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu': cpu_percent,
                    'memory': memory_percent,
                    'disk': disk_percent,
                    'status': 'normal'
                }
                
                # 임계값 체크
                if (cpu_percent > self.performance_monitor['cpu_threshold'] or
                    memory_percent > self.performance_monitor['memory_threshold']):
                    
                    perf_data['status'] = 'warning'
                    logging.warning(f"성능 경고 - CPU: {cpu_percent}%, RAM: {memory_percent}%")
                    
                    # 자동 최적화 조치
                    if memory_percent > 85:
                        self.unload_model_if_idle()
                        gc.collect()
                        
                self.mqtt_client.publish(
                    "aquaponics/system/performance", 
                    json.dumps(perf_data)
                )
                
                time.sleep(60)  # 1분마다 체크
                
            except Exception as e:
                logging.error(f"성능 모니터링 오류: {e}")
                time.sleep(60)

# Flask API 엔드포인트들 (경량화)
server_system = LowSpecServerSystem()

@app.route('/api/analyze', methods=['POST'])
def analyze_frame():
    """경량화된 프레임 분석"""
    try:
        # 큐가 가득 찬 경우 거부
        if server_system.analysis_queue.full():
            return jsonify({'error': '서버 과부하'}), 503
            
        frame_file = request.files.get('frame')
        events_data = json.loads(request.form.get('events', '[]'))
        
        if not frame_file:
            return jsonify({'error': '프레임 없음'}), 400
            
        # 간단한 분석만 수행 (모델 로드 최소화)
        timestamp = datetime.now().isoformat()
        
        # 이벤트 데이터만 처리 (프레임 분석 생략)
        enhanced_events = []
        for event in events_data:
            enhanced_event = event.copy()
            enhanced_event['server_timestamp'] = timestamp
            enhanced_event['analysis_type'] = 'lightweight'
            enhanced_events.append(enhanced_event)
            
        # 큐에 추가
        server_system.analysis_queue.put({
            'topic': 'api/analyze',
            'payload': {'events': enhanced_events},
            'timestamp': datetime.now()
        })
        
        return jsonify({
            'status': 'queued',
            'events_count': len(enhanced_events),
            'timestamp': timestamp
        })
        
    except Exception as e:
        logging.error(f"API 분석 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/summary', methods=['GET'])
def get_daily_summary():
    """일간 요약 정보"""
    try:
        today = datetime.now().date()
        
        # 오늘의 이벤트 통계
        cursor = server_system.conn.execute('''
            SELECT event_type, COUNT(*) as count
            FROM events 
            WHERE DATE(timestamp) = ?
            GROUP BY event_type
        ''', (today,))
        
        event_stats = dict(cursor.fetchall())
        
        # 시스템 건강도 (간단한 계산)
        total_events = sum(event_stats.values())
        alert_events = event_stats.get('fish_growth', 0) + event_stats.get('plant_growth', 0)
        health_score = max(0.5, 1.0 - (alert_events / max(total_events, 1)) * 0.5)
        
        return jsonify({
            'date': today.isoformat(),
            'event_statistics': event_stats,
            'total_events': total_events,
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.8 else 'warning'
        })
        
    except Exception as e:
        logging.error(f"요약 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """시스템 상태 조회"""
    try:
        # 성능 정보
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # 큐 상태
        queue_status = {
            'analysis_queue': server_system.analysis_queue.qsize(),
            'alert_queue': server_system.alert_queue.qsize()
        }
        
        # 최근 활동
        cursor = server_system.conn.execute('''
            SELECT COUNT(*) FROM events 
            WHERE timestamp > datetime('now', '-1 hour')
        ''')
        recent_activity = cursor.fetchone()[0]
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'performance': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent
            },
            'queues': queue_status,
            'recent_activity': recent_activity,
            'model_loaded': server_system.model_loaded,
            'uptime': time.time() - server_system.start_time if hasattr(server_system, 'start_time') else 0
        })
        
    except Exception as e:
        logging.error(f"상태 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_recent_alerts():
    """최근 알림 조회"""
    try:
        hours = request.args.get('hours', 24, type=int)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor = server_system.conn.execute('''
            SELECT timestamp, event_type, object_id, data
            FROM events 
            WHERE timestamp > ? AND event_type LIKE '%growth%'
            ORDER BY timestamp DESC LIMIT 50
        ''', (cutoff_time,))
        
        alerts = []
        for row in cursor.fetchall():
            try:
                data = json.loads(row[3])
                alerts.append({
                    'timestamp': row[0],
                    'type': row[1],
                    'object_id': row[2],
                    'change': data.get('change', 0),
                    'severity': 'high' if data.get('change', 0) > 0.5 else 'normal'
                })
            except:
                continue
                
        return jsonify({
            'alerts': alerts,
            'count': len(alerts),
            'period_hours': hours
        })
        
    except Exception as e:
        logging.error(f"알림 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

def start_server():
    """서버 시작"""
    server_system.start_time = time.time()
    logging.info("=== 아쿠아포닉스 서버 시작 ===")
    logging.info(f"CPU 코어: {psutil.cpu_count()}")
    logging.info(f"메모리: {psutil.virtual_memory().total // (1024**3)}GB")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    start_server()
