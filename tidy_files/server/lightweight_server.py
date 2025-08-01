# server/lightweight_server.py - 경량화된 통합 서버 시스템
import os
import sys
import time
import yaml
import logging
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path

# Flask 웹 서버
from flask import Flask, request, jsonify, send_from_directory

# 공통 모듈
sys.path.append('..')
from common.data_types import SystemEvent, ControlCommand, generate_event_id
from common.mqtt_client import MQTTClient
from common.yolo_detector import YOLODetector

# 서버 모듈
from database_manager import DatabaseManager
from analysis_engine import AnalysisEngine
from alert_manager import AlertManager

# Flask 앱 초기화
app = Flask(__name__)

class LightweightServer:
    """경량화된 아쿠아포닉스 서버 시스템"""
    
    def __init__(self, config_path: str = "../config.yaml"):
        # 설정 로드
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device_id = self.config['server']['device_id']
        self.setup_logging()
        
        # 컴포넌트 초기화 (경량화)
        self.database_manager = DatabaseManager(self.config)
        self.analysis_engine = AnalysisEngine(self.config)
        self.alert_manager = AlertManager(self.config)
        self.mqtt_client = MQTTClient(self.config)
        
        # YOLO 모델 (필요시에만 로드)
        self.yolo_detector = None
        self.model_loaded = False
        self.last_model_use = 0
        self.model_timeout = 600  # 10분 후 언로드
        
        # 처리 큐들 (크기 제한)
        self.event_queue = queue.Queue(maxsize=50)
        self.analysis_queue = queue.Queue(maxsize=20)
        self.alert_queue = queue.Queue(maxsize=10)
        
        # 시스템 상태
        self.running = False
        self.start_time = time.time()
        self.processed_events = 0
        self.performance_stats = {
            'events_processed': 0,
            'analysis_completed': 0,
            'alerts_sent': 0,
            'avg_processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0
        }
        
        # MQTT 구독 설정
        self.setup_mqtt_subscriptions()
        
        logging.info(f"경량 서버 시스템 초기화 완료: {self.device_id}")
    
    def setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['system']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/server_{self.device_id}.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_mqtt_subscriptions(self):
        """MQTT 구독 설정"""
        def on_event_message(topic, payload):
            try:
                event = SystemEvent.from_dict(payload)
                
                if not self.event_queue.full():
                    self.event_queue.put(event)
                else:
                    logging.warning("이벤트 큐 가득참 - 이벤트 드롭")
                    
            except Exception as e:
                logging.error(f"MQTT 이벤트 처리 오류: {e}")
        
        # 이벤트 및 센서 데이터 구독
        self.mqtt_client.subscribe('events', on_event_message)
        self.mqtt_client.subscribe('sensors', on_event_message)
    
    def load_yolo_model(self):
        """필요시 YOLO 모델 로드"""
        if not self.model_loaded:
            try:
                device = 'cuda' if self.config['server']['hardware']['use_gpu'] else 'cpu'
                self.yolo_detector = YOLODetector(self.config['server']['yolo'], device)
                self.model_loaded = True
                logging.info("YOLO 모델 로드 완료")
            except Exception as e:
                logging.error(f"YOLO 모델 로드 실패: {e}")
                return False
        
        self.last_model_use = time.time()
        return True
    
    def unload_yolo_model(self):
        """모델 언로드 (메모리 절약)"""
        if self.model_loaded and time.time() - self.last_model_use > self.model_timeout:
            del self.yolo_detector
            self.yolo_detector = None
            self.model_loaded = False
            
            # 가비지 컬렉션
            import gc
            gc.collect()
            
            logging.info("YOLO 모델 언로드 (메모리 절약)")
    
    def start(self):
        """서버 시작"""
        self.running = True
        logging.info("경량 서버 시작")
        
        # 백그라운드 워커들 시작
        workers = [
            threading.Thread(target=self.event_processor, daemon=True),
            threading.Thread(target=self.analysis_worker, daemon=True),
            threading.Thread(target=self.alert_worker, daemon=True),
            threading.Thread(target=self.maintenance_worker, daemon=True),
            threading.Thread(target=self.performance_monitor, daemon=True)
        ]
        
        for worker in workers:
            worker.start()
        
        logging.info("모든 백그라운드 워커 시작 완료")
    
    def event_processor(self):
        """이벤트 처리 워커"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=2)
                start_time = time.time()
                
                # 데이터베이스 저장
                self.database_manager.save_edge_event(event)
                
                # 분석 필요 여부 판단
                if self._needs_analysis(event):
                    if not self.analysis_queue.full():
                        self.analysis_queue.put(event)
                    else:
                        logging.warning("분석 큐 가득참")
                
                # 즉시 알림 필요 여부 판단
                if self._needs_immediate_alert(event):
                    if not self.alert_queue.full():
                        self.alert_queue.put(event)
                
                # 처리 시간 통계 업데이트
                processing_time = time.time() - start_time
                self._update_processing_stats(processing_time)
                
                self.processed_events += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"이벤트 처리 오류: {e}")
    
    def analysis_worker(self):
        """분석 워커"""
        while self.running:
            try:
                event = self.analysis_queue.get(timeout=5)
                
                # YOLO 모델 로드 (필요시)
                if not self.load_yolo_model():
                    continue
                
                # 서버 재분석 수행
                server_event = self._perform_server_analysis(event)
                
                if server_event:
                    # 서버 결과 저장
                    self.database_manager.save_server_event(server_event)
                    
                    # 인사이트 생성
                    insights = self.analysis_engine.generate_insights(server_event)
                    if insights:
                        logging.info(f"인사이트 생성: {len(insights.get('insights', []))}개")
                    
                    # 분석 결과 기반 알림
                    alerts = self.alert_manager.create_alerts(server_event)
                    for alert in alerts:
                        if not self.alert_queue.full():
                            self.alert_queue.put(alert)
                
                self.performance_stats['analysis_completed'] += 1
                
                # 분석 완료 후 일정 시간 뒤 모델 언로드 체크
                threading.Timer(60, self.unload_yolo_model).start()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"분석 워커 오류: {e}")
    
    def alert_worker(self):
        """알림 워커"""
        while self.running:
            try:
                # 이벤트 또는 알림 객체 처리
                alert_item = self.alert_queue.get(timeout=10)
                
                if isinstance(alert_item, SystemEvent):
                    # 이벤트에서 알림 생성
                    alerts = self.alert_manager.create_alerts(alert_item)
                    for alert in alerts:
                        success = self.alert_manager.send_alert(alert)
                        if success:
                            self.performance_stats['alerts_sent'] += 1
                
                elif isinstance(alert_item, dict):
                    # 직접 알림 전송
                    success = self.alert_manager.send_alert(alert_item)
                    if success:
                        self.performance_stats['alerts_sent'] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"알림 워커 오류: {e}")
    
    def maintenance_worker(self):
        """유지보수 워커"""
        while self.running:
            try:
                # 1시간마다 실행
                time.sleep(3600)
                
                # 데이터베이스 정리
                self.database_manager.cleanup_old_data()
                
                # 메모리 정리
                self.unload_yolo_model()
                
                # 데이터베이스 최적화 (매일 1회)
                current_hour = datetime.now().hour
                if current_hour == 3:  # 새벽 3시
                    self.database_manager.optimize_database()
                
                logging.info("유지보수 작업 완료")
                
            except Exception as e:
                logging.error(f"유지보수 워커 오류: {e}")
    
    def performance_monitor(self):
        """성능 모니터링"""
        while self.running:
            try:
                # 시스템 리소스 체크
                try:
                    import psutil
                    self.performance_stats['memory_usage_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
                    self.performance_stats['cpu_usage_percent'] = psutil.cpu_percent()
                except ImportError:
                    pass
                
                # 성능 데이터 MQTT 전송
                perf_data = {
                    'timestamp': datetime.now().isoformat(),
                    'device_id': self.device_id,
                    'uptime_hours': (time.time() - self.start_time) / 3600,
                    'performance': self.performance_stats.copy(),
                    'queue_sizes': {
                        'events': self.event_queue.qsize(),
                        'analysis': self.analysis_queue.qsize(),
                        'alerts': self.alert_queue.qsize()
                    },
                    'model_loaded': self.model_loaded
                }
                
                self.mqtt_client.publish('status', perf_data)
                
                time.sleep(60)  # 1분마다
                
            except Exception as e:
                logging.error(f"성능 모니터링 오류: {e}")
                time.sleep(60)
    
    def _needs_analysis(self, event: SystemEvent) -> bool:
        """분석 필요 여부 판단"""
        # YOLO 결과가 있는 경우
        if event.yolo_result and event.yolo_result.objects:
            return True
        
        # 센서 이상값
        if event.sensors and self.analysis_engine.has_sensor_anomaly(event.sensors):
            return True
        
        return False
    
    def _needs_immediate_alert(self, event: SystemEvent) -> bool:
        """즉시 알림 필요 여부"""
        # 센서 임계값 체크
        if event.sensors:
            if (event.sensors.water_temp and 
                (event.sensors.water_temp < 18 or event.sensors.water_temp > 30)):
                return True
            
            if (event.sensors.ph and 
                (event.sensors.ph < 6.0 or event.sensors.ph > 8.0)):
                return True
        
        # 건강도 임계값
        if event.yolo_result:
            for obj in event.yolo_result.objects:
                if obj.health_score and obj.health_score < 0.3:
                    return True
        
        return False
    
    def _perform_server_analysis(self, event: SystemEvent) -> Optional[SystemEvent]:
        """서버 고정밀 분석"""
        try:
            # 프레임이 있는 경우에만 YOLO 재분석
            if event.frame_path and Path(event.frame_path).exists():
                import cv2
                frame = cv2.imread(event.frame_path)
                
                if frame is not None:
                    # 고정밀 YOLO 분석
                    server_yolo_result = self.yolo_detector.detect(
                        frame, event.yolo_result.frame_id if event.yolo_result else generate_event_id()
                    )
                    
                    # 물리적 측정값 계산
                    for obj in server_yolo_result.objects:
                        if obj.class_name == 'fish':
                            length_pixels = max(obj.bbox.width, obj.bbox.height)
                            obj.length_cm = length_pixels * 0.08  # 픽셀 to cm 변환
                            obj.weight_g = (obj.length_cm ** 2) * 0.8  # 무게 추정
                        
                        elif obj.class_name == 'plant':
                            obj.height_cm = obj.bbox.height * 0.08
                        
                        # 건강도 분석
                        roi = frame[obj.bbox.y1:obj.bbox.y2, obj.bbox.x1:obj.bbox.x2]
                        obj.health_score = self.analysis_engine.calculate_health_score(obj, roi)
                        obj.activity_level = self.analysis_engine.calculate_activity_level(obj)
                    
                    # 서버 이벤트 생성
                    server_event = SystemEvent.create_server_event(event, server_yolo_result)
                    server_event.server_latency = 0.5  # 예시값
                    
                    return server_event
            
            # 센서 데이터만 있는 경우
            elif event.sensors:
                enhanced_sensors = self.analysis_engine.enhance_sensor_data(event.sensors)
                server_event = SystemEvent.create_server_event(event)
                server_event.sensors = enhanced_sensors
                return server_event
            
            return None
            
        except Exception as e:
            logging.error(f"서버 분석 오류: {e}")
            return None
    
    def _update_processing_stats(self, processing_time: float):
        """처리 통계 업데이트"""
        stats = self.performance_stats
        
        # 이동 평균 계산
        if stats['avg_processing_time'] == 0:
            stats['avg_processing_time'] = processing_time
        else:
            stats['avg_processing_time'] = (stats['avg_processing_time'] * 0.9 + processing_time * 0.1)
        
        stats['events_processed'] += 1
    
    def get_system_status(self) -> Dict:
        """시스템 상태 조회"""
        try:
            import psutil
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'server_id': self.device_id,
                'uptime_seconds': time.time() - self.start_time,
                'system': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent,
                    'cpu_count': psutil.cpu_count()
                },
                'processing': {
                    'events_processed': self.processed_events,
                    'queue_sizes': {
                        'events': self.event_queue.qsize(),
                        'analysis': self.analysis_queue.qsize(),
                        'alerts': self.alert_queue.qsize()
                    },
                    'model_status': {
                        'loaded': self.model_loaded,
                        'last_use': self.last_model_use,
                        'timeout': self.model_timeout
                    }
                },
                'performance': self.performance_stats.copy(),
                'database': self.database_manager.get_stats()
            }
            
            return status
            
        except Exception as e:
            logging.error(f"시스템 상태 조회 오류: {e}")
            return {}
    
    def cleanup(self):
        """리소스 정리"""
        logging.info("서버 시스템 종료 중...")
        self.running = False
        
        # 모델 언로드
        if self.model_loaded:
            del self.yolo_detector
        
        # 데이터베이스 연결 종료
        self.database_manager.close()
        
        # MQTT 연결 종료
        self.mqtt_client.disconnect()
        
        logging.info("서버 시스템 종료 완료")

# 전역 서버 인스턴스
server = None

# Flask API 엔드포인트들
@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """시스템 상태 조회"""
    try:
        if server:
            status = server.get_system_status()
            return jsonify(status)
        else:
            return jsonify({'error': '서버 초기화되지 않음'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/edge/events', methods=['POST'])
def receive_edge_events():
    """Edge 이벤트 수신"""
    try:
        data = request.get_json()
        
        if not data or 'events' not in data:
            return jsonify({'error': '이벤트 데이터 없음'}), 400
        
        device_id = data.get('device_id', 'unknown')
        events_data = data['events']
        
        received_events = []
        for event_data in events_data:
            try:
                event = SystemEvent.from_dict(event_data)
                
                if not server.event_queue.full():
                    server.event_queue.put(event)
                    received_events.append(event.event_id)
                else:
                    logging.warning(f"이벤트 큐 가득참 - {event.event_id} 드롭")
                    
            except Exception as e:
                logging.error(f"이벤트 복원 오류: {e}")
        
        # 제어 명령 생성 (간단한 규칙 기반)
        control_commands = []
        for event_data in events_data:
            if 'sensors' in event_data:
                commands = server.analysis_engine.generate_control_commands(
                    event_data['sensors'], device_id
                )
                control_commands.extend([cmd.to_dict() for cmd in commands])
        
        return jsonify({
            'status': 'received',
            'processed_events': len(received_events),
            'control_commands': control_commands,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Edge 이벤트 수신 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/summary', methods=['GET'])
def get_analysis_summary():
    """분석 요약"""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        # 기본 요약 정보 (실제로는 DB에서 조회)
        summary = {
            'period_hours': hours,
            'total_events': server.processed_events,
            'object_detections': {
                'fish': 15,
                'plant': 8
            },
            'sensor_readings': {
                'temperature': 45,
                'ph': 42,
                'dissolved_oxygen': 38
            },
            'alerts_generated': server.performance_stats['alerts_sent'],
            'growth_analysis': server.analysis_engine.get_growth_summary(
                datetime.now() - timedelta(hours=hours)
            ),
            'health_metrics': server.analysis_engine.get_health_summary(
                datetime.now() - timedelta(hours=hours)
            ),
            'system_performance': server.performance_stats.copy()
        }
        
        return jsonify(summary)
        
    except Exception as e:
        logging.error(f"분석 요약 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/objects/tracking', methods=['GET'])
def get_object_tracking():
    """객체 추적"""
    try:
        object_type = request.args.get('type', 'all')
        hours = request.args.get('hours', 24, type=int)
        
        tracking_data = server.analysis_engine.get_object_tracking(object_type, hours)
        
        return jsonify({
            'object_type': object_type,
            'period_hours': hours,
            'tracking_data': tracking_data
        })
        
    except Exception as e:
        logging.error(f"객체 추적 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/environment/trends', methods=['GET'])
def get_environment_trends():
    """환경 트렌드"""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        trends = server.analysis_engine.get_environment_trends(hours)
        
        return jsonify({
            'period_hours': hours,
            'trends': trends
        })
        
    except Exception as e:
        logging.error(f"환경 트렌드 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/recent', methods=['GET'])
def get_recent_alerts():
    """최근 알림"""
    try:
        hours = request.args.get('hours', 24, type=int)
        severity = request.args.get('severity', 'all')
        
        alerts = server.alert_manager.get_recent_alerts(hours, severity)
        
        return jsonify({
            'period_hours': hours,
            'severity_filter': severity,
            'alerts': alerts
        })
        
    except Exception as e:
        logging.error(f"알림 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/control/command', methods=['POST'])
def send_control_command():
    """제어 명령 전송"""
    try:
        data = request.get_json()
        
        if not data or 'device_id' not in data:
            return jsonify({'error': '제어 명령 데이터 없음'}), 400
        
        command = ControlCommand(
            command_id=generate_event_id(),
            device_id=data['device_id'],
            command_type=data['command_type'],
            target_value=data['target_value'],
            duration=data.get('duration'),
            priority=data.get('priority', 1)
        )
        
        # MQTT로 제어 명령 전송
        success = server.mqtt_client.publish('control', {
            'target_device': command.device_id,
            'command': command.to_dict()
        })
        
        if success:
            # 데이터베이스에 명령 기록
            server.database_manager.save_control_command(command)
            
            return jsonify({
                'status': 'sent',
                'command_id': command.command_id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'MQTT 전송 실패'}), 500
            
    except Exception as e:
        logging.error(f"제어 명령 전송 오류: {e}")
        return jsonify({'error': str(e)}), 500

def main():
    """메인 실행 함수"""
    global server
    
    try:
        # 서버 초기화
        server = LightweightServer()
        server.start()
        
        # Flask 앱 실행
        host = server.config['server']['host']
        port = server.config['server']['port']
        
        logging.info(f"Flask 서버 시작: http://{host}:{port}")
        
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logging.info("서버 종료 요청")
        if server:
            server.cleanup()
    except Exception as e:
        logging.error(f"메인 실행 오류: {e}")
        if server:
            server.cleanup()

if __name__ == "__main__":
    main()
