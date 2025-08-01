# edge/edge_controller.py - Edge 디바이스 메인 컨트롤러
import cv2
import numpy as np
import yaml
import time
import threading
import queue
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import requests
import gc

# 공통 모듈 import
import sys
sys.path.append('..')
from common.data_types import *
from common.mqtt_client import MQTTClient
from common.sensor_manager import SensorManager
from common.yolo_detector import YOLODetector
from edge.control_policy import ControlPolicy
from edge.camera_manager import CameraManager

class EdgeController:
    """Edge 디바이스 메인 컨트롤러"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # 설정 로드
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device_id = self.config['edge']['device_id']
        self.setup_logging()
        
        # 컴포넌트 초기화
        self.camera_manager = CameraManager(self.config)
        self.sensor_manager = SensorManager(self.config)
        self.yolo_detector = YOLODetector(self.config['edge']['yolo'], device='cpu')
        self.mqtt_client = MQTTClient(self.config)
        self.control_policy = ControlPolicy(self.config)
        
        # 데이터 큐
        self.event_queue = queue.Queue(maxsize=self.config['edge']['processing']['queue_max_size'])
        self.control_queue = queue.Queue(maxsize=20)
        
        # 상태 관리
        self.running = False
        self.last_sensor_read = 0
        self.last_control_execution = 0
        self.frame_count = 0
        
        # 성능 모니터링
        self.performance_stats = {
            'fps': 0,
            'detection_time': 0,
            'sensor_read_time': 0,
            'total_events': 0,
            'mqtt_errors': 0
        }
        
        logging.info(f"Edge Controller 초기화 완료: {self.device_id}")
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=getattr(logging, self.config['system']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/edge_{self.device_id}.log'),
                logging.StreamHandler()
            ]
        )
        
    def start(self):
        """메인 실행 시작"""
        self.running = True
        logging.info("Edge Controller 시작")
        
        # 백그라운드 스레드들 시작
        threads = [
            threading.Thread(target=self.sensor_loop, daemon=True),
            threading.Thread(target=self.control_loop, daemon=True),
            threading.Thread(target=self.event_processor, daemon=True),
            threading.Thread(target=self.performance_monitor, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        # 메인 비전 루프
        self.vision_loop()
        
    def vision_loop(self):
        """메인 비전 처리 루프"""
        fps_counter = 0
        fps_start_time = time.time()
        frame_skip = self.config['edge']['processing']['frame_skip']
        
        try:
            while self.running:
                start_time = time.time()
                
                # 프레임 획득
                frame = self.camera_manager.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # 프레임 스키핑
                if self.frame_count % frame_skip != 0:
                    continue
                
                # YOLO 객체 탐지
                detection_start = time.time()
                yolo_result = self.yolo_detector.detect(frame, generate_frame_id())
                detection_time = time.time() - detection_start
                
                # 센서 데이터 읽기 (주기적)
                sensor_data = None
                current_time = time.time()
                if current_time - self.last_sensor_read >= self.config['sensors']['read_interval']:
                    sensor_data = self.sensor_manager.read_all_sensors()
                    self.last_sensor_read = current_time
                
                # 시스템 이벤트 생성
                if yolo_result.objects or sensor_data:
                    event = SystemEvent.create_edge_event(
                        device_id=self.device_id,
                        sensors=sensor_data,
                        yolo_result=yolo_result if yolo_result.objects else None
                    )
                    event.edge_latency = time.time() - start_time
                    
                    # 프레임 저장 (디버그 모드)
                    if self.config['debug']['save_frames']:
                        frame_path = self.save_debug_frame(frame, event.event_id)
                        event.frame_path = frame_path
                    
                    # 이벤트 큐에 추가
                    try:
                        self.event_queue.put_nowait(event)
                    except queue.Full:
                        logging.warning("이벤트 큐가 가득참 - 이벤트 드롭")
                
                # 제어 정책 실행 (주기적)
                if (current_time - self.last_control_execution >= 
                    self.config['edge']['control']['control_interval']):
                    self.execute_control_policy(sensor_data, yolo_result)
                    self.last_control_execution = current_time
                
                # FPS 계산
                fps_counter += 1
                if fps_counter >= 30:
                    elapsed = time.time() - fps_start_time
                    self.performance_stats['fps'] = fps_counter / elapsed
                    self.performance_stats['detection_time'] = detection_time
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # FPS 제한
                target_fps = self.config['edge']['camera']['fps']
                sleep_time = max(0, 1.0/target_fps - (time.time() - start_time))
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logging.info("사용자 중단 요청")
        except Exception as e:
            logging.error(f"비전 루프 오류: {e}")
        finally:
            self.cleanup()
    
    def sensor_loop(self):
        """센서 백그라운드 루프"""
        while self.running:
            try:
                if self.sensor_manager.has_sensors():
                    start_time = time.time()
                    sensor_data = self.sensor_manager.read_all_sensors()
                    self.performance_stats['sensor_read_time'] = time.time() - start_time
                    
                    # 센서 전용 이벤트 생성 (중요한 변화가 있을 때만)
                    if self.sensor_manager.has_significant_change(sensor_data):
                        event = SystemEvent.create_edge_event(
                            device_id=self.device_id,
                            sensors=sensor_data
                        )
                        
                        try:
                            self.event_queue.put_nowait(event)
                        except queue.Full:
                            logging.warning("센서 이벤트 드롭 - 큐 가득참")
                
                time.sleep(self.config['sensors']['read_interval'])
                
            except Exception as e:
                logging.error(f"센서 루프 오류: {e}")
                time.sleep(5)
    
    def control_loop(self):
        """제어 명령 실행 루프"""
        while self.running:
            try:
                # 제어 큐에서 명령 처리
                try:
                    command = self.control_queue.get(timeout=1)
                    self.execute_control_command(command)
                    self.control_queue.task_done()
                except queue.Empty:
                    continue
                    
            except Exception as e:
                logging.error(f"제어 루프 오류: {e}")
                time.sleep(1)
    
    def event_processor(self):
        """이벤트 처리 및 전송 루프"""
        batch = []
        batch_size = 3
        last_send = 0
        
        while self.running:
            try:
                # 이벤트 수집
                try:
                    event = self.event_queue.get(timeout=2)
                    batch.append(event)
                except queue.Empty:
                    pass
                
                # 배치 전송 (크기 또는 시간 기준)
                current_time = time.time()
                if (len(batch) >= batch_size or 
                    (batch and current_time - last_send > 5)):
                    
                    self.send_events_batch(batch)
                    self.performance_stats['total_events'] += len(batch)
                    batch = []
                    last_send = current_time
                    
            except Exception as e:
                logging.error(f"이벤트 처리 오류: {e}")
                time.sleep(1)
    
    def execute_control_policy(self, sensor_data: EnvironmentData, yolo_result: YOLOResult):
        """제어 정책 실행"""
        try:
            if not self.config['edge']['control']['enable_auto_control']:
                return
            
            # 제어 정책에서 명령 생성
            commands = self.control_policy.generate_commands(sensor_data, yolo_result)
            
            for command in commands:
                try:
                    self.control_queue.put_nowait(command)
                except queue.Full:
                    logging.warning(f"제어 명령 드롭: {command.command_type}")
                    
        except Exception as e:
            logging.error(f"제어 정책 실행 오류: {e}")
    
    def execute_control_command(self, command: ControlCommand):
        """개별 제어 명령 실행"""
        try:
            logging.info(f"제어 명령 실행: {command.device_id} = {command.target_value}")
            
            # 실제 하드웨어 제어 (GPIO/I2C/등)
            success = self.control_policy.execute_command(command)
            
            if success:
                # 제어 명령 성공을 MQTT로 전송
                control_event = {
                    'event_id': generate_event_id(),
                    'timestamp': datetime.now().isoformat(),
                    'device_id': self.device_id,
                    'event_type': 'control_executed',
                    'command': command.to_dict(),
                    'status': 'success'
                }
                
                self.mqtt_client.publish('control', control_event)
                
        except Exception as e:
            logging.error(f"제어 명령 실행 실패: {e}")
            
            # 실패 이벤트 전송
            error_event = {
                'event_id': generate_event_id(),
                'timestamp': datetime.now().isoformat(),
                'device_id': self.device_id,
                'event_type': 'control_error',
                'command': command.to_dict(),
                'error': str(e)
            }
            
            self.mqtt_client.publish('alerts', error_event)
    
    def send_events_batch(self, events: List[SystemEvent]):
        """이벤트 배치 전송"""
        try:
            # MQTT 전송
            for event in events:
                success = self.mqtt_client.publish('events', event.to_dict())
                if not success:
                    self.performance_stats['mqtt_errors'] += 1
            
            # Server HTTP 전송 (중요한 이벤트만)
            critical_events = [e for e in events if self.is_critical_event(e)]
            if critical_events:
                self.send_to_server(critical_events)
                
        except Exception as e:
            logging.error(f"이벤트 전송 오류: {e}")
            self.performance_stats['mqtt_errors'] += len(events)
    
    def is_critical_event(self, event: SystemEvent) -> bool:
        """중요 이벤트 판단"""
        if not event.yolo_result:
            return False
            
        # 새로운 객체 탐지
        if len(event.yolo_result.objects) > 0:
            return True
            
        # 건강도 임계값 미달
        for obj in event.yolo_result.objects:
            if obj.health_score and obj.health_score < self.config['edge']['control']['emergency_threshold']:
                return True
                
        return False
    
    def send_to_server(self, events: List[SystemEvent]):
        """서버로 이벤트 전송 (HTTP)"""
        try:
            server_url = f"http://{self.config['server']['host']}:{self.config['server']['port']}"
            
            # 이벤트 데이터 준비
            event_data = {
                'device_id': self.device_id,
                'events': [event.to_dict() for event in events],
                'timestamp': datetime.now().isoformat()
            }
            
            # HTTP POST 전송
            response = requests.post(
                f"{server_url}/api/edge/events",
                json=event_data,
                timeout=10
            )
            
            if response.status_code == 200:
                logging.info(f"서버 전송 성공: {len(events)}개 이벤트")
                
                # 서버 응답에서 제어 명령 확인
                response_data = response.json()
                if 'control_commands' in response_data:
                    for cmd_data in response_data['control_commands']:
                        command = ControlCommand(
                            command_id=cmd_data['command_id'],
                            device_id=cmd_data['device_id'],
                            command_type=cmd_data['command_type'],
                            target_value=cmd_data['target_value'],
                            duration=cmd_data.get('duration'),
                            priority=cmd_data.get('priority', 1)
                        )
                        self.control_queue.put_nowait(command)
                        
            else:
                logging.warning(f"서버 전송 실패: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"서버 전송 오류: {e}")
        except Exception as e:
            logging.error(f"서버 전송 처리 오류: {e}")
    
    def save_debug_frame(self, frame: np.ndarray, event_id: str) -> str:
        """디버그용 프레임 저장"""
        try:
            debug_dir = Path(self.config['debug']['frame_save_path'])
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{event_id}_{datetime.now().strftime('%H%M%S')}.jpg"
            filepath = debug_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            return str(filepath)
            
        except Exception as e:
            logging.error(f"프레임 저장 실패: {e}")
            return None
    
    def performance_monitor(self):
        """성능 모니터링 루프"""
        while self.running:
            try:
                # 성능 통계 MQTT 전송
                perf_data = {
                    'timestamp': datetime.now().isoformat(),
                    'device_id': self.device_id,
                    'performance': self.performance_stats.copy(),
                    'queue_sizes': {
                        'events': self.event_queue.qsize(),
                        'control': self.control_queue.qsize()
                    },
                    'memory_usage': self.get_memory_usage()
                }
                
                self.mqtt_client.publish('status', perf_data)
                
                # 메모리 정리 (필요시)
                if perf_data['memory_usage'] > 80:
                    gc.collect()
                    logging.info("메모리 정리 수행")
                
                time.sleep(30)  # 30초마다 성능 모니터링
                
            except Exception as e:
                logging.error(f"성능 모니터링 오류: {e}")
                time.sleep(30)
    
    def get_memory_usage(self) -> float:
        """메모리 사용률 조회"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def cleanup(self):
        """리소스 정리"""
        logging.info("Edge Controller 종료 중...")
        self.running = False
        
        # 컴포넌트 정리
        if hasattr(self, 'camera_manager'):
            self.camera_manager.cleanup()
            
        if hasattr(self, 'sensor_manager'):
            self.sensor_manager.cleanup()
            
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.disconnect()
        
        logging.info("Edge Controller 종료 완료")

def main():
    """메인 실행 함수"""
    try:
        controller = EdgeController()
        controller.start()
    except KeyboardInterrupt:
        logging.info("프로그램 종료")
    except Exception as e:
        logging.error(f"메인 실행 오류: {e}")

if __name__ == "__main__":
    main()
