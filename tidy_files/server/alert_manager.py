# server/alert_manager.py - 경량화된 알림 관리자
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

import sys
sys.path.append('..')
from common.data_types import SystemEvent

class AlertManager:
    """경량화된 알림 관리 시스템"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_config = config.get('alerts', {})
        
        # 알림 쿨다운 (스팸 방지)
        self.alert_cooldowns = {}
        self.default_cooldown = 300  # 5분
        
        # 알림 이력 (메모리 기반, 경량화)
        self.recent_alerts = []
        self.max_alert_history = 100
        
        # 임계값
        self.thresholds = self.alert_config.get('thresholds', {})
        
        logging.info("알림 관리자 초기화 완료")
    
    def create_alerts(self, event: SystemEvent) -> List[Dict]:
        """이벤트 기반 알림 생성"""
        alerts = []
        
        try:
            # 환경 센서 알림
            if event.sensors:
                sensor_alerts = self._check_sensor_alerts(event.sensors)
                alerts.extend(sensor_alerts)
            
            # 객체 탐지 알림
            if event.yolo_result and event.yolo_result.objects:
                detection_alerts = self._check_detection_alerts(event.yolo_result)
                alerts.extend(detection_alerts)
            
            # 시스템 알림
            system_alerts = self._check_system_alerts(event)
            alerts.extend(system_alerts)
            
            # 알림 이력에 추가
            for alert in alerts:
                alert['event_id'] = event.event_id
                alert['device_id'] = event.device_id
                alert['created_at'] = datetime.now().isoformat()
                
                self.recent_alerts.append(alert)
                
                # 이력 크기 제한
                if len(self.recent_alerts) > self.max_alert_history:
                    self.recent_alerts.pop(0)
            
            return alerts
            
        except Exception as e:
            logging.error(f"알림 생성 오류: {e}")
            return []
    
    def _check_sensor_alerts(self, sensor_data) -> List[Dict]:
        """센서 기반 알림 체크"""
        alerts = []
        
        try:
            # 수온 체크
            if sensor_data.water_temp:
                temp = sensor_data.water_temp
                temp_min = self.thresholds.get('temperature_min', 20.0)
                temp_max = self.thresholds.get('temperature_max', 28.0)
                
                if temp < temp_min:
                    alerts.append({
                        'type': 'temperature_low',
                        'severity': 'high' if temp < temp_min - 2 else 'medium',
                        'title': '수온 경고',
                        'message': f'수온이 너무 낮습니다: {temp:.1f}°C (최소: {temp_min}°C)',
                        'value': temp,
                        'threshold': temp_min,
                        'action': 'heater_on'
                    })
                elif temp > temp_max:
                    alerts.append({
                        'type': 'temperature_high',
                        'severity': 'high' if temp > temp_max + 2 else 'medium',
                        'title': '수온 경고',
                        'message': f'수온이 너무 높습니다: {temp:.1f}°C (최대: {temp_max}°C)',
                        'value': temp,
                        'threshold': temp_max,
                        'action': 'cooling_on'
                    })
            
            # pH 체크
            if sensor_data.ph:
                ph = sensor_data.ph
                ph_min = self.thresholds.get('ph_min', 6.5)
                ph_max = self.thresholds.get('ph_max', 7.5)
                
                if ph < ph_min:
                    alerts.append({
                        'type': 'ph_low',
                        'severity': 'medium',
                        'title': 'pH 경고',
                        'message': f'pH가 너무 낮습니다: {ph:.1f} (최소: {ph_min})',
                        'value': ph,
                        'threshold': ph_min,
                        'action': 'ph_up_dosing'
                    })
                elif ph > ph_max:
                    alerts.append({
                        'type': 'ph_high',
                        'severity': 'medium',
                        'title': 'pH 경고',
                        'message': f'pH가 너무 높습니다: {ph:.1f} (최대: {ph_max})',
                        'value': ph,
                        'threshold': ph_max,
                        'action': 'ph_down_dosing'
                    })
            
            # 용존산소 체크
            if sensor_data.dissolved_oxygen:
                do = sensor_data.dissolved_oxygen
                do_min = 6.0  # 최소 용존산소
                
                if do < do_min:
                    severity = 'high' if do < 4.0 else 'medium'
                    alerts.append({
                        'type': 'oxygen_low',
                        'severity': severity,
                        'title': '용존산소 부족',
                        'message': f'용존산소가 부족합니다: {do:.1f}mg/L (최소: {do_min}mg/L)',
                        'value': do,
                        'threshold': do_min,
                        'action': 'aeration_increase'
                    })
            
            return alerts
            
        except Exception as e:
            logging.error(f"센서 알림 체크 오류: {e}")
            return []
    
    def _check_detection_alerts(self, yolo_result) -> List[Dict]:
        """객체 탐지 기반 알림 체크"""
        alerts = []
        
        try:
            fish_objects = [obj for obj in yolo_result.objects if obj.class_name == 'fish']
            plant_objects = [obj for obj in yolo_result.objects if obj.class_name == 'plant']
            
            # 건강도 체크
            health_min = self.thresholds.get('health_score_min', 0.7)
            
            for obj in fish_objects + plant_objects:
                if obj.health_score and obj.health_score < health_min:
                    severity = 'high' if obj.health_score < 0.5 else 'medium'
                    alerts.append({
                        'type': 'health_low',
                        'severity': severity,
                        'title': f'{obj.class_name.title()} 건강 경고',
                        'message': f'{obj.object_id} 건강도 낮음: {obj.health_score:.2f}',
                        'object_id': obj.object_id,
                        'value': obj.health_score,
                        'threshold': health_min,
                        'action': 'health_check_required'
                    })
            
            # 개체수 체크
            if len(fish_objects) == 0:
                alerts.append({
                    'type': 'no_fish_detected',
                    'severity': 'medium',
                    'title': '물고기 미감지',
                    'message': '물고기가 감지되지 않습니다. 카메라 또는 시스템을 점검하세요.',
                    'action': 'system_check'
                })
            
            if len(plant_objects) == 0:
                alerts.append({
                    'type': 'no_plants_detected',
                    'severity': 'low',
                    'title': '식물 미감지',
                    'message': '식물이 감지되지 않습니다.',
                    'action': 'plant_check'
                })
            
            # 급격한 성장 변화 (예: 스트레스)
            for obj in fish_objects:
                if obj.length_cm and obj.length_cm < 3.0:  # 너무 작음
                    alerts.append({
                        'type': 'growth_concern',
                        'severity': 'medium',
                        'title': '성장 이상',
                        'message': f'{obj.object_id} 크기 이상: {obj.length_cm:.1f}cm',
                        'object_id': obj.object_id,
                        'value': obj.length_cm,
                        'action': 'growth_investigation'
                    })
            
            return alerts
            
        except Exception as e:
            logging.error(f"탐지 알림 체크 오류: {e}")
            return []
    
    def _check_system_alerts(self, event: SystemEvent) -> List[Dict]:
        """시스템 상태 알림 체크"""
        alerts = []
        
        try:
            # 처리 지연 체크
            if event.edge_latency and event.edge_latency > 5.0:  # 5초 이상
                alerts.append({
                    'type': 'processing_delay',
                    'severity': 'low',
                    'title': '처리 지연',
                    'message': f'Edge 처리 지연: {event.edge_latency:.1f}초',
                    'value': event.edge_latency,
                    'action': 'system_optimization'
                })
            
            if event.server_latency and event.server_latency > 10.0:  # 10초 이상
                alerts.append({
                    'type': 'server_delay',
                    'severity': 'medium',
                    'title': '서버 처리 지연',
                    'message': f'서버 처리 지연: {event.server_latency:.1f}초',
                    'value': event.server_latency,
                    'action': 'server_check'
                })
            
            return alerts
            
        except Exception as e:
            logging.error(f"시스템 알림 체크 오류: {e}")
            return []
    
    def send_alert(self, alert: Dict) -> bool:
        """알림 전송"""
        try:
            alert_type = alert['type']
            
            # 쿨다운 체크
            if not self._can_send_alert(alert_type):
                return False
            
            success = False
            
            # MQTT 알림 (기본)
            if self.alert_config.get('mqtt', {}).get('enable', True):
                success |= self._send_mqtt_alert(alert)
            
            # 이메일 알림 (선택적)
            if (self.alert_config.get('email', {}).get('enable', False) and 
                alert['severity'] in ['high', 'critical']):
                success |= self._send_email_alert(alert)
            
            # SMS 알림 (긴급시)
            if (self.alert_config.get('sms', {}).get('enable', False) and 
                alert['severity'] == 'critical'):
                success |= self._send_sms_alert(alert)
            
            # 웹훅 알림 (확장성)
            if 'webhook_url' in self.alert_config:
                success |= self._send_webhook_alert(alert)
            
            if success:
                self._update_alert_cooldown(alert_type)
                logging.info(f"알림 전송 성공: {alert['title']}")
            
            return success
            
        except Exception as e:
            logging.error(f"알림 전송 오류: {e}")
            return False
    
    def _can_send_alert(self, alert_type: str) -> bool:
        """알림 쿨다운 체크"""
        if alert_type not in self.alert_cooldowns:
            return True
        
        last_sent = self.alert_cooldowns[alert_type]
        cooldown = self.default_cooldown
        
        return (datetime.now() - last_sent).total_seconds() >= cooldown
    
    def _update_alert_cooldown(self, alert_type: str):
        """알림 쿨다운 업데이트"""
        self.alert_cooldowns[alert_type] = datetime.now()
    
    def _send_mqtt_alert(self, alert: Dict) -> bool:
        """MQTT 알림 전송"""
        try:
            # 실제 구현에서는 MQTT 클라이언트 사용
            # 여기서는 로깅만
            logging.info(f"MQTT 알림: {alert['title']} - {alert['message']}")
            return True
            
        except Exception as e:
            logging.error(f"MQTT 알림 전송 오류: {e}")
            return False
    
    def _send_email_alert(self, alert: Dict) -> bool:
        """이메일 알림 전송"""
        try:
            email_config = self.alert_config.get('email', {})
            
            if not email_config.get('enable', False):
                return False
            
            # 간단한 이메일 전송 로직
            # 실제로는 smtplib 사용
            logging.info(f"이메일 알림 전송: {alert['title']}")
            
            # 실제 구현 예시:
            # import smtplib
            # from email.mime.text import MIMEText
            # 
            # msg = MIMEText(alert['message'])
            # msg['Subject'] = f"[아쿠아포닉스] {alert['title']}"
            # msg['From'] = email_config['username']
            # msg['To'] = ', '.join(email_config['recipients'])
            
            return True
            
        except Exception as e:
            logging.error(f"이메일 알림 오류: {e}")
            return False
    
    def _send_sms_alert(self, alert: Dict) -> bool:
        """SMS 알림 전송"""
        try:
            sms_config = self.alert_config.get('sms', {})
            
            if not sms_config.get('enable', False):
                return False
            
            # SMS API 호출 (실제로는 Twilio 등 사용)
            logging.info(f"SMS 알림 전송: {alert['title']}")
            
            return True
            
        except Exception as e:
            logging.error(f"SMS 알림 오류: {e}")
            return False
    
    def _send_webhook_alert(self, alert: Dict) -> bool:
        """웹훅 알림 전송"""
        try:
            webhook_url = self.alert_config.get('webhook_url')
            
            if not webhook_url:
                return False
            
            payload = {
                'alert': alert,
                'timestamp': datetime.now().isoformat(),
                'source': 'aquaponics_system'
            }
            
            response = requests.post(webhook_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                logging.info(f"웹훅 알림 전송 성공: {alert['title']}")
                return True
            else:
                logging.warning(f"웹훅 알림 실패: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"웹훅 알림 오류: {e}")
            return False
    
    def get_recent_alerts(self, hours: int = 24, severity: str = 'all') -> List[Dict]:
        """최근 알림 조회"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            filtered_alerts = []
            for alert in self.recent_alerts:
                alert_time = datetime.fromisoformat(alert['created_at'])
                
                if alert_time >= cutoff_time:
                    if severity == 'all' or alert['severity'] == severity:
                        filtered_alerts.append(alert)
            
            # 최신순 정렬
            filtered_alerts.sort(key=lambda x: x['created_at'], reverse=True)
            
            return filtered_alerts
            
        except Exception as e:
            logging.error(f"최근 알림 조회 오류: {e}")
            return []
    
    def get_alert_statistics(self) -> Dict:
        """알림 통계"""
        try:
            total_alerts = len(self.recent_alerts)
            
            if total_alerts == 0:
                return {
                    'total_alerts': 0,
                    'by_severity': {},
                    'by_type': {},
                    'recent_24h': 0
                }
            
            # 심각도별 통계
            by_severity = {}
            by_type = {}
            recent_24h = 0
            
            cutoff_24h = datetime.now() - timedelta(hours=24)
            
            for alert in self.recent_alerts:
                severity = alert['severity']
                alert_type = alert['type']
                alert_time = datetime.fromisoformat(alert['created_at'])
                
                by_severity[severity] = by_severity.get(severity, 0) + 1
                by_type[alert_type] = by_type.get(alert_type, 0) + 1
                
                if alert_time >= cutoff_24h:
                    recent_24h += 1
            
            return {
                'total_alerts': total_alerts,
                'by_severity': by_severity,
                'by_type': by_type,
                'recent_24h': recent_24h
            }
            
        except Exception as e:
            logging.error(f"알림 통계 오류: {e}")
            return {}
