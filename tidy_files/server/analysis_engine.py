# server/analysis_engine.py - 경량화된 분석 엔진
import numpy as np
import cv2
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

import sys
sys.path.append('..')
from common.data_types import *

class AnalysisEngine:
    """경량화된 분석 엔진 - 서버 사양 고려"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.analysis_config = config.get('analysis', {})
        
        # 캐시된 분석 결과 (메모리 절약)
        self.analysis_cache = {}
        self.cache_max_size = 100
        
        # 분석 임계값들
        self.thresholds = {
            'growth_change': self.analysis_config.get('growth', {}).get('growth_alert_threshold', 0.2),
            'health_critical': 0.3,
            'sensor_anomaly': 2.0  # 표준편차 배수
        }
        
        logging.info("분석 엔진 초기화 완료 (경량 모드)")
    
    def has_sensor_anomaly(self, sensor_data: EnvironmentData) -> bool:
        """센서 이상값 간단 감지"""
        try:
            # 기본 범위 체크
            if sensor_data.water_temp:
                if sensor_data.water_temp < 15 or sensor_data.water_temp > 35:
                    return True
            
            if sensor_data.ph:
                if sensor_data.ph < 5.5 or sensor_data.ph > 8.5:
                    return True
                    
            if sensor_data.dissolved_oxygen:
                if sensor_data.dissolved_oxygen < 3.0:
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"센서 이상 감지 오류: {e}")
            return False
    
    def enhance_sensor_data(self, sensor_data: EnvironmentData) -> EnvironmentData:
        """센서 데이터 보정 (간단한 필터링)"""
        try:
            enhanced = EnvironmentData(timestamp=sensor_data.timestamp)
            
            # 단순 이상값 제거 및 보정
            for field in ['water_temp', 'air_temp', 'ph', 'ec', 'dissolved_oxygen', 
                         'humidity', 'light_intensity']:
                value = getattr(sensor_data, field)
                if value is not None:
                    # 기본 범위 체크 후 보정
                    corrected_value = self._correct_sensor_value(field, value)
                    setattr(enhanced, field, corrected_value)
            
            return enhanced
            
        except Exception as e:
            logging.error(f"센서 데이터 보정 오류: {e}")
            return sensor_data
    
    def _correct_sensor_value(self, sensor_type: str, value: float) -> float:
        """개별 센서값 보정"""
        # 센서별 유효 범위
        ranges = {
            'water_temp': (0, 40),
            'air_temp': (-10, 50),
            'ph': (0, 14),
            'ec': (0, 5),
            'dissolved_oxygen': (0, 20),
            'humidity': (0, 100),
            'light_intensity': (0, 1)
        }
        
        if sensor_type in ranges:
            min_val, max_val = ranges[sensor_type]
            return max(min_val, min(max_val, value))
        
        return value
    
    def analyze_color(self, roi: np.ndarray) -> Dict:
        """간단한 색상 분석"""
        try:
            if roi.size == 0:
                return {}
            
            # HSV 색공간 변환
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 기본 통계
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])
            
            # 색상 건강도 추정 (간단한 규칙)
            health_indicator = 1.0
            
            # 어두운 색상 감지 (병든 상태)
            if v_mean < 80:
                health_indicator *= 0.7
            
            # 채도가 너무 낮음 (창백함)
            if s_mean < 50:
                health_indicator *= 0.8
            
            return {
                'hue_mean': float(h_mean),
                'saturation_mean': float(s_mean),
                'value_mean': float(v_mean),
                'health_indicator': health_indicator
            }
            
        except Exception as e:
            logging.error(f"색상 분석 오류: {e}")
            return {}
    
    def calculate_health_score(self, obj: DetectedObject, roi: np.ndarray) -> float:
        """건강도 점수 계산 (간단한 알고리즘)"""
        try:
            health_score = 1.0  # 기본값
            
            # 색상 기반 건강도
            if roi.size > 0:
                color_analysis = self.analyze_color(roi)
                health_score *= color_analysis.get('health_indicator', 1.0)
            
            # 크기 기반 건강도 (너무 작으면 건강하지 않음)
            if obj.class_name == 'fish' and obj.length_cm:
                if obj.length_cm < 5.0:  # 5cm 미만
                    health_score *= 0.6
                elif obj.length_cm > 25.0:  # 25cm 초과는 양호
                    health_score *= 1.1
            
            # 신뢰도 기반 보정
            health_score *= obj.confidence
            
            return min(1.0, max(0.0, health_score))
            
        except Exception as e:
            logging.error(f"건강도 계산 오류: {e}")
            return 0.5
    
    def calculate_activity_level(self, obj: DetectedObject) -> float:
        """활동도 계산 (단순화)"""
        try:
            # 실제로는 이전 프레임과의 위치 비교 필요
            # 여기서는 바운딩 박스 크기와 신뢰도 기반 추정
            
            activity = 0.5  # 기본값
            
            # 바운딩 박스 크기가 클수록 활발함 (어느 정도까지)
            bbox_area = obj.bbox.area
            if bbox_area > 1000:
                activity += 0.2
            
            # 신뢰도가 높으면 잘 탐지됨 = 활발함
            activity += (obj.confidence - 0.5) * 0.5
            
            return min(1.0, max(0.0, activity))
            
        except Exception as e:
            logging.error(f"활동도 계산 오류: {e}")
            return 0.5
    
    def generate_control_commands(self, sensor_data: Dict, device_id: str) -> List[ControlCommand]:
        """센서 데이터 기반 제어 명령 생성"""
        commands = []
        
        try:
            current_time = datetime.now()
            
            # 수온 제어
            if 'water_temp' in sensor_data:
                temp = sensor_data['water_temp']
                if temp < 22.0:
                    commands.append(ControlCommand(
                        command_id=generate_event_id(),
                        device_id='water_heater',
                        command_type='pwm',
                        target_value=0.7,
                        duration=300,
                        priority=2,
                        timestamp=current_time
                    ))
                elif temp > 26.0:
                    commands.append(ControlCommand(
                        command_id=generate_event_id(),
                        device_id='water_heater',
                        command_type='pwm',
                        target_value=0.0,
                        duration=60,
                        priority=2,
                        timestamp=current_time
                    ))
            
            # pH 제어
            if 'ph' in sensor_data:
                ph = sensor_data['ph']
                if ph < 6.8:
                    commands.append(ControlCommand(
                        command_id=generate_event_id(),
                        device_id='ph_dosing',
                        command_type='relay',
                        target_value=1.0,
                        duration=5,
                        priority=2,
                        timestamp=current_time
                    ))
            
            # 용존산소 제어
            if 'dissolved_oxygen' in sensor_data:
                do = sensor_data['dissolved_oxygen']
                if do < 6.0:
                    commands.append(ControlCommand(
                        command_id=generate_event_id(),
                        device_id='water_pump',
                        command_type='pwm',
                        target_value=0.8,
                        duration=600,
                        priority=3,
                        timestamp=current_time
                    ))
            
            return commands
            
        except Exception as e:
            logging.error(f"제어 명령 생성 오류: {e}")
            return []
    
    def get_object_tracking(self, object_type: str, hours: int) -> List[Dict]:
        """객체 추적 정보 (단순화된 버전)"""
        try:
            # 실제로는 데이터베이스에서 조회
            # 여기서는 시뮬레이션 데이터 반환
            
            tracking_data = []
            
            if object_type == 'fish' or object_type == 'all':
                for i in range(5):  # 5마리 시뮬레이션
                    tracking_data.append({
                        'object_id': f'fish_{i+1}',
                        'class_name': 'fish',
                        'length_cm': 12.0 + i * 2.5,
                        'weight_g': 15.0 + i * 8.0,
                        'health_score': 0.7 + i * 0.05,
                        'last_seen': datetime.now().isoformat()
                    })
            
            if object_type == 'plant' or object_type == 'all':
                for i in range(3):  # 3개 식물 시뮬레이션
                    tracking_data.append({
                        'object_id': f'plant_{i+1}',
                        'class_name': 'plant',
                        'height_cm': 20.0 + i * 5.0,
                        'health_score': 0.8 + i * 0.03,
                        'last_seen': datetime.now().isoformat()
                    })
            
            return tracking_data
            
        except Exception as e:
            logging.error(f"객체 추적 조회 오류: {e}")
            return []
    
    def get_environment_trends(self, hours: int) -> List[Dict]:
        """환경 트렌드 (단순화된 시뮬레이션)"""
        try:
            trends = []
            current_time = datetime.now()
            
            # 최근 시간대별 데이터 시뮬레이션
            for i in range(min(hours, 24)):  # 최대 24시간
                timestamp = current_time - timedelta(hours=i)
                
                # 시간대별 변화 시뮬레이션
                hour = timestamp.hour
                
                trends.append({
                    'timestamp': timestamp.isoformat(),
                    'water_temp': 24.0 + np.sin(hour * np.pi / 12) * 2.0,
                    'ph': 7.0 + np.random.normal(0, 0.1),
                    'dissolved_oxygen': 7.5 + np.random.normal(0, 0.5),
                    'light_intensity': 0.8 if 6 <= hour <= 18 else 0.2
                })
            
            return trends[::-1]  # 시간순 정렬
            
        except Exception as e:
            logging.error(f"환경 트렌드 조회 오류: {e}")
            return []
    
    def analyze_growth_trends(self) -> Optional[Dict]:
        """성장 트렌드 분석 (간단한 버전)"""
        try:
            # 실제로는 시계열 데이터 분석
            # 여기서는 기본 통계만 계산
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'fish_trends': {
                    'avg_growth_rate': 0.1,  # cm/day
                    'growth_trend': 'stable',
                    'health_trend': 'improving'
                },
                'plant_trends': {
                    'avg_growth_rate': 0.3,  # cm/day
                    'growth_trend': 'increasing',
                    'health_trend': 'stable'
                }
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"성장 트렌드 분석 오류: {e}")
            return None
    
    def analyze_environment_correlations(self) -> Optional[Dict]:
        """환경 상관관계 분석 (간단한 버전)"""
        try:
            # 간단한 상관관계 시뮬레이션
            correlations = {
                'timestamp': datetime.now().isoformat(),
                'temperature_growth': 0.65,  # 온도와 성장률 상관관계
                'ph_health': 0.72,           # pH와 건강도 상관관계
                'light_plant_growth': 0.88,  # 조명과 식물성장 상관관계
                'oxygen_fish_activity': 0.91 # 용존산소와 물고기 활동도
            }
            
            return correlations
            
        except Exception as e:
            logging.error(f"환경 상관관계 분석 오류: {e}")
            return None
    
    def detect_anomaly_patterns(self) -> List[Dict]:
        """이상 패턴 감지 (간단한 규칙 기반)"""
        try:
            anomalies = []
            
            # 현재 시뮬레이션에서는 랜덤 이상 패턴 생성
            if np.random.random() < 0.1:  # 10% 확률로 이상 패턴
                anomalies.append({
                    'type': 'temperature_spike',
                    'severity': 'medium',
                    'timestamp': datetime.now().isoformat(),
                    'description': '수온 급상승 감지',
                    'recommended_action': '히터 점검 및 순환 강화'
                })
            
            if np.random.random() < 0.05:  # 5% 확률로 심각한 이상
                anomalies.append({
                    'type': 'health_decline',
                    'severity': 'high',
                    'timestamp': datetime.now().isoformat(),
                    'description': '다수 개체 건강도 저하',
                    'recommended_action': '즉시 수질 점검 필요'
                })
            
            return anomalies
            
        except Exception as e:
            logging.error(f"이상 패턴 감지 오류: {e}")
            return []
    
    def get_growth_summary(self, cutoff_time: datetime) -> Dict:
        """성장 요약 정보"""
        try:
            # 간단한 요약 통계
            summary = {
                'avg_fish_growth_rate': 0.12,  # cm/day
                'avg_plant_growth_rate': 0.28,  # cm/day
                'total_fish_detected': 5,
                'total_plants_detected': 3,
                'growth_acceleration': 0.05  # 증가율
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"성장 요약 오류: {e}")
            return {}
    
    def get_health_summary(self, cutoff_time: datetime) -> Dict:
        """건강 요약 정보"""
        try:
            summary = {
                'avg_fish_health': 0.82,
                'avg_plant_health': 0.85,
                'healthy_fish_count': 4,
                'healthy_plant_count': 3,
                'critical_health_alerts': 0
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"건강 요약 오료: {e}")
            return {}
    
    def generate_insights(self, event: SystemEvent) -> Optional[Dict]:
        """이벤트 기반 인사이트 생성"""
        try:
            insights = {
                'timestamp': datetime.now().isoformat(),
                'event_id': event.event_id,
                'insights': []
            }
            
            # YOLO 결과 기반 인사이트
            if event.yolo_result and event.yolo_result.objects:
                fish_count = len([obj for obj in event.yolo_result.objects if obj.class_name == 'fish'])
                plant_count = len([obj for obj in event.yolo_result.objects if obj.class_name == 'plant'])
                
                if fish_count > 0:
                    avg_health = np.mean([obj.health_score or 0.7 for obj in event.yolo_result.objects 
                                        if obj.class_name == 'fish'])
                    
                    if avg_health < 0.6:
                        insights['insights'].append({
                            'type': 'health_warning',
                            'message': f'물고기 평균 건강도 낮음: {avg_health:.2f}',
                            'recommendation': '수질 점검 및 사료 조정 필요'
                        })
                
                if plant_count == 0:
                    insights['insights'].append({
                        'type': 'ecosystem_balance',
                        'message': '식물이 감지되지 않음',
                        'recommendation': '식물 상태 점검 또는 추가 필요'
                    })
            
            # 센서 데이터 기반 인사이트
            if event.sensors:
                if event.sensors.water_temp and event.sensors.water_temp > 27:
                    insights['insights'].append({
                        'type': 'temperature_alert',
                        'message': f'수온 높음: {event.sensors.water_temp:.1f}°C',
                        'recommendation': '냉각 시스템 가동 또는 환기 강화'
                    })
                
                if event.sensors.ph and (event.sensors.ph < 6.5 or event.sensors.ph > 7.5):
                    insights['insights'].append({
                        'type': 'ph_alert',
                        'message': f'pH 이상: {event.sensors.ph:.1f}',
                        'recommendation': 'pH 조정제 투입 필요'
                    })
            
            return insights if insights['insights'] else None
            
        except Exception as e:
            logging.error(f"인사이트 생성 오류: {e}")
            return None
    
    def _manage_cache(self):
        """캐시 크기 관리"""
        if len(self.analysis_cache) > self.cache_max_size:
            # 가장 오래된 항목 제거
            oldest_key = min(self.analysis_cache.keys(), 
                           key=lambda k: self.analysis_cache[k].get('timestamp', 0))
            del self.analysis_cache[oldest_key]
