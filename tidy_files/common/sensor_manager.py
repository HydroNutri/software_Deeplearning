# common/sensor_manager.py - 센서 관리자 공통 모듈
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import random  # 시뮬레이션용

from .data_types import SensorReading, EnvironmentData

class SensorManager:
    """센서 관리자 공통 클래스"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.sensors_enabled = config.get('sensors', {}).get('enable', False)
        self.sensor_config = config.get('sensors', {})
        self.previous_readings = {}
        self.change_threshold = 0.1  # 10% 변화 시 significant change
        
        if self.sensors_enabled:
            self.initialize_sensors()
        
        logging.info(f"센서 관리자 초기화: {'활성화' if self.sensors_enabled else '비활성화'}")
    
    def initialize_sensors(self):
        """센서 초기화"""
        try:
            # 실제 하드웨어 센서 초기화 코드
            # GPIO, I2C, SPI 등 설정
            logging.info("센서 하드웨어 초기화 완료")
            
        except Exception as e:
            logging.error(f"센서 초기화 오류: {e}")
            self.sensors_enabled = False
    
    def has_sensors(self) -> bool:
        """센서 사용 가능 여부"""
        return self.sensors_enabled
    
    def read_sensor(self, sensor_type: str, sensor_config: Dict) -> Optional[SensorReading]:
        """개별 센서 읽기"""
        try:
            # 실제 센서 읽기 코드 (시뮬레이션)
            if sensor_type == 'temperature':
                value = self.simulate_temperature()
            elif sensor_type == 'ph':
                value = self.simulate_ph()
            elif sensor_type == 'ec':
                value = self.simulate_ec()
            elif sensor_type == 'dissolved_oxygen':
                value = self.simulate_dissolved_oxygen()
            elif sensor_type == 'air_temperature':
                value = self.simulate_air_temperature()
            elif sensor_type == 'humidity':
                value = self.simulate_humidity()
            elif sensor_type == 'light_intensity':
                value = self.simulate_light_intensity()
            else:
                return None
            
            # 범위 체크
            sensor_range = sensor_config.get('range', [0, 100])
            if not (sensor_range[0] <= value <= sensor_range[1]):
                status = 'warning'
            else:
                status = 'normal'
            
            # 단위 설정
            unit_mapping = {
                'temperature': '°C',
                'air_temperature': '°C',
                'ph': 'pH',
                'ec': 'mS/cm',
                'dissolved_oxygen': 'mg/L',
                'humidity': '%',
                'light_intensity': 'lux'
            }
            
            reading = SensorReading(
                sensor_id=f"{sensor_type}_01",
                sensor_type=sensor_type,
                value=value,
                unit=unit_mapping.get(sensor_type, ''),
                timestamp=datetime.now(),
                status=status
            )
            
            return reading
            
        except Exception as e:
            logging.error(f"센서 읽기 오류 ({sensor_type}): {e}")
            return None
    
    def read_all_sensors(self) -> Optional[EnvironmentData]:
        """모든 센서 읽기"""
        if not self.sensors_enabled:
            return None
        
        try:
            readings = {}
            
            # 수질 센서들
            water_sensors = self.sensor_config.get('water', {})
            for sensor_type, sensor_config in water_sensors.items():
                reading = self.read_sensor(sensor_type, sensor_config)
                if reading:
                    readings[sensor_type] = reading.value
            
            # 환경 센서들  
            env_sensors = self.sensor_config.get('environment', {})
            for sensor_type, sensor_config in env_sensors.items():
                reading = self.read_sensor(sensor_type, sensor_config)
                if reading:
                    readings[sensor_type] = reading.value
            
            # EnvironmentData 객체 생성
            env_data = EnvironmentData(
                timestamp=datetime.now(),
                water_temp=readings.get('temperature'),
                air_temp=readings.get('air_temperature'),
                ph=readings.get('ph'),
                ec=readings.get('ec'),
                dissolved_oxygen=readings.get('dissolved_oxygen'),
                humidity=readings.get('humidity'),
                light_intensity=readings.get('light_intensity')
            )
            
            return env_data
            
        except Exception as e:
            logging.error(f"전체 센서 읽기 오류: {e}")
            return None
    
    def has_significant_change(self, current_data: EnvironmentData) -> bool:
        """의미있는 변화 감지"""
        if not self.previous_readings:
            self.previous_readings = current_data.to_dict()
            return True
        
        try:
            current_dict = current_data.to_dict()
            
            for key, current_value in current_dict.items():
                if key == 'timestamp' or current_value is None:
                    continue
                    
                previous_value = self.previous_readings.get(key)
                if previous_value is None:
                    continue
                
                # 변화율 계산
                if previous_value != 0:
                    change_rate = abs(current_value - previous_value) / abs(previous_value)
                    if change_rate > self.change_threshold:
                        self.previous_readings = current_dict
                        return True
            
            return False
            
        except Exception as e:
            logging.error(f"변화 감지 오류: {e}")
            return False
    
    # 센서 시뮬레이션 함수들 (실제 구현에서는 하드웨어 읽기로 대체)
    def simulate_temperature(self) -> float:
        return random.uniform(22.0, 26.0)
    
    def simulate_ph(self) -> float:
        return random.uniform(6.8, 7.2)
    
    def simulate_ec(self) -> float:
        return random.uniform(1.0, 1.5)
    
    def simulate_dissolved_oxygen(self) -> float:
        return random.uniform(6.0, 8.0)
    
    def simulate_air_temperature(self) -> float:
        return random.uniform(20.0, 28.0)
    
    def simulate_humidity(self) -> float:
        return random.uniform(60.0, 80.0)
    
    def simulate_light_intensity(self) -> float:
        # 시간대별 조명 강도 시뮬레이션
        hour = datetime.now().hour
        if 6 <= hour <= 18:  # 낮시간
            return random.uniform(0.7, 1.0)
        else:  # 밤시간
            return random.uniform(0.1, 0.3)
    
    def cleanup(self):
        """센서 정리"""
        logging.info("센서 관리자 정리 완료")
