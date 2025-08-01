# edge/control_policy.py - 제어 정책 모듈
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json

# 공통 모듈 import
import sys
sys.path.append('..')
from common.data_types import ControlCommand, EnvironmentData, YOLOResult, generate_event_id

class ControlPolicy:
    """제어 정책 클래스 - 규칙 기반 제어 시스템"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.control_config = config.get('control', {})
        self.analysis_config = config.get('analysis', {})
        
        # 제어 상태 추적
        self.last_control_times = {}
        self.control_cooldowns = {
            'water_pump': 60,      # 1분
            'water_heater': 300,   # 5분  
            'ph_dosing': 600,      # 10분
            'fish_feeder': 3600,   # 1시간
            'led_lights': 1800     # 30분
        }
        
        # 제어 임계값
        self.thresholds = {
            'temperature': {'min': 22.0, 'max': 26.0, 'optimal': 24.0},
            'ph': {'min': 6.8, 'max': 7.2, 'optimal': 7.0},
            'dissolved_oxygen': {'min': 6.0, 'max': 10.0, 'optimal': 8.0},
            'light_intensity': {'min': 0.7, 'max': 1.0, 'optimal': 0.8}
        }
        
        # 하드웨어 초기화 (GPIO 등)
        self.initialize_hardware()
        
        logging.info("제어 정책 초기화 완료")
    
    def initialize_hardware(self):
        """하드웨어 제어 초기화"""
        try:
            # GPIO 초기화 (실제 구현에서는 RPi.GPIO 사용)
            # import RPi.GPIO as GPIO
            # GPIO.setmode(GPIO.BCM)
            
            # 릴레이/PWM 핀 설정
            self.control_pins = {}
            for device, pin_config in self.control_config.items():
                if isinstance(pin_config, dict) and 'pin' in pin_config:
                    pin_num = pin_config['pin']
                    self.control_pins[device] = pin_num
                    # GPIO.setup(pin_num, GPIO.OUT)
            
            logging.info("하드웨어 제어 초기화 완료")
            
        except Exception as e:
            logging.error(f"하드웨어 제어 초기화 오류: {e}")
    
    def generate_commands(self, sensor_data: EnvironmentData, 
                         yolo_result: YOLOResult) -> List[ControlCommand]:
        """센서 데이터와 비전 결과를 기반으로 제어 명령 생성"""
        commands = []
        
        try:
            # 환경 센서 기반 제어
            if sensor_data:
                commands.extend(self._generate_environment_commands(sensor_data))
            
            # 비전 결과 기반 제어
            if yolo_result and yolo_result.objects:
                commands.extend(self._generate_vision_commands(yolo_result))
            
            # 스케줄 기반 제어
            commands.extend(self._generate_scheduled_commands())
            
            # 응급 상황 제어
            emergency_commands = self._generate_emergency_commands(sensor_data, yolo_result)
            if emergency_commands:
                commands.extend(emergency_commands)
            
            return commands
            
        except Exception as e:
            logging.error(f"제어 명령 생성 오류: {e}")
            return []
    
    def _generate_environment_commands(self, sensor_data: EnvironmentData) -> List[ControlCommand]:
        """환경 센서 기반 제어 명령"""
        commands = []
        
        try:
            # 수온 제어
            if sensor_data.water_temp is not None:
                temp_commands = self._control_temperature(sensor_data.water_temp)
                commands.extend(temp_commands)
            
            # pH 제어
            if sensor_data.ph is not None:
                ph_commands = self._control_ph(sensor_data.ph)
                commands.extend(ph_commands)
            
            # 용존산소 제어 (수중펌프)
            if sensor_data.dissolved_oxygen is not None:
                oxygen_commands = self._control_oxygen(sensor_data.dissolved_oxygen)
                commands.extend(oxygen_commands)
            
            # 조명 제어
            if sensor_data.light_intensity is not None:
                light_commands = self._control_lighting(sensor_data.light_intensity)
                commands.extend(light_commands)
                
        except Exception as e:
            logging.error(f"환경 제어 명령 생성 오류: {e}")
        
        return commands
    
    def _control_temperature(self, current_temp: float) -> List[ControlCommand]:
        """수온 제어"""
        commands = []
        thresholds = self.thresholds['temperature']
        
        if not self._can_control('water_heater'):
            return commands
        
        if current_temp < thresholds['min']:
            # 히터 켜기
            power_level = min(1.0, (thresholds['optimal'] - current_temp) * 0.5)
            
            command = ControlCommand(
                command_id=generate_event_id(),
                device_id='water_heater',
                command_type='pwm',
                target_value=power_level,
                priority=2,
                duration=300  # 5분
            )
            commands.append(command)
            
        elif current_temp > thresholds['max']:
            # 히터 끄기
            command = ControlCommand(
                command_id=generate_event_id(),
                device_id='water_heater',
                command_type='pwm',
                target_value=0.0,
                priority=2,
                duration=300
            )
            commands.append(command)
        
        return commands
    
    def _control_ph(self, current_ph: float) -> List[ControlCommand]:
        """pH 제어"""
        commands = []
        thresholds = self.thresholds['ph']
        
        if not self._can_control('ph_dosing'):
            return commands
        
        if current_ph < thresholds['min']:
            # pH 상승제 투입
            dose_time = min(10, (thresholds['optimal'] - current_ph) * 5)
            
            command = ControlCommand(
                command_id=generate_event_id(),
                device_id='ph_dosing',
                command_type='relay',
                target_value=1.0,
                priority=2,
                duration=int(dose_time)
            )
            commands.append(command)
            
        elif current_ph > thresholds['max']:
            # pH 하강제 투입 (별도 펌프가 있다면)
            dose_time = min(10, (current_ph - thresholds['optimal']) * 5)
            
            command = ControlCommand(
                command_id=generate_event_id(),
                device_id='ph_dosing_down',
                command_type='relay',
                target_value=1.0,
                priority=2,
                duration=int(dose_time)
            )
            commands.append(command)
        
        return commands
    
    def _control_oxygen(self, current_oxygen: float) -> List[ControlCommand]:
        """용존산소 제어"""
        commands = []
        thresholds = self.thresholds['dissolved_oxygen']
        
        if not self._can_control('water_pump'):
            return commands
        
        if current_oxygen < thresholds['min']:
            # 에어펌프/순환펌프 강화
            pump_power = min(1.0, (thresholds['optimal'] - current_oxygen) * 0.3)
            
            command = ControlCommand(
                command_id=generate_event_id(),
                device_id='water_pump',
                command_type='pwm',
                target_value=pump_power,
                priority=3,  # 높은 우선순위
                duration=600  # 10분
            )
            commands.append(command)
        
        return commands
    
    def _control_lighting(self, current_light: float) -> List[ControlCommand]:
        """조명 제어"""
        commands = []
        
        if not self._can_control('led_lights'):
            return commands
        
        # 시간대별 조명 제어
        current_hour = datetime.now().hour
        schedule = self.control_config.get('led_lights', {}).get('schedule', {})
        
        on_time = schedule.get('on_time', '06:00')
        off_time = schedule.get('off_time', '22:00')
        
        on_hour = int(on_time.split(':')[0])
        off_hour = int(off_time.split(':')[0])
        
        target_intensity = 0.0
        if on_hour <= current_hour < off_hour:
            target_intensity = 0.8  # 낮시간 조명
        
        # 현재 조명이 목표와 다른 경우 조정
        if abs(current_light - target_intensity) > 0.2:
            command = ControlCommand(
                command_id=generate_event_id(),
                device_id='led_lights',
                command_type='pwm',
                target_value=target_intensity,
                priority=1,
                duration=1800  # 30분
            )
            commands.append(command)
        
        return commands
    
    def _generate_vision_commands(self, yolo_result: YOLOResult) -> List[ControlCommand]:
        """비전 결과 기반 제어 명령"""
        commands = []
        
        try:
            fish_count = len([obj for obj in yolo_result.objects if obj.class_name == 'fish'])
            plant_count = len([obj for obj in yolo_result.objects if obj.class_name == 'plant'])
            
            # 물고기 활동도 기반 사료 공급
            if fish_count > 0:
                avg_activity = sum(obj.activity_level or 0.5 for obj in yolo_result.objects 
                                 if obj.class_name == 'fish') / fish_count
                
                if avg_activity > 0.7 and self._can_control('fish_feeder'):
                    # 활발한 활동 시 사료 공급
                    feed_amount = min(1.0, fish_count * 0.1)
                    
                    command = ControlCommand(
                        command_id=generate_event_id(),
                        device_id='fish_feeder',
                        command_type='servo',
                        target_value=feed_amount,
                        priority=1,
                        duration=5
                    )
                    commands.append(command)
            
            # 건강도 기반 응급 조치
            unhealthy_objects = [obj for obj in yolo_result.objects 
                               if obj.health_score and obj.health_score < 0.5]
            
            if unhealthy_objects:
                # 순환펌프 강화로 수질 개선
                command = ControlCommand(
                    command_id=generate_event_id(),
                    device_id='water_pump',
                    command_type='pwm',
                    target_value=0.8,
                    priority=3,
                    duration=1800
                )
                commands.append(command)
                
        except Exception as e:
            logging.error(f"비전 기반 제어 명령 생성 오류: {e}")
        
        return commands
    
    def _generate_scheduled_commands(self) -> List[ControlCommand]:
        """스케줄 기반 제어 명령"""
        commands = []
        
        try:
            current_time = datetime.now()
            current_time_str = current_time.strftime('%H:%M')
            
            # 사료 공급 스케줄
            feeder_schedule = self.control_config.get('fish_feeder', {}).get('schedule', [])
            
            if current_time_str in feeder_schedule and self._can_control('fish_feeder'):
                command = ControlCommand(
                    command_id=generate_event_id(),
                    device_id='fish_feeder',
                    command_type='servo',
                    target_value=0.5,  # 중간 크기 사료 공급
                    priority=1,
                    duration=3
                )
                commands.append(command)
                
        except Exception as e:
            logging.error(f"스케줄 기반 제어 명령 생성 오류: {e}")
        
        return commands
    
    def _generate_emergency_commands(self, sensor_data: EnvironmentData, 
                                   yolo_result: YOLOResult) -> List[ControlCommand]:
        """응급 상황 제어 명령"""
        commands = []
        
        try:
            emergency_detected = False
            
            # 환경 응급상황
            if sensor_data:
                if (sensor_data.water_temp and 
                    (sensor_data.water_temp < 18 or sensor_data.water_temp > 30)):
                    emergency_detected = True
                    
                if (sensor_data.ph and 
                    (sensor_data.ph < 6.0 or sensor_data.ph > 8.0)):
                    emergency_detected = True
                    
                if (sensor_data.dissolved_oxygen and 
                    sensor_data.dissolved_oxygen < 4.0):
                    emergency_detected = True
            
            # 생물 건강 응급상황
            if yolo_result:
                critical_health_count = len([obj for obj in yolo_result.objects 
                                           if obj.health_score and obj.health_score < 0.3])
                if critical_health_count > 0:
                    emergency_detected = True
            
            # 응급 조치 실행
            if emergency_detected:
                # 모든 순환시스템 최대 가동
                emergency_commands = [
                    ControlCommand(
                        command_id=generate_event_id(),
                        device_id='water_pump',
                        command_type='pwm',
                        target_value=1.0,
                        priority=3,
                        duration=3600
                    ),
                    # 응급 알림 (실제로는 별도 알림 시스템으로)
                ]
                
                commands.extend(emergency_commands)
                
        except Exception as e:
            logging.error(f"응급 제어 명령 생성 오류: {e}")
        
        return commands
    
    def _can_control(self, device_id: str) -> bool:
        """제어 쿨다운 체크"""
        last_time = self.last_control_times.get(device_id)
        if last_time is None:
            return True
        
        cooldown = self.control_cooldowns.get(device_id, 60)
        return (datetime.now() - last_time).total_seconds() >= cooldown
    
    def execute_command(self, command: ControlCommand) -> bool:
        """제어 명령 실행"""
        try:
            device_id = command.device_id
            command_type = command.command_type
            target_value = command.target_value
            
            logging.info(f"제어 실행: {device_id} = {target_value} ({command_type})")
            
            # 실제 하드웨어 제어 (시뮬레이션)
            success = self._execute_hardware_command(device_id, command_type, target_value)
            
            if success:
                # 마지막 제어 시간 업데이트
                self.last_control_times[device_id] = datetime.now()
                
                # 지연 제어 (duration이 있는 경우)
                if command.duration:
                    self._schedule_delayed_stop(device_id, command_type, command.duration)
            
            return success
            
        except Exception as e:
            logging.error(f"제어 명령 실행 오류: {e}")
            return False
    
    def _execute_hardware_command(self, device_id: str, command_type: str, 
                                value: float) -> bool:
        """실제 하드웨어 제어"""
        try:
            pin = self.control_pins.get(device_id)
            if pin is None:
                logging.warning(f"알 수 없는 장치: {device_id}")
                return False
            
            if command_type == 'relay':
                # 릴레이 제어 (ON/OFF)
                # GPIO.output(pin, GPIO.HIGH if value > 0.5 else GPIO.LOW)
                logging.info(f"릴레이 제어: PIN {pin} = {'ON' if value > 0.5 else 'OFF'}")
                
            elif command_type == 'pwm':
                # PWM 제어 (0-100%)
                # pwm = GPIO.PWM(pin, 1000)  # 1kHz
                # pwm.start(value * 100)
                logging.info(f"PWM 제어: PIN {pin} = {value * 100:.1f}%")
                
            elif command_type == 'servo':
                # 서보 제어 (각도)
                angle = value * 180  # 0-1을 0-180도로 변환
                # servo_control(pin, angle)
                logging.info(f"서보 제어: PIN {pin} = {angle:.1f}도")
                
            else:
                logging.warning(f"지원하지 않는 제어 타입: {command_type}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"하드웨어 제어 오류: {e}")
            return False
    
    def _schedule_delayed_stop(self, device_id: str, command_type: str, duration: int):
        """지연 정지 스케줄링"""
        def delayed_stop():
            time.sleep(duration)
            try:
                self._execute_hardware_command(device_id, command_type, 0.0)
                logging.info(f"지연 정지 실행: {device_id}")
            except Exception as e:
                logging.error(f"지연 정지 오류: {e}")
        
        import threading
        stop_thread = threading.Thread(target=delayed_stop, daemon=True)
        stop_thread.start()
    
    def get_control_status(self) -> Dict:
        """현재 제어 상태 조회"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'last_control_times': {
                device: time.isoformat() for device, time in self.last_control_times.items()
            },
            'available_controls': list(self.control_pins.keys()),
            'thresholds': self.thresholds,
            'cooldown_remaining': {}
        }
        
        # 남은 쿨다운 시간 계산
        current_time = datetime.now()
        for device_id, last_time in self.last_control_times.items():
            cooldown = self.control_cooldowns.get(device_id, 60)
            elapsed = (current_time - last_time).total_seconds()
            remaining = max(0, cooldown - elapsed)
            status['cooldown_remaining'][device_id] = remaining
        
        return status
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # GPIO 정리
            # GPIO.cleanup()
            logging.info("제어 정책 정리 완료")
        except Exception as e:
            logging.error(f"제어 정책 정리 오류: {e}")
