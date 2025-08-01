# common/__init__.py - 공통 모듈 초기화
"""
아쿠아포닉스 시스템 공통 모듈

이 모듈은 Edge 디바이스와 Server 간에 공유되는 
공통 기능들을 제공합니다.

주요 구성요소:
- data_types: 데이터 구조 정의
- mqtt_client: MQTT 통신 클라이언트
- yolo_detector: YOLO 객체 탐지기
- sensor_manager: 센서 관리자
"""

__version__ = "2.0.0"
__author__ = "Aquaponics Development Team"

# 주요 클래스들을 패키지 레벨에서 import 가능하게 함
from .data_types import (
    SystemEvent,
    DetectedObject,
    BoundingBox,
    SensorReading,
    EnvironmentData,
    YOLOResult,
    ControlCommand,
    HealthMetrics,
    generate_event_id,
    generate_frame_id
)

from .mqtt_client import MQTTClient
from .yolo_detector import YOLODetector
from .sensor_manager import SensorManager

__all__ = [
    # Data types
    'SystemEvent',
    'DetectedObject', 
    'BoundingBox',
    'SensorReading',
    'EnvironmentData',
    'YOLOResult',
    'ControlCommand',
    'HealthMetrics',
    'generate_event_id',
    'generate_frame_id',
    
    # Managers
    'MQTTClient',
    'YOLODetector',
    'SensorManager'
]

# 로깅 설정
import logging

def setup_common_logging(level=logging.INFO):
    """공통 모듈 로깅 설정"""
    logger = logging.getLogger('aquaponics.common')
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger

# 기본 로거
logger = setup_common_logging()
