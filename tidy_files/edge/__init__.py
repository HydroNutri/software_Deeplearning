# edge/__init__.py - Edge 모듈 초기화
"""
아쿠아포닉스 Edge 디바이스 모듈

Raspberry Pi나 Jetson Nano 등의 Edge 디바이스에서 실행되는
실시간 모니터링 및 제어 시스템입니다.

주요 구성요소:
- edge_controller: 메인 Edge 컨트롤러
- control_policy: 제어 정책 및 하드웨어 제어
- camera_manager: 카메라 스트림 관리
- gpio_manager: GPIO 제어 관리
"""

__version__ = "2.0.0"
__author__ = "Aquaponics Development Team"

# 주요 클래스들 import
from .edge_controller import EdgeController
from .control_policy import ControlPolicy
from .camera_manager import CameraManager

__all__ = [
    'EdgeController',
    'ControlPolicy', 
    'CameraManager'
]

# Edge 전용 로깅 설정
import logging

def setup_edge_logging(device_id: str = "edge_001", level=logging.INFO):
    """Edge 전용 로깅 설정"""
    logger = logging.getLogger(f'aquaponics.edge.{device_id}')
    
    if not logger.handlers:
        # 파일 핸들러
        file_handler = logging.FileHandler(f'logs/edge_{device_id}.log')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(level)
    
    return logger

# 하드웨어 감지 함수
def detect_hardware():
    """하드웨어 플랫폼 감지"""
    try:
        # Raspberry Pi 감지
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip()
            if 'Raspberry Pi' in model:
                return {
                    'platform': 'raspberry_pi',
                    'model': model,
                    'gpio_available': True
                }
    except FileNotFoundError:
        pass
    
    try:
        # Jetson 감지
        with open('/etc/nv_tegra_release', 'r') as f:
            if 'tegra' in f.read().lower():
                return {
                    'platform': 'jetson',
                    'model': 'NVIDIA Jetson',
                    'gpio_available': True
                }
    except FileNotFoundError:
        pass
    
    # 일반 Linux
    return {
        'platform': 'linux',
        'model': 'Generic Linux',
        'gpio_available': False
    }

# 기본 하드웨어 정보
HARDWARE_INFO = detect_hardware()

# 기본 로거
logger = setup_edge_logging()
