# server/__init__.py - 서버 모듈 초기화
"""
아쿠아포닉스 서버 시스템 모듈

우분투 서버에서 실행되는 고정밀 분석 및 데이터 관리 시스템입니다.

주요 구성요소:
- lightweight_server: 경량화된 메인 서버
- database_manager: SQLite 데이터베이스 관리
- analysis_engine: 분석 엔진 (성장/건강 분석)
- alert_manager: 알림 관리 시스템
- server_utils: 서버 유틸리티 함수들
"""

__version__ = "2.0.0"
__author__ = "Aquaponics Development Team"

# 주요 클래스들 import
from .lightweight_server import LightweightServer
from .database_manager import DatabaseManager
from .analysis_engine import AnalysisEngine
from .alert_manager import AlertManager

__all__ = [
    'LightweightServer',
    'DatabaseManager',
    'AnalysisEngine', 
    'AlertManager'
]

# 서버 전용 로깅 설정
import logging
import os
from pathlib import Path

def setup_server_logging(device_id: str = "server_001", level=logging.INFO):
    """서버 전용 로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(f'aquaponics.server.{device_id}')
    
    if not logger.handlers:
        # 파일 핸들러 (상세 로그)
        file_handler = logging.FileHandler(f'logs/server_{device_id}.log')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # 콘솔 핸들러 (요약 로그)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # 에러 전용 파일 핸들러
        error_handler = logging.FileHandler(f'logs/server_{device_id}_error.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.addHandler(error_handler)
        logger.setLevel(level)
    
    return logger

def get_server_info():
    """서버 정보 수집"""
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
        'disk_gb': round(psutil.disk_usage('/').total / (1024**3), 1)
    }
    
    # GPU 정보 (NVIDIA)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')[0].split(', ')
            info['gpu_name'] = gpu_info[0]
            info['gpu_memory_mb'] = int(gpu_info[1])
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
        info['gpu_name'] = 'Not available'
        info['gpu_memory_mb'] = 0
    
    return info

# 기본 로거
logger = setup_server_logging()

# 서버 시작 시 정보 로깅
try:
    server_info = get_server_info()
    logger.info(f"서버 환경: {server_info['platform']}")
    logger.info(f"CPU: {server_info['cpu_count']}코어, RAM: {server_info['memory_gb']}GB")
    if server_info['gpu_memory_mb'] > 0:
        logger.info(f"GPU: {server_info['gpu_name']} ({server_info['gpu_memory_mb']}MB)")
except Exception as e:
    logger.warning(f"서버 정보 수집 실패: {e}")
