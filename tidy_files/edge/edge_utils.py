# edge/edge_utils.py - Edge 디바이스 유틸리티 함수들
import os
import sys
import time
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_system_info() -> Dict:
    """시스템 정보 수집"""
    info = {
        'timestamp': datetime.now().isoformat(),
        'platform': sys.platform,
        'python_version': sys.version,
        'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    }
    
    try:
        # CPU 정보
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = f.read()
            if 'model name' in cpu_info:
                for line in cpu_info.split('\n'):
                    if 'model name' in line:
                        info['cpu'] = line.split(':')[1].strip()
                        break
        
        # 메모리 정보
        with open('/proc/meminfo', 'r') as f:
            mem_info = f.read()
            for line in mem_info.split('\n'):
                if 'MemTotal' in line:
                    total_kb = int(line.split()[1])
                    info['memory_mb'] = total_kb // 1024
                    break
        
        # 라즈베리파이 모델 정보
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                info['hardware_model'] = f.read().strip('\x00')
        
        # 온도 정보 (라즈베리파이)
        if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millicelsius = int(f.read().strip())
                info['cpu_temperature'] = temp_millicelsius / 1000.0
                
    except Exception as e:
        logger.warning(f"시스템 정보 수집 중 오류: {e}")
    
    return info

def get_performance_stats() -> Dict:
    """성능 통계 수집"""
    stats = {
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # CPU 사용률
        import psutil
        stats['cpu_percent'] = psutil.cpu_percent(interval=1)
        stats['cpu_count'] = psutil.cpu_count()
        
        # 메모리 사용률
        memory = psutil.virtual_memory()
        stats['memory_percent'] = memory.percent
        stats['memory_available_mb'] = memory.available // (1024 * 1024)
        
        # 디스크 사용률
        disk = psutil.disk_usage('/')
        stats['disk_percent'] = (disk.used / disk.total) * 100
        stats['disk_free_mb'] = disk.free // (1024 * 1024)
        
        # 네트워크 통계
        net_io = psutil.net_io_counters()
        stats['network'] = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
    except ImportError:
        # psutil이 없는 경우 기본적인 정보만
        try:
            # 로드 애버리지
            with open('/proc/loadavg', 'r') as f:
                load_avg = f.read().strip().split()
                stats['load_average'] = [float(load_avg[0]), float(load_avg[1]), float(load_avg[2])]
                
            # 메모리 정보
            with open('/proc/meminfo', 'r') as f:
                mem_info = {}
                for line in f:
                    key, value = line.split(':')
                    mem_info[key.strip()] = int(value.strip().split()[0])
                
                total = mem_info['MemTotal']
                available = mem_info.get('MemAvailable', mem_info['MemFree'])
                stats['memory_percent'] = ((total - available) / total) * 100
                stats['memory_available_mb'] = available // 1024
                
        except Exception as e:
            logger.warning(f"성능 통계 수집 중 오류: {e}")
    
    return stats

def check_camera_devices() -> List[Dict]:
    """카메라 디바이스 체크"""
    cameras = []
    
    try:
        # v4l2 디바이스 체크
        video_devices = list(Path('/dev').glob('video*'))
        
        for device in video_devices:
            try:
                # 디바이스 정보 수집
                result = subprocess.run(
                    ['v4l2-ctl', '--device', str(device), '--info'],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0:
                    cameras.append({
                        'device': str(device),
                        'type': 'v4l2',
                        'info': result.stdout,
                        'available': True
                    })
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                cameras.append({
                    'device': str(device),
                    'type': 'unknown',
                    'info': 'Unable to query device info',
                    'available': False
                })
        
        # 라즈베리파이 카메라 체크
        try:
            result = subprocess.run(['vcgencmd', 'get_camera'], 
                                  capture_output=True, text=True, timeout=5)
            if 'detected=1' in result.stdout:
                cameras.append({
                    'device': 'raspberry_pi_camera',
                    'type': 'csi',
                    'info': result.stdout.strip(),
                    'available': True
                })
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
    except Exception as e:
        logger.error(f"카메라 디바이스 체크 실패: {e}")
    
    return cameras

def check_gpio_permissions() -> bool:
    """GPIO 권한 체크"""
    try:
        import grp
        import os
        
        # 현재 사용자의 그룹 확인
        groups = [g.gr_name for g in grp.getgrall() if os.getlogin() in g.gr_mem]
        primary_group = grp.getgrgid(os.getgid()).gr_name
        
        all_groups = groups + [primary_group]
        
        return 'gpio' in all_groups
        
    except Exception as e:
        logger.warning(f"GPIO 권한 체크 실패: {e}")
        return False

def optimize_system_performance():
    """시스템 성능 최적화"""
    optimizations = []
    
    try:
        # CPU 거버너 설정 (성능 모드)
        cpu_dirs = list(Path('/sys/devices/system/cpu').glob('cpu[0-9]*'))
        for cpu_dir in cpu_dirs:
            governor_file = cpu_dir / 'cpufreq/scaling_governor'
            if governor_file.exists():
                try:
                    with open(governor_file, 'w') as f:
                        f.write('performance')
                    optimizations.append(f"CPU {cpu_dir.name} 성능 모드 설정")
                except PermissionError:
                    logger.warning(f"CPU 거버너 설정 권한 없음: {cpu_dir.name}")
        
        # GPU 메모리 분할 확인 (라즈베리파이)
        if os.path.exists('/boot/config.txt'):
            try:
                with open('/boot/config.txt', 'r') as f:
                    config_content = f.read()
                    
                if 'gpu_mem=' not in config_content:
                    logger.info("GPU 메모리 설정 권장: /boot/config.txt에 gpu_mem=128 추가")
                    optimizations.append("GPU 메모리 설정 권장")
                    
            except Exception as e:
                logger.warning(f"부트 설정 확인 실패: {e}")
        
        # 스왑 설정 체크
        try:
            with open('/proc/swaps', 'r') as f:
                swaps = f.read()
                if not swaps.strip() or 'partition' not in swaps:
                    logger.info("스왑 파일 설정 권장")
                    optimizations.append("스왑 파일 설정 권장")
                    
        except Exception as e:
            logger.warning(f"스왑 설정 확인 실패: {e}")
        
        # I/O 스케줄러 최적화
        try:
            block_devices = list(Path('/sys/block').glob('mmcblk*'))  # SD 카드
            for device in block_devices:
                scheduler_file = device / 'queue/scheduler'
                if scheduler_file.exists():
                    try:
                        with open(scheduler_file, 'r') as f:
                            current = f.read().strip()
                            if 'deadline' not in current or '[deadline]' not in current:
                                logger.info(f"I/O 스케줄러 최적화 권장: {device.name}")
                                optimizations.append(f"I/O 스케줄러 최적화: {device.name}")
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.warning(f"I/O 스케줄러 확인 실패: {e}")
            
    except Exception as e:
        logger.error(f"시스템 최적화 체크 실패: {e}")
    
    return optimizations

def create_system_service(service_name: str, exec_path: str, 
                         working_dir: str, user: str = "pi") -> str:
    """systemd 서비스 파일 생성"""
    service_content = f"""[Unit]
Description=Aquaponics {service_name.title()}
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={working_dir}
Environment=PATH={working_dir}/venv/bin
ExecStart={working_dir}/venv/bin/python {exec_path}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    return service_content

def install_system_service(service_name: str, service_content: str) -> bool:
    """시스템 서비스 설치"""
    try:
        service_file = f"/etc/systemd/system/aquaponics-{service_name}.service"
        
        # 서비스 파일 작성
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        # systemd 리로드
        subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)
        
        # 서비스 활성화
        subprocess.run(['sudo', 'systemctl', 'enable', f'aquaponics-{service_name}'], check=True)
        
        logger.info(f"시스템 서비스 설치 완료: aquaponics-{service_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"시스템 서비스 설치 실패: {e}")
        return False
    except PermissionError:
        logger.error("시스템 서비스 설치 권한 없음 (sudo 필요)")
        return False

def monitor_system_resources(callback=None, interval: int = 30):
    """시스템 리소스 모니터링"""
    def monitor_loop():
        while True:
            try:
                stats = get_performance_stats()
                
                # 임계값 체크
                warnings = []
                
                if stats.get('cpu_percent', 0) > 80:
                    warnings.append(f"CPU 사용률 높음: {stats['cpu_percent']:.1f}%")
                
                if stats.get('memory_percent', 0) > 85:
                    warnings.append(f"메모리 사용률 높음: {stats['memory_percent']:.1f}%")
                
                if stats.get('disk_percent', 0) > 90:
                    warnings.append(f"디스크 사용률 높음: {stats['disk_percent']:.1f}%")
                
                # 온도 체크 (라즈베리파이)
                system_info = get_system_info()
                if 'cpu_temperature' in system_info:
                    temp = system_info['cpu_temperature']
                    if temp > 70:  # 70도 이상
                        warnings.append(f"CPU 온도 높음: {temp:.1f}°C")
                
                # 경고가 있으면 로그
                for warning in warnings:
                    logger.warning(warning)
                
                # 콜백 함수 호출
                if callback:
                    callback(stats, warnings)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"리소스 모니터링 오류: {e}")
                time.sleep(interval)
    
    # 백그라운드 스레드로 실행
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    return monitor_thread

def setup_watchdog(pid_file: str = "/tmp/aquaponics_edge.pid", 
                  check_interval: int = 60):
    """프로세스 와치독 설정"""
    def watchdog_loop():
        while True:
            try:
                # PID 파일 작성
                with open(pid_file, 'w') as f:
                    f.write(str(os.getpid()))
                
                # 시스템 상태 체크
                stats = get_performance_stats()
                
                # 메모리 사용률이 너무 높으면 가비지 컬렉션
                if stats.get('memory_percent', 0) > 90:
                    import gc
                    gc.collect()
                    logger.info("메모리 정리 수행")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"와치독 오류: {e}")
                time.sleep(check_interval)
    
    watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
    watchdog_thread.start()
    return watchdog_thread

def network_connectivity_check(host: str = "8.8.8.8", timeout: int = 5) -> Dict:
    """네트워크 연결성 체크"""
    result = {
        'timestamp': datetime.now().isoformat(),
        'internet': False,
        'local_network': False,
        'dns': False
    }
    
    try:
        # 인터넷 연결 체크 (ping)
        ping_result = subprocess.run(
            ['ping', '-c', '1', '-W', str(timeout), host],
            capture_output=True, text=True
        )
        result['internet'] = ping_result.returncode == 0
        
        # 로컬 네트워크 체크 (게이트웨이)
        try:
            route_result = subprocess.run(
                ['ip', 'route', 'show', 'default'],
                capture_output=True, text=True
            )
            if route_result.returncode == 0 and 'default via' in route_result.stdout:
                gateway = route_result.stdout.split('via')[1].split()[0]
                
                gateway_ping = subprocess.run(
                    ['ping', '-c', '1', '-W', str(timeout), gateway],
                    capture_output=True, text=True
                )
                result['local_network'] = gateway_ping.returncode == 0
        except:
            pass
        
        # DNS 체크
        try:
            dns_result = subprocess.run(
                ['nslookup', 'google.com'],
                capture_output=True, text=True, timeout=timeout
            )
            result['dns'] = dns_result.returncode == 0
        except:
            pass
            
    except Exception as e:
        logger.error(f"네트워크 연결성 체크 실패: {e}")
    
    return result

def cleanup_old_files(directory: str, max_age_days: int = 7, 
                     pattern: str = "*", max_size_mb: int = 1000):
    """오래된 파일 정리"""
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        total_size = 0
        deleted_files = []
        
        # 파일 목록 수집
        files = list(dir_path.glob(pattern))
        files.sort(key=lambda f: f.stat().st_mtime)  # 오래된 순 정렬
        
        for file_path in files:
            try:
                file_stat = file_path.stat()
                file_time = datetime.fromtimestamp(file_stat.st_mtime)
                file_size_mb = file_stat.st_size / (1024 * 1024)
                
                # 오래된 파일이거나 전체 크기가 초과하는 경우 삭제
                if (file_time < cutoff_time or total_size > max_size_mb):
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    logger.debug(f"파일 삭제: {file_path}")
                else:
                    total_size += file_size_mb
                    
            except Exception as e:
                logger.warning(f"파일 처리 실패 {file_path}: {e}")
        
        if deleted_files:
            logger.info(f"오래된 파일 {len(deleted_files)}개 삭제: {directory}")
        
        return deleted_files
        
    except Exception as e:
        logger.error(f"파일 정리 실패: {e}")
        return []

def create_backup(source_files: List[str], backup_dir: str, 
                 max_backups: int = 5) -> str:
    """설정 파일 백업"""
    try:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.tar.gz"
        backup_file = backup_path / backup_name
        
        # tar 아카이브 생성
        import tarfile
        
        with tarfile.open(backup_file, 'w:gz') as tar:
            for source_file in source_files:
                if os.path.exists(source_file):
                    tar.add(source_file, arcname=os.path.basename(source_file))
        
        # 오래된 백업 정리
        backup_files = list(backup_path.glob("backup_*.tar.gz"))
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        for old_backup in backup_files[max_backups:]:
            old_backup.unlink()
            logger.debug(f"오래된 백업 삭제: {old_backup}")
        
        logger.info(f"백업 생성 완료: {backup_file}")
        return str(backup_file)
        
    except Exception as e:
        logger.error(f"백업 생성 실패: {e}")
        return None

def restart_service(service_name: str = "edge") -> bool:
    """서비스 재시작"""
    try:
        service_full_name = f"aquaponics-{service_name}"
        
        # 서비스 중지
        subprocess.run(['sudo', 'systemctl', 'stop', service_full_name], 
                      check=True, timeout=30)
        
        # 잠시 대기
        time.sleep(2)
        
        # 서비스 시작
        subprocess.run(['sudo', 'systemctl', 'start', service_full_name], 
                      check=True, timeout=30)
        
        # 상태 확인
        result = subprocess.run(['sudo', 'systemctl', 'is-active', service_full_name],
                               capture_output=True, text=True)
        
        is_active = result.stdout.strip() == 'active'
        
        if is_active:
            logger.info(f"서비스 재시작 성공: {service_full_name}")
        else:
            logger.error(f"서비스 재시작 실패: {service_full_name}")
        
        return is_active
        
    except subprocess.CalledProcessError as e:
        logger.error(f"서비스 재시작 명령 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"서비스 재시작 오류: {e}")
        return False

def emergency_shutdown(reason: str = "Emergency shutdown requested"):
    """응급 시스템 종료"""
    logger.critical(f"응급 종료: {reason}")
    
    try:
        # 중요한 데이터 백업
        create_backup(['config.yaml', 'logs/*.log'], 'emergency_backup')
        
        # 안전한 GPIO 정리
        try:
            from .gpio_manager import GPIOManager
            gpio = GPIOManager({})
            gpio.cleanup_all()
        except:
            pass
        
        # 프로세스 정리
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
        
    except Exception as e:
        logger.error(f"응급 종료 처리 오류: {e}")
        # 강제 종료
        os._exit(1)

# 편의 함수들
def is_raspberry_pi() -> bool:
    """라즈베리파이 여부 확인"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'Raspberry Pi' in f.read()
    except:
        return False

def get_pi_model() -> Optional[str]:
    """라즈베리파이 모델 조회"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return f.read().strip('\x00')
    except:
        return None

def get_cpu_temperature() -> Optional[float]:
    """CPU 온도 조회"""
    try:
        if is_raspberry_pi():
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                return float(temp_str.replace('temp=', '').replace("'C", ''))
        else:
            # 일반 Linux 시스템
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return int(f.read().strip()) / 1000.0
    except:
        return None
