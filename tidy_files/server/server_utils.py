# server/server_utils.py - 서버 유틸리티 함수들
import os
import sys
import time
import logging
import threading
import subprocess
import json
import sqlite3
import shutil
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def get_server_performance() -> Dict:
    """서버 성능 메트릭 수집"""
    metrics = {
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        import psutil
        
        # CPU 정보
        metrics['cpu'] = {
            'percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count(),
            'freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        metrics['memory'] = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent': memory.percent
        }
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        metrics['disk'] = {
            'total_gb': round(disk.total / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'percent': round((disk.used / disk.total) * 100, 1)
        }
        
        # 네트워크 정보
        net_io = psutil.net_io_counters()
        metrics['network'] = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errin': net_io.errin,
            'errout': net_io.errout
        }
        
        # GPU 정보 (NVIDIA)
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                metrics['gpu'] = {
                    'utilization_percent': int(gpu_data[0]),
                    'memory_used_mb': int(gpu_data[1]),
                    'memory_total_mb': int(gpu_data[2]),
                    'temperature_c': int(gpu_data[3])
                }
        except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
            metrics['gpu'] = None
            
    except ImportError:
        logger.warning("psutil 패키지가 없어 기본 성능 정보만 수집")
        
        # 기본적인 메모리 정보
        try:
            with open('/proc/meminfo', 'r') as f:
                mem_info = {}
                for line in f:
                    key, value = line.split(':')
                    mem_info[key.strip()] = int(value.strip().split()[0])
                
                total_gb = mem_info['MemTotal'] / (1024 * 1024)
                available_gb = mem_info.get('MemAvailable', mem_info['MemFree']) / (1024 * 1024)
                
                metrics['memory'] = {
                    'total_gb': round(total_gb, 2),
                    'available_gb': round(available_gb, 2),
                    'percent': round(((total_gb - available_gb) / total_gb) * 100, 1)
                }
        except Exception as e:
            logger.error(f"메모리 정보 수집 실패: {e}")
    
    return metrics

def optimize_database(db_path: str) -> Dict:
    """데이터베이스 최적화"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'operations': [],
        'size_before_mb': 0,
        'size_after_mb': 0,
        'success': False
    }
    
    try:
        if not os.path.exists(db_path):
            logger.error(f"데이터베이스 파일 없음: {db_path}")
            return results
        
        # 최적화 전 크기
        results['size_before_mb'] = os.path.getsize(db_path) / (1024 * 1024)
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # VACUUM (공간 회수)
            logger.info("데이터베이스 VACUUM 수행 중...")
            cursor.execute("VACUUM")
            results['operations'].append("VACUUM")
            
            # ANALYZE (통계 업데이트)
            logger.info("데이터베이스 ANALYZE 수행 중...")
            cursor.execute("ANALYZE")
            results['operations'].append("ANALYZE")
            
            # 인덱스 재구성
            cursor.execute("REINDEX")
            results['operations'].append("REINDEX")
            
            # WAL 모드 설정 (성능 향상)
            cursor.execute("PRAGMA journal_mode=WAL")
            results['operations'].append("WAL_MODE")
            
            # 동기화 설정 최적화
            cursor.execute("PRAGMA synchronous=NORMAL")
            results['operations'].append("SYNC_NORMAL")
            
            conn.commit()
        
        # 최적화 후 크기
        results['size_after_mb'] = os.path.getsize(db_path) / (1024 * 1024)
        results['space_saved_mb'] = results['size_before_mb'] - results['size_after_mb']
        results['success'] = True
        
        logger.info(f"데이터베이스 최적화 완료: {results['space_saved_mb']:.2f}MB 절약")
        
    except Exception as e:
        logger.error(f"데이터베이스 최적화 실패: {e}")
        results['error'] = str(e)
    
    return results

def backup_database(db_path: str, backup_dir: str = "backup", 
                   max_backups: int = 7) -> Optional[str]:
    """데이터베이스 백업"""
    try:
        if not os.path.exists(db_path):
            logger.error(f"데이터베이스 파일 없음: {db_path}")
            return None
        
        # 백업 디렉토리 생성
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 백업 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = Path(db_path).stem
        backup_filename = f"{db_name}_backup_{timestamp}.db"
        backup_file = backup_path / backup_filename
        
        # 백업 수행 (온라인 백업)
        with sqlite3.connect(db_path) as source:
            with sqlite3.connect(str(backup_file)) as backup:
                source.backup(backup)
        
        logger.info(f"데이터베이스 백업 완료: {backup_file}")
        
        # 오래된 백업 정리
        backup_files = list(backup_path.glob(f"{db_name}_backup_*.db"))
        backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        for old_backup in backup_files[max_backups:]:
            old_backup.unlink()
            logger.debug(f"오래된 백업 삭제: {old_backup}")
        
        return str(backup_file)
        
    except Exception as e:
        logger.error(f"데이터베이스 백업 실패: {e}")
        return None

def export_data_to_csv(db_path: str, export_dir: str = "data/exports") -> Dict[str, str]:
    """데이터를 CSV로 내보내기"""
    export_files = {}
    
    try:
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with sqlite3.connect(db_path) as conn:
            # 주요 테이블들을 CSV로 내보내기
            tables = {
                'events': 'SELECT * FROM events ORDER BY timestamp DESC',
                'detections': '''
                    SELECT d.*, e.timestamp as event_timestamp, e.device_id 
                    FROM detections d 
                    JOIN events e ON d.event_id = e.event_id 
                    ORDER BY e.timestamp DESC
                ''',
                'sensor_readings': 'SELECT * FROM sensor_readings ORDER BY timestamp DESC',
                'daily_summary': 'SELECT * FROM daily_summary ORDER BY date DESC'
            }
            
            for table_name, query in tables.items():
                try:
                    df = pd.read_sql_query(query, conn)
                    
                    if not df.empty:
                        csv_filename = f"{table_name}_{timestamp}.csv"
                        csv_file = export_path / csv_filename
                        
                        df.to_csv(csv_file, index=False, encoding='utf-8')
                        export_files[table_name] = str(csv_file)
                        
                        logger.info(f"테이블 {table_name} 내보내기 완료: {len(df)}행")
                        
                except Exception as e:
                    logger.error(f"테이블 {table_name} 내보내기 실패: {e}")
        
        logger.info(f"데이터 내보내기 완료: {len(export_files)}개 파일")
        
    except Exception as e:
        logger.error(f"데이터 내보내기 실패: {e}")
    
    return export_files

def generate_analytics_report(db_path: str) -> Dict:
    """분석 리포트 생성"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'trends': {},
        'insights': [],
        'recommendations': []
    }
    
    try:
        with sqlite3.connect(db_path) as conn:
            # 기본 통계
            summary_queries = {
                'total_events': 'SELECT COUNT(*) FROM events',
                'total_detections': 'SELECT COUNT(*) FROM detections',
                'total_sensor_readings': 'SELECT COUNT(*) FROM sensor_readings',
                'unique_devices': 'SELECT COUNT(DISTINCT device_id) FROM events',
                'date_range': '''
                    SELECT MIN(timestamp) as start_date, MAX(timestamp) as end_date 
                    FROM events
                '''
            }
            
            for key, query in summary_queries.items():
                try:
                    result = pd.read_sql_query(query, conn)
                    if key == 'date_range':
                        report['summary'][key] = result.iloc[0].to_dict()
                    else:
                        report['summary'][key] = result.iloc[0, 0]
                except Exception as e:
                    logger.warning(f"통계 조회 실패 ({key}): {e}")
            
            # 최근 7일 트렌드
            try:
                trend_query = '''
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as daily_events,
                        COUNT(DISTINCT device_id) as active_devices
                    FROM events 
                    WHERE timestamp >= datetime('now', '-7 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                '''
                
                trend_df = pd.read_sql_query(trend_query, conn)
                report['trends']['daily_activity'] = trend_df.to_dict('records')
                
            except Exception as e:
                logger.warning(f"트렌드 분석 실패: {e}")
            
            # 객체 탐지 통계
            try:
                detection_query = '''
                    SELECT 
                        class_name,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence,
                        AVG(CASE WHEN health_score IS NOT NULL THEN health_score END) as avg_health
                    FROM detections 
                    WHERE event_id IN (
                        SELECT event_id FROM events 
                        WHERE timestamp >= datetime('now', '-7 days')
                    )
                    GROUP BY class_name
                '''
                
                detection_df = pd.read_sql_query(detection_query, conn)
                report['trends']['object_detection'] = detection_df.to_dict('records')
                
            except Exception as e:
                logger.warning(f"객체 탐지 통계 실패: {e}")
            
            # 센서 데이터 분석
            try:
                sensor_query = '''
                    SELECT 
                        sensor_type,
                        COUNT(*) as readings_count,
                        AVG(value) as avg_value,
                        MIN(value) as min_value,
                        MAX(value) as max_value,
                        unit
                    FROM sensor_readings 
                    WHERE timestamp >= datetime('now', '-7 days')
                    GROUP BY sensor_type, unit
                '''
                
                sensor_df = pd.read_sql_query(sensor_query, conn)
                report['trends']['sensor_data'] = sensor_df.to_dict('records')
                
            except Exception as e:
                logger.warning(f"센서 데이터 분석 실패: {e}")
        
        # 인사이트 생성
        report['insights'] = generate_insights(report)
        
        # 권장사항 생성
        report['recommendations'] = generate_recommendations(report)
        
        logger.info("분석 리포트 생성 완료")
        
    except Exception as e:
        logger.error(f"분석 리포트 생성 실패: {e}")
        report['error'] = str(e)
    
    return report

def generate_insights(report: Dict) -> List[str]:
    """데이터 기반 인사이트 생성"""
    insights = []
    
    try:
        summary = report.get('summary', {})
        trends = report.get('trends', {})
        
        # 데이터 볼륨 인사이트
        total_events = summary.get('total_events', 0)
        if total_events > 1000:
            insights.append(f"시스템이 활발히 운영 중입니다 (총 {total_events:,}개 이벤트)")
        elif total_events < 100:
            insights.append("데이터 수집량이 적습니다. 센서 상태를 확인해보세요")
        
        # 객체 탐지 인사이트
        detection_data = trends.get('object_detection', [])
        for detection in detection_data:
            class_name = detection.get('class_name', '')
            avg_health = detection.get('avg_health')
            
            if avg_health is not None:
                if avg_health > 0.8:
                    insights.append(f"{class_name} 개체들의 건강 상태가 양호합니다 (평균 {avg_health:.2f})")
                elif avg_health < 0.6:
                    insights.append(f"{class_name} 개체들의 건강 상태가 우려됩니다 (평균 {avg_health:.2f})")
        
        # 센서 데이터 인사이트
        sensor_data = trends.get('sensor_data', [])
        for sensor in sensor_data:
            sensor_type = sensor.get('sensor_type', '')
            avg_value = sensor.get('avg_value', 0)
            
            if sensor_type == 'temperature' and avg_value > 0:
                if avg_value < 20:
                    insights.append(f"수온이 낮습니다 (평균 {avg_value:.1f}°C)")
                elif avg_value > 28:
                    insights.append(f"수온이 높습니다 (평균 {avg_value:.1f}°C)")
                else:
                    insights.append(f"수온이 적정 범위입니다 (평균 {avg_value:.1f}°C)")
            
            elif sensor_type == 'ph' and avg_value > 0:
                if avg_value < 6.5:
                    insights.append(f"pH가 낮습니다 (평균 {avg_value:.1f})")
                elif avg_value > 7.5:
                    insights.append(f"pH가 높습니다 (평균 {avg_value:.1f})")
        
    except Exception as e:
        logger.error(f"인사이트 생성 오류: {e}")
    
    return insights

def generate_recommendations(report: Dict) -> List[str]:
    """권장사항 생성"""
    recommendations = []
    
    try:
        summary = report.get('summary', {})
        trends = report.get('trends', {})
        
        # 데이터 수집 관련 권장사항
        total_events = summary.get('total_events', 0)
        if total_events < 50:
            recommendations.append("데이터 수집량이 부족합니다. 센서 연결 상태를 확인하세요")
        
        # 활동성 관련 권장사항
        daily_activity = trends.get('daily_activity', [])
        if len(daily_activity) > 0:
            recent_events = daily_activity[-1].get('daily_events', 0)
            if recent_events == 0:
                recommendations.append("최근 이벤트가 없습니다. 시스템 상태를 점검하세요")
        
        # 객체 탐지 관련 권장사항
        detection_data = trends.get('object_detection', [])
        fish_detected = any(d.get('class_name') == 'fish' for d in detection_data)
        plant_detected = any(d.get('class_name') == 'plant' for d in detection_data)
        
        if not fish_detected:
            recommendations.append("물고기가 감지되지 않습니다. 카메라 각도를 조정하세요")
        
        if not plant_detected:
            recommendations.append("식물이 감지되지 않습니다. 식물 베드를 확인하세요")
        
        # 센서 데이터 기반 권장사항
        sensor_data = trends.get('sensor_data', [])
        for sensor in sensor_data:
            sensor_type = sensor.get('sensor_type', '')
            avg_value = sensor.get('avg_value', 0)
            readings_count = sensor.get('readings_count', 0)
            
            if readings_count < 10:
                recommendations.append(f"{sensor_type} 센서 데이터가 부족합니다")
            
            if sensor_type == 'dissolved_oxygen' and avg_value < 6.0:
                recommendations.append("용존산소 농도가 낮습니다. 에어펌프를 점검하세요")
        
        if not recommendations:
            recommendations.append("시스템이 정상적으로 운영되고 있습니다")
        
    except Exception as e:
        logger.error(f"권장사항 생성 오류: {e}")
    
    return recommendations

def cleanup_old_logs(log_dir: str = "logs", max_age_days: int = 30) -> int:
    """오래된 로그 파일 정리"""
    deleted_count = 0
    
    try:
        log_path = Path(log_dir)
        if not log_path.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        for log_file in log_path.glob("*.log"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if file_time < cutoff_time:
                    log_file.unlink()
                    deleted_count += 1
                    logger.debug(f"오래된 로그 삭제: {log_file}")
                    
            except Exception as e:
                logger.warning(f"로그 파일 처리 실패 {log_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"오래된 로그 파일 {deleted_count}개 삭제")
        
    except Exception as e:
        logger.error(f"로그 정리 실패: {e}")
    
    return deleted_count

def create_system_health_check() -> Dict:
    """시스템 건강 상태 체크"""
    health = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'checks': {}
    }
    
    try:
        # 디스크 공간 체크
        disk_usage = shutil.disk_usage('/')
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        
        health['checks']['disk_space'] = {
            'status': 'warning' if disk_percent > 85 else 'healthy',
            'usage_percent': round(disk_percent, 1),
            'free_gb': round(disk_usage.free / (1024**3), 1)
        }
        
        # 메모리 사용률 체크
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            health['checks']['memory'] = {
                'status': 'warning' if memory.percent > 85 else 'healthy',
                'usage_percent': memory.percent,
                'available_gb': round(memory.available / (1024**3), 1)
            }
        except ImportError:
            health['checks']['memory'] = {'status': 'unknown', 'reason': 'psutil not available'}
        
        # 데이터베이스 연결 체크
        try:
            db_path = 'data/aquaponics.db'
            if os.path.exists(db_path):
                with sqlite3.connect(db_path, timeout=5) as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT 1')
                    
                health['checks']['database'] = {
                    'status': 'healthy',
                    'size_mb': round(os.path.getsize(db_path) / (1024*1024), 1)
                }
            else:
                health['checks']['database'] = {
                    'status': 'error',
                    'reason': 'Database file not found'
                }
        except Exception as e:
            health['checks']['database'] = {
                'status': 'error',
                'reason': str(e)
            }
        
        # 로그 파일 상태 체크
        log_dir = Path('logs')
        if log_dir.exists():
            log_files = list(log_dir.glob('*.log'))
            total_log_size = sum(f.stat().st_size for f in log_files) / (1024*1024)
            
            health['checks']['logs'] = {
                'status': 'warning' if total_log_size > 100 else 'healthy',
                'file_count': len(log_files),
                'total_size_mb': round(total_log_size, 1)
            }
        else:
            health['checks']['logs'] = {
                'status': 'warning',
                'reason': 'Log directory not found'
            }
        
        # 전체 상태 결정
        check_statuses = [check.get('status', 'unknown') for check in health['checks'].values()]
        
        if 'error' in check_statuses:
            health['overall_status'] = 'error'
        elif 'warning' in check_statuses:
            health['overall_status'] = 'warning'
        else:
            health['overall_status']
