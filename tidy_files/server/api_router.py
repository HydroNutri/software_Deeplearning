# server/api_router.py - API 라우터 모듈
from flask import Blueprint, request, jsonify, send_file
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import io
import csv

logger = logging.getLogger(__name__)

# API Blueprint 생성
api_bp = Blueprint('api', __name__, url_prefix='/api')

# 전역 서버 인스턴스 (lightweight_server.py에서 설정)
server_instance = None

def set_server_instance(server):
    """서버 인스턴스 설정"""
    global server_instance
    server_instance = server

def validate_request_data(required_fields: List[str], data: Dict) -> Optional[str]:
    """요청 데이터 유효성 검사"""
    if not data:
        return "요청 데이터가 없습니다"
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return f"필수 필드 누락: {', '.join(missing_fields)}"
    
    return None

# ============== 시스템 관련 API ==============

@api_bp.route('/system/status', methods=['GET'])
def get_system_status():
    """시스템 상태 조회"""
    try:
        if not server_instance:
            return jsonify({'error': '서버 인스턴스가 초기화되지 않음'}), 500
        
        status = server_instance.get_system_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"시스템 상태 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/system/health', methods=['GET'])
def get_system_health():
    """시스템 건강 상태 조회"""
    try:
        from .server_utils import create_system_health_check
        
        health = create_system_health_check()
        return jsonify(health)
        
    except Exception as e:
        logger.error(f"시스템 건강 상태 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/system/performance', methods=['GET'])
def get_system_performance():
    """시스템 성능 메트릭"""
    try:
        from .server_utils import get_server_performance
        
        performance = get_server_performance()
        return jsonify(performance)
        
    except Exception as e:
        logger.error(f"성능 메트릭 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/system/logs', methods=['GET'])
def get_system_logs():
    """시스템 로그 조회"""
    try:
        lines = request.args.get('lines', 100, type=int)
        level = request.args.get('level', 'all')  # all, error, warning, info
        
        from pathlib import Path
        
        log_files = list(Path('logs').glob('*.log'))
        if not log_files:
            return jsonify({'logs': [], 'message': '로그 파일이 없습니다'})
        
        # 가장 최근 로그 파일 사용
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_log, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # 레벨 필터링
        if level != 'all':
            filtered_lines = []
            for line in all_lines:
                if level.upper() in line:
                    filtered_lines.append(line)
            all_lines = filtered_lines
        
        # 최근 N줄만 반환
        recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return jsonify({
            'logs': [line.strip() for line in recent_lines],
            'total_lines': len(all_lines),
            'file': str(latest_log),
            'level_filter': level
        })
        
    except Exception as e:
        logger.error(f"로그 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============== Edge 이벤트 관련 API ==============

@api_bp.route('/edge/events', methods=['POST'])
def receive_edge_events():
    """Edge 이벤트 수신"""
    try:
        data = request.get_json()
        
        validation_error = validate_request_data(['device_id', 'events'], data)
        if validation_error:
            return jsonify({'error': validation_error}), 400
        
        if not server_instance:
            return jsonify({'error': '서버 인스턴스가 초기화되지 않음'}), 500
        
        device_id = data['device_id']
        events_data = data['events']
        
        # 이벤트 처리 통계
        received_events = []
        failed_events = []
        
        for event_data in events_data:
            try:
                from common.data_types import SystemEvent
                event = SystemEvent.from_dict(event_data)
                
                if not server_instance.event_queue.full():
                    server_instance.event_queue.put(event)
                    received_events.append(event.event_id)
                else:
                    failed_events.append({
                        'event_id': event.event_id,
                        'reason': 'queue_full'
                    })
                    
            except Exception as e:
                failed_events.append({
                    'event_id': event_data.get('event_id', 'unknown'),
                    'reason': str(e)
                })
        
        # 제어 명령 생성
        control_commands = []
        if server_instance.analysis_engine:
            for event_data in events_data:
                if 'sensors' in event_data:
                    commands = server_instance.analysis_engine.generate_control_commands(
                        event_data['sensors'], device_id
                    )
                    control_commands.extend([cmd.to_dict() for cmd in commands])
        
        response = {
            'status': 'processed',
            'received_count': len(received_events),
            'failed_count': len(failed_events),
            'control_commands': control_commands,
            'timestamp': datetime.now().isoformat()
        }
        
        if failed_events:
            response['failed_events'] = failed_events
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Edge 이벤트 수신 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/edge/devices', methods=['GET'])
def get_edge_devices():
    """연결된 Edge 디바이스 목록"""
    try:
        if not server_instance or not server_instance.database_manager:
            return jsonify({'error': '데이터베이스 연결 없음'}), 500
        
        # 최근 1시간 내 활동한 디바이스 조회
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        with server_instance.database_manager.conn_lock:
            conn = server_instance.database_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    device_id,
                    COUNT(*) as event_count,
                    MAX(timestamp) as last_activity,
                    MIN(timestamp) as first_activity
                FROM events 
                WHERE timestamp > ?
                GROUP BY device_id
                ORDER BY last_activity DESC
            ''', (cutoff_time.isoformat(),))
            
            devices = []
            for row in cursor.fetchall():
                devices.append({
                    'device_id': row[0],
                    'event_count': row[1],
                    'last_activity': row[2],
                    'first_activity': row[3],
                    'status': 'active'
                })
        
        return jsonify({
            'devices': devices,
            'total_count': len(devices),
            'check_period_hours': 1
        })
        
    except Exception as e:
        logger.error(f"Edge 디바이스 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============== 분석 관련 API ==============

@api_bp.route('/analysis/summary', methods=['GET'])
def get_analysis_summary():
    """분석 요약 조회"""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        if not server_instance:
            return jsonify({'error': '서버 인스턴스가 초기화되지 않음'}), 500
        
        # 기본 요약 정보
        summary = {
            'period_hours': hours,
            'total_events': server_instance.processed_events,
            'object_detections': {'fish': 0, 'plant': 0, 'food': 0},
            'sensor_readings': {},
            'alerts_generated': server_instance.performance_stats.get('alerts_sent', 0),
            'system_performance': server_instance.performance_stats.copy()
        }
        
        # 데이터베이스에서 추가 정보 조회
        if server_instance.database_manager:
            try:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                # 객체 탐지 통계
                detections = server_instance.database_manager.get_recent_detections(hours)
                for detection in detections:
                    class_name = detection.get('class_name', 'unknown')
                    if class_name in summary['object_detections']:
                        summary['object_detections'][class_name] += 1
                
                # 센서 데이터 통계
                sensor_trends = server_instance.database_manager.get_sensor_trends(hours)
                sensor_counts = {}
                for trend in sensor_trends:
                    for key, value in trend.items():
                        if key != 'timestamp' and value is not None:
                            sensor_counts[key] = sensor_counts.get(key, 0) + 1
                
                summary['sensor_readings'] = sensor_counts
                
            except Exception as e:
                logger.warning(f"데이터베이스 조회 중 오류: {e}")
        
        # 분석 엔진에서 추가 정보
        if server_instance.analysis_engine:
            try:
                growth_summary = server_instance.analysis_engine.get_growth_summary(
                    datetime.now() - timedelta(hours=hours)
                )
                health_summary = server_instance.analysis_engine.get_health_summary(
                    datetime.now() - timedelta(hours=hours)
                )
                
                summary['growth_analysis'] = growth_summary
                summary['health_metrics'] = health_summary
                
            except Exception as e:
                logger.warning(f"분석 엔진 조회 중 오류: {e}")
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"분석 요약 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/analysis/report', methods=['GET'])
def generate_analysis_report():
    """분석 리포트 생성"""
    try:
        format_type = request.args.get('format', 'json')  # json, csv, excel
        
        if not server_instance or not server_instance.database_manager:
            return jsonify({'error': '데이터베이스 연결 없음'}), 500
        
        from .server_utils import generate_analytics_report
        
        report = generate_analytics_report(server_instance.database_manager.db_path)
        
        if format_type == 'json':
            return jsonify(report)
        
        elif format_type == 'csv':
            # CSV 형태로 변환
            output = io.StringIO()
            writer = csv.writer(output)
            
            # 헤더
            writer.writerow(['섹션', '항목', '값'])
            
            # 요약 정보
            for key, value in report.get('summary', {}).items():
                writer.writerow(['요약', key, str(value)])
            
            # 인사이트
            for i, insight in enumerate(report.get('insights', [])):
                writer.writerow(['인사이트', f'insight_{i+1}', insight])
            
            # 권장사항
            for i, rec in enumerate(report.get('recommendations', [])):
                writer.writerow(['권장사항', f'recommendation_{i+1}', rec])
            
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        
        else:
            return jsonify({'error': f'지원하지 않는 형식: {format_type}'}), 400
            
    except Exception as e:
        logger.error(f"분석 리포트 생성 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============== 객체 추적 관련 API ==============

@api_bp.route('/objects/tracking', methods=['GET'])
def get_object_tracking():
    """객체 추적 정보 조회"""
    try:
        object_type = request.args.get('type', 'all')
        hours = request.args.get('hours', 24, type=int)
        
        if not server_instance or not server_instance.analysis_engine:
            return jsonify({'error': '분석 엔진이 초기화되지 않음'}), 500
        
        tracking_data = server_instance.analysis_engine.get_object_tracking(object_type, hours)
        
        return jsonify({
            'object_type': object_type,
            'period_hours': hours,
            'tracking_data': tracking_data,
            'total_objects': len(tracking_data)
        })
        
    except Exception as e:
        logger.error(f"객체 추적 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/objects/statistics', methods=['GET'])
def get_object_statistics():
    """객체 통계 정보"""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        if not server_instance or not server_instance.database_manager:
            return jsonify({'error': '데이터베이스 연결 없음'}), 500
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        detections = server_instance.database_manager.get_recent_detections(hours)
        
        # 통계 계산
        stats = {
            'total_detections': len(detections),
            'by_class': {},
            'confidence_stats': {},
            'health_stats': {}
        }
        
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            confidence = detection.get('confidence', 0)
            health_score = detection.get('health_score')
            
            # 클래스별 통계
            if class_name not in stats['by_class']:
                stats['by_class'][class_name] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'confidences': []
                }
            
            stats['by_class'][class_name]['count'] += 1
            stats['by_class'][class_name]['confidences'].append(confidence)
            
            # 건강도 통계
            if health_score is not None:
                if class_name not in stats['health_stats']:
                    stats['health_stats'][class_name] = {
                        'count': 0,
                        'avg_health': 0,
                        'health_scores': []
                    }
                
                stats['health_stats'][class_name]['count'] += 1
                stats['health_stats'][class_name]['health_scores'].append(health_score)
        
        # 평균값 계산
        for class_name, class_stats in stats['by_class'].items():
            confidences = class_stats['confidences']
            if confidences:
                class_stats['avg_confidence'] = sum(confidences) / len(confidences)
                class_stats['min_confidence'] = min(confidences)
                class_stats['max_confidence'] = max(confidences)
            del class_stats['confidences']  # 원본 데이터 제거
        
        for class_name, health_stats in stats['health_stats'].items():
            health_scores = health_stats['health_scores']
            if health_scores:
                health_stats['avg_health'] = sum(health_scores) / len(health_scores)
                health_stats['min_health'] = min(health_scores)
                health_stats['max_health'] = max(health_scores)
            del health_stats['health_scores']  # 원본 데이터 제거
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"객체 통계 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============== 환경 관련 API ==============

@api_bp.route('/environment/trends', methods=['GET'])
def get_environment_trends():
    """환경 트렌드 조회"""
    try:
        hours = request.args.get('hours', 24, type=int)
        
        if not server_instance or not server_instance.analysis_engine:
            return jsonify({'error': '분석 엔진이 초기화되지 않음'}), 500
        
        trends = server_instance.analysis_engine.get_environment_trends(hours)
        
        return jsonify({
            'period_hours': hours,
            'trends': trends,
            'data_points': len(trends)
        })
        
    except Exception as e:
        logger.error(f"환경 트렌드 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/environment/current', methods=['GET'])
def get_current_environment():
    """현재 환경 상태"""
    try:
        if not server_instance or not server_instance.database_manager:
            return jsonify({'error': '데이터베이스 연결 없음'}), 500
        
        # 최근 10분 내 센서 데이터 조회
        cutoff_time = datetime.now() - timedelta(minutes=10)
        
        with server_instance.database_manager.conn_lock:
            conn = server_instance.database_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT sensor_type, value, unit, timestamp, status
                FROM sensor_readings 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time.isoformat(),))
            
            current_data = {}
            for row in cursor.fetchall():
                sensor_type, value, unit, timestamp, status = row
                
                # 각 센서 타입별로 가장 최근 값만 유지
                if sensor_type not in current_data:
                    current_data[sensor_type] = {
                        'value': value,
                        'unit': unit,
                        'timestamp': timestamp,
                        'status': status
                    }
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'current_readings': current_data,
            'data_age_minutes': 10
        })
        
    except Exception as e:
        logger.error(f"현재 환경 상태 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============== 알림 관련 API ==============

@api_bp.route('/alerts/recent', methods=['GET'])
def get_recent_alerts():
    """최근 알림 조회"""
    try:
        hours = request.args.get('hours', 24, type=int)
        severity = request.args.get('severity', 'all')
        
        if not server_instance or not server_instance.alert_manager:
            return jsonify({'error': '알림 관리자가 초기화되지 않음'}), 500
        
        alerts = server_instance.alert_manager.get_recent_alerts(hours, severity)
        
        return jsonify({
            'period_hours': hours,
            'severity_filter': severity,
            'alerts': alerts,
            'total_count': len(alerts)
        })
        
    except Exception as e:
        logger.error(f"최근 알림 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/alerts/statistics', methods=['GET'])
def get_alert_statistics():
    """알림 통계"""
    try:
        if not server_instance or not server_instance.alert_manager:
            return jsonify({'error': '알림 관리자가 초기화되지 않음'}), 500
        
        stats = server_instance.alert_manager.get_alert_statistics()
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"알림 통계 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/alerts/test', methods=['POST'])
def send_test_alert():
    """테스트 알림 전송"""
    try:
        data = request.get_json()
        
        if not server_instance or not server_instance.alert_manager:
            return jsonify({'error': '알림 관리자가 초기화되지 않음'}), 500
        
        # 테스트 알림 생성
        test_alert = {
            'type': 'test_alert',
            'severity': data.get('severity', 'low'),
            'title': '테스트 알림',
            'message': data.get('message', '시스템 테스트용 알림입니다'),
            'timestamp': datetime.now().isoformat()
        }
        
        success = server_instance.alert_manager.send_alert(test_alert)
        
        return jsonify({
            'success': success,
            'alert': test_alert,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"테스트 알림 전송 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============== 제어 관련 API ==============

@api_bp.route('/control/command', methods=['POST'])
def send_control_command():
    """제어 명령 전송"""
    try:
        data = request.get_json()
        
        validation_error = validate_request_data(
            ['device_id', 'command_type', 'target_value'], data
        )
        if validation_error:
            return jsonify({'error': validation_error}), 400
        
        if not server_instance:
            return jsonify({'error': '서버 인스턴스가 초기화되지 않음'}), 500
        
        from common.data_types import ControlCommand, generate_event_id
        
        command = ControlCommand(
            command_id=generate_event_id(),
            device_id=data['device_id'],
            command_type=data['command_type'],
            target_value=data['target_value'],
            duration=data.get('duration'),
            priority=data.get('priority', 1)
        )
        
        # MQTT로 제어 명령 전송
        success = server_instance.mqtt_client.publish('control', {
            'target_device': command.device_id,
            'command': command.to_dict()
        })
        
        if success:
            # 데이터베이스에 명령 기록
            if server_instance.database_manager:
                server_instance.database_manager.save_control_command(command)
            
            return jsonify({
                'status': 'sent',
                'command_id': command.command_id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'MQTT 전송 실패'}), 500
            
    except Exception as e:
        logger.error(f"제어 명령 전송 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/control/history', methods=['GET'])
def get_control_history():
    """제어 명령 이력 조회"""
    try:
        hours = request.args.get('hours', 24, type=int)
        device_id = request.args.get('device_id')
        
        if not server_instance or not server_instance.database_manager:
            return jsonify({'error': '데이터베이스 연결 없음'}), 500
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with server_instance.database_manager.conn_lock:
            conn = server_instance.database_manager.get_connection()
            cursor = conn.cursor()
            
            query = '''
                SELECT command_id, timestamp, device_id, command_type, 
                       target_value, duration, status, result
                FROM control_history 
                WHERE timestamp > ?
            '''
            params = [cutoff_time.isoformat()]
            
            if device_id:
                query += ' AND device_id = ?'
                params.append(device_id)
            
            query += ' ORDER BY timestamp DESC LIMIT 100'
            
            cursor.execute(query, params)
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'command_id': row[0],
                    'timestamp': row[1],
                    'device_id': row[2],
                    'command_type': row[3],
                    'target_value': row[4],
                    'duration': row[5],
                    'status': row[6],
                    'result': row[7]
                })
        
        return jsonify({
            'period_hours': hours,
            'device_filter': device_id,
            'history': history,
            'total_count': len(history)
        })
        
    except Exception as e:
        logger.error(f"제어 이력 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============== 데이터베이스 관리 API ==============

@api_bp.route('/database/export', methods=['GET'])
def export_database():
    """데이터베이스 내보내기"""
    try:
        format_type = request.args.get('format', 'json')
        hours = request.args.get('hours', 24, type=int)
        
        if not server_instance or not server_instance.database_manager:
            return jsonify({'error': '데이터베이스 연결 없음'}), 500
        
        if format_type == 'csv':
            from .server_utils import export_data_to_csv
            
            export_files = export_data_to_csv(server_instance.database_manager.db_path)
            
            return jsonify({
                'status': 'exported',
                'format': 'csv',
                'files': export_files,
                'timestamp': datetime.now().isoformat()
            })
        
        elif format_type == 'json':
            # JSON 형태로 데이터 내보내기
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            export_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'period_hours': hours,
                    'format': 'json'
                },
                'events': [],
                'detections': [],
                'sensor_readings': []
            }
            
            # 이벤트 데이터
            events_count = server_instance.database_manager.count_events_since(cutoff_time)
            export_data['events'] = {'count': events_count, 'note': 'Use specific endpoints for detailed data'}
            
            return jsonify(export_data)
        
        else:
            return jsonify({'error': f'지원하지 않는 형식: {format_type}'}), 400
            
    except Exception as e:
        logger.error(f"데이터베이스 내보내기 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/database/backup', methods=['POST'])
def create_database_backup():
    """데이터베이스 백업 생성"""
    try:
        if not server_instance or not server_instance.database_manager:
            return jsonify({'error': '데이터베이스 연결 없음'}), 500
        
        from .server_utils import backup_database
        
        backup_file = backup_database(server_instance.database_manager.db_path)
        
        if backup_file:
            return jsonify({
                'status': 'success',
                'backup_file': backup_file,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': '백업 생성 실패'}), 500
            
    except Exception as e:
        logger.error(f"데이터베이스 백업 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/database/optimize', methods=['POST'])
def optimize_database():
    """데이터베이스 최적화"""
    try:
        if not server_instance or not server_instance.database_manager:
            return jsonify({'error': '데이터베이스 연결 없음'}), 500
        
        from .server_utils import optimize_database
        
        result = optimize_database(server_instance.database_manager.db_path)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"데이터베이스 최적화 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============== 유틸리티 API ==============

@api_bp.route('/utils/ping', methods=['GET'])
def ping():
    """서버 연결 테스트"""
    return jsonify({
        'status': 'pong',
        'timestamp': datetime.now().isoformat(),
        'server_version': '2.0.0'
    })

@api_bp.route('/utils/config', methods=['GET'])
def get_config_info():
    """설정 정보 조회 (민감 정보 제외)"""
    try:
        if not server_instance:
            return jsonify({'error': '서버 인스턴스가 초기화되지 않음'}), 500
        
        # 민감하지 않은 설정 정보만 반환
        safe_config = {
            'system': {
                'name': server_instance.config.get('system', {}).get('name', 'aquaponics_v2'),
                'version': server_instance.config.get('system', {}).get('version', '2.0.0'),
                'mode': server_instance.config.get('system', {}).get('mode', 'production')
            },
            'server': {
                'device_id': server_instance.config.get('server', {}).get('device_id', 'server_001'),
                'hardware': server_instance.config.get('server', {}).get('hardware', {}),
                'processing': server_instance.config.get('server', {}).get('processing', {})
            }
        }
        
        return jsonify(safe_config)
        
    except Exception as e:
        logger.error(f"설정 정보 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/utils/docs', methods=['GET'])
def get_api_docs():
    """API 문서 조회"""
    try:
        from .server_utils import create_api_documentation
        
        docs = create_api_documentation()
        return jsonify(docs)
        
    except Exception as e:
        logger.error(f"API 문서 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

# ============== 에러 핸들러 ==============

@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad Request', 'message': str(error)}), 400

@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not Found', 'message': 'API endpoint not found'}), 404

@api_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal Server Error', 'message': 'An unexpected error occurred'}), 500

# ============== API 정보 ==============

@api_bp.route('/', methods=['GET'])
def api_info():
    """API 정보 페이지"""
    return jsonify({
        'name': 'Aquaponics Server API',
        'version': '2.0.0',
        'description': 'Edge-Server 분리 아키텍처 기반 아쿠아포닉스 모니터링 시스템',
        'endpoints': {
            'system': '/api/system/*',
            'edge': '/api/edge/*',
            'analysis': '/api/analysis/*',
            'objects': '/api/objects/*',
            'environment': '/api/environment/*',
            'alerts': '/api/alerts/*',
            'control': '/api/control/*',
            'database': '/api/database/*',
            'utils': '/api/utils/*'
        },
        'docs': '/api/utils/docs',
        'timestamp': datetime.now().isoformat()
    })
