# common/data_types.py - 공통 데이터 구조 및 타입 정의
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import uuid
import json
import numpy as np

@dataclass
class BoundingBox:
    """바운딩 박스 데이터"""
    x1: int
    y1: int  
    x2: int
    y2: int
    confidence: float
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
        
    @property
    def height(self) -> int:
        return self.y2 - self.y1
        
    @property
    def area(self) -> int:
        return self.width * self.height
        
    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

@dataclass
class DetectedObject:
    """탐지된 객체 정보"""
    object_id: str
    class_name: str
    class_id: int
    bbox: BoundingBox
    confidence: float
    tracking_id: Optional[int] = None
    
    # 물리적 측정값
    length_cm: Optional[float] = None
    width_cm: Optional[float] = None
    height_cm: Optional[float] = None
    weight_g: Optional[float] = None
    
    # 생물학적 특성
    health_score: Optional[float] = None
    activity_level: Optional[float] = None
    color_analysis: Optional[Dict] = None
    
    # 성장 분석
    growth_rate: Optional[float] = None
    age_days: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        result = {
            'object_id': self.object_id,
            'class_name': self.class_name,
            'class_id': self.class_id,
            'bbox': {
                'x1': self.bbox.x1,
                'y1': self.bbox.y1, 
                'x2': self.bbox.x2,
                'y2': self.bbox.y2,
                'confidence': self.bbox.confidence
            },
            'confidence': self.confidence,
            'tracking_id': self.tracking_id
        }
        
        # 선택적 필드들
        for field_name in ['length_cm', 'width_cm', 'height_cm', 'weight_g', 
                          'health_score', 'activity_level', 'growth_rate', 'age_days']:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
                
        if self.color_analysis:
            result['color_analysis'] = self.color_analysis
            
        return result

@dataclass
class SensorReading:
    """센서 측정값"""
    sensor_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime
    status: str = "normal"  # normal, warning, error
    
    def to_dict(self) -> Dict:
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status
        }

@dataclass
class EnvironmentData:
    """환경 데이터 통합"""
    timestamp: datetime
    water_temp: Optional[float] = None
    air_temp: Optional[float] = None
    ph: Optional[float] = None
    ec: Optional[float] = None
    dissolved_oxygen: Optional[float] = None
    humidity: Optional[float] = None
    light_intensity: Optional[float] = None
    
    def to_dict(self) -> Dict:
        result = {'timestamp': self.timestamp.isoformat()}
        for field in ['water_temp', 'air_temp', 'ph', 'ec', 'dissolved_oxygen', 
                     'humidity', 'light_intensity']:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
        return result
    
    def get_sensor_readings(self) -> List[SensorReading]:
        """SensorReading 리스트로 변환"""
        readings = []
        sensor_mapping = {
            'water_temp': ('temperature', '°C'),
            'air_temp': ('air_temperature', '°C'), 
            'ph': ('ph', 'pH'),
            'ec': ('conductivity', 'mS/cm'),
            'dissolved_oxygen': ('dissolved_oxygen', 'mg/L'),
            'humidity': ('humidity', '%'),
            'light_intensity': ('light', 'lux')
        }
        
        for field, (sensor_type, unit) in sensor_mapping.items():
            value = getattr(self, field)
            if value is not None:
                readings.append(SensorReading(
                    sensor_id=f"{sensor_type}_01",
                    sensor_type=sensor_type,
                    value=value,
                    unit=unit,
                    timestamp=self.timestamp
                ))
        return readings

@dataclass
class YOLOResult:
    """YOLO 탐지 결과"""
    model_name: str
    model_version: str
    processing_time: float
    frame_id: str
    timestamp: datetime
    objects: List[DetectedObject]
    frame_shape: tuple  # (height, width, channels)
    device_info: str
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'processing_time': self.processing_time,
            'frame_id': self.frame_id,
            'timestamp': self.timestamp.isoformat(),
            'objects': [obj.to_dict() for obj in self.objects],
            'frame_shape': self.frame_shape,
            'device_info': self.device_info,
            'object_count': len(self.objects)
        }

@dataclass
class ControlCommand:
    """제어 명령"""
    command_id: str
    device_id: str
    command_type: str  # relay, pwm, servo
    target_value: float
    duration: Optional[int] = None  # seconds
    priority: int = 1  # 1=normal, 2=high, 3=emergency
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'command_id': self.command_id,
            'device_id': self.device_id,
            'command_type': self.command_type,
            'target_value': self.target_value,
            'duration': self.duration,
            'priority': self.priority,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class SystemEvent:
    """시스템 이벤트 (Edge/Server 공통)"""
    event_id: str
    timestamp: datetime
    device_id: str
    event_type: str  # detection, sensor, control, alert, system
    
    # 센서 데이터
    sensors: Optional[EnvironmentData] = None
    
    # YOLO 결과 
    yolo_result: Optional[YOLOResult] = None
    
    # 제어 명령
    control_commands: List[ControlCommand] = field(default_factory=list)
    
    # 메타데이터
    frame_path: Optional[str] = None
    edge_latency: Optional[float] = None
    server_latency: Optional[float] = None
    analysis_type: str = "edge"  # edge, server, hybrid
    
    # 추가 데이터
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create_edge_event(cls, device_id: str, sensors: EnvironmentData = None, 
                         yolo_result: YOLOResult = None) -> 'SystemEvent':
        """Edge 이벤트 생성"""
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            device_id=device_id,
            event_type="detection" if yolo_result else "sensor",
            sensors=sensors,
            yolo_result=yolo_result,
            analysis_type="edge"
        )
    
    @classmethod  
    def create_server_event(cls, edge_event: 'SystemEvent', 
                           server_yolo_result: YOLOResult = None) -> 'SystemEvent':
        """Server 재분석 이벤트 생성"""
        return cls(
            event_id=edge_event.event_id,  # 동일한 event_id 유지
            timestamp=edge_event.timestamp,
            device_id=edge_event.device_id,
            event_type="reanalysis",
            sensors=edge_event.sensors,
            yolo_result=server_yolo_result,
            frame_path=edge_event.frame_path,
            edge_latency=edge_event.edge_latency,
            analysis_type="server",
            metadata={'original_edge_result': edge_event.yolo_result.to_dict() if edge_event.yolo_result else None}
        )
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (MQTT/HTTP 전송용)"""
        result = {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'device_id': self.device_id,
            'event_type': self.event_type,
            'analysis_type': self.analysis_type
        }
        
        if self.sensors:
            result['sensors'] = self.sensors.to_dict()
            
        if self.yolo_result:
            result['yolo_result'] = self.yolo_result.to_dict()
            
        if self.control_commands:
            result['control_commands'] = [cmd.to_dict() for cmd in self.control_commands]
            
        if self.frame_path:
            result['frame_path'] = self.frame_path
            
        if self.edge_latency:
            result['edge_latency'] = self.edge_latency
            
        if self.server_latency:
            result['server_latency'] = self.server_latency
            
        if self.metadata:
            result['metadata'] = self.metadata
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemEvent':
        """딕셔너리에서 복원"""
        # 기본 필드들
        event = cls(
            event_id=data['event_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            device_id=data['device_id'],
            event_type=data['event_type'],
            analysis_type=data.get('analysis_type', 'edge')
        )
        
        # 센서 데이터 복원
        if 'sensors' in data:
            sensor_data = data['sensors']
            event.sensors = EnvironmentData(
                timestamp=datetime.fromisoformat(sensor_data['timestamp']),
                **{k: v for k, v in sensor_data.items() if k != 'timestamp'}
            )
        
        # YOLO 결과 복원  
        if 'yolo_result' in data:
            yolo_data = data['yolo_result']
            objects = []
            for obj_data in yolo_data.get('objects', []):
                bbox = BoundingBox(
                    x1=obj_data['bbox']['x1'],
                    y1=obj_data['bbox']['y1'],
                    x2=obj_data['bbox']['x2'], 
                    y2=obj_data['bbox']['y2'],
                    confidence=obj_data['bbox']['confidence']
                )
                
                obj = DetectedObject(
                    object_id=obj_data['object_id'],
                    class_name=obj_data['class_name'],
                    class_id=obj_data['class_id'],
                    bbox=bbox,
                    confidence=obj_data['confidence'],
                    tracking_id=obj_data.get('tracking_id')
                )
                
                # 선택적 필드들 복원
                for field in ['length_cm', 'width_cm', 'height_cm', 'weight_g',
                             'health_score', 'activity_level', 'growth_rate', 'age_days']:
                    if field in obj_data:
                        setattr(obj, field, obj_data[field])
                        
                if 'color_analysis' in obj_data:
                    obj.color_analysis = obj_data['color_analysis']
                    
                objects.append(obj)
            
            event.yolo_result = YOLOResult(
                model_name=yolo_data['model_name'],
                model_version=yolo_data['model_version'],
                processing_time=yolo_data['processing_time'],
                frame_id=yolo_data['frame_id'],
                timestamp=datetime.fromisoformat(yolo_data['timestamp']),
                objects=objects,
                frame_shape=tuple(yolo_data['frame_shape']),
                device_info=yolo_data['device_info']
            )
        
        # 제어 명령 복원
        if 'control_commands' in data:
            commands = []
            for cmd_data in data['control_commands']:
                cmd = ControlCommand(
                    command_id=cmd_data['command_id'],
                    device_id=cmd_data['device_id'],
                    command_type=cmd_data['command_type'],
                    target_value=cmd_data['target_value'],
                    duration=cmd_data.get('duration'),
                    priority=cmd_data.get('priority', 1),
                    timestamp=datetime.fromisoformat(cmd_data['timestamp'])
                )
                commands.append(cmd)
            event.control_commands = commands
        
        # 메타데이터 복원
        event.frame_path = data.get('frame_path')
        event.edge_latency = data.get('edge_latency')  
        event.server_latency = data.get('server_latency')
        event.metadata = data.get('metadata', {})
        
        return event

@dataclass
class HealthMetrics:
    """건강 상태 메트릭"""
    timestamp: datetime
    object_id: str
    object_type: str
    
    # 기본 지표
    health_score: float  # 0-1
    activity_level: float  # 0-1
    growth_rate: float  # cm/day or g/day
    
    # 환경 상관관계
    temperature_correlation: float
    ph_correlation: float
    light_correlation: float
    
    # 이상 탐지
    anomaly_score: float  # 0-1
    anomaly_type: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'object_id': self.object_id,
            'object_type': self.object_type,
            'health_score': self.health_score,
            'activity_level': self.activity_level,
            'growth_rate': self.growth_rate,
            'temperature_correlation': self.temperature_correlation,
            'ph_correlation': self.ph_correlation,
            'light_correlation': self.light_correlation,
            'anomaly_score': self.anomaly_score,
            'anomaly_type': self.anomaly_type
        }

# 유틸리티 함수들
def generate_event_id() -> str:
    """고유 이벤트 ID 생성"""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

def generate_frame_id() -> str:
    """프레임 ID 생성"""
    return f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

def validate_bbox(bbox: BoundingBox, frame_shape: tuple) -> bool:
    """바운딩 박스 유효성 검사"""
    h, w = frame_shape[:2]
    return (0 <= bbox.x1 < bbox.x2 <= w and 
            0 <= bbox.y1 < bbox.y2 <= h and
            0 <= bbox.confidence <= 1.0)

def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """IoU 계산"""
    x1 = max(bbox1.x1, bbox2.x1)
    y1 = max(bbox1.y1, bbox2.y1)
    x2 = min(bbox1.x2, bbox2.x2)
    y2 = min(bbox1.y2, bbox2.y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    union = bbox1.area + bbox2.area - intersection
    
    return intersection / union if union > 0 else 0.0

def merge_environment_data(edge_data: EnvironmentData, 
                         server_data: EnvironmentData) -> EnvironmentData:
    """Edge와 Server 환경 데이터 병합"""
    merged = EnvironmentData(timestamp=edge_data.timestamp)
    
    for field in ['water_temp', 'air_temp', 'ph', 'ec', 'dissolved_oxygen', 
                 'humidity', 'light_intensity']:
        edge_value = getattr(edge_data, field)
        server_value = getattr(server_data, field) if server_data else None
        
        # Server 데이터 우선, 없으면 Edge 데이터 사용
        setattr(merged, field, server_value if server_value is not None else edge_value)
    
    return merged
