# common/yolo_detector.py - YOLO 탐지기 공통 모듈
import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO
from typing import List, Dict, Optional
import torch

from .data_types import YOLOResult, DetectedObject, BoundingBox, generate_frame_id

class YOLODetector:
    """YOLO 탐지기 공통 클래스"""
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        self.config = config
        self.device = device
        self.model = None
        self.model_name = config['model_path']
        self.load_model()
        
        # 클래스 매핑
        self.class_mapping = {
            0: 'fish',
            1: 'plant', 
            2: 'food'
        }
        
        logging.info(f"YOLO 탐지기 초기화: {self.model_name} on {device}")
    
    def load_model(self):
        """모델 로드"""
        try:
            self.model = YOLO(self.config['model_path'])
            
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            else:
                self.model.to('cpu')
                
            # 모델 워밍업
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_input, verbose=False)
            
            logging.info("YOLO 모델 로드 완료")
            
        except Exception as e:
            logging.error(f"YOLO 모델 로드 실패: {e}")
            raise
    
    def detect(self, frame: np.ndarray, frame_id: str = None) -> YOLOResult:
        """객체 탐지 수행"""
        if frame_id is None:
            frame_id = generate_frame_id()
            
        start_time = time.time()
        
        try:
            # YOLO 추론
            results = self.model(
                frame,
                conf=self.config['confidence'],
                iou=self.config['iou_threshold'],
                max_det=self.config['max_det'],
                verbose=False
            )[0]
            
            # 탐지 결과 파싱
            objects = []
            if results.boxes is not None:
                for i, box in enumerate(results.boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 클래스 필터링
                    if class_id not in self.class_mapping:
                        continue
                    
                    bbox = BoundingBox(
                        x1=coords[0], y1=coords[1],
                        x2=coords[2], y2=coords[3],
                        confidence=confidence
                    )
                    
                    # 추적 ID (있는 경우)
                    tracking_id = None
                    if hasattr(box, 'id') and box.id is not None:
                        tracking_id = int(box.id[0])
                    
                    obj = DetectedObject(
                        object_id=f"{self.class_mapping[class_id]}_{i}_{frame_id}",
                        class_name=self.class_mapping[class_id],
                        class_id=class_id,
                        bbox=bbox,
                        confidence=confidence,
                        tracking_id=tracking_id
                    )
                    
                    objects.append(obj)
            
            processing_time = time.time() - start_time
            
            # 결과 객체 생성
            yolo_result = YOLOResult(
                model_name=self.model_name,
                model_version=self.model.model.yaml.get('version', '1.0'),
                processing_time=processing_time,
                frame_id=frame_id,
                timestamp=datetime.now(),
                objects=objects,
                frame_shape=frame.shape,
                device_info=f"{self.device}_{torch.cuda.get_device_name() if self.device == 'cuda' else 'cpu'}"
            )
            
            return yolo_result
            
        except Exception as e:
            logging.error(f"YOLO 탐지 오류: {e}")
            
            # 오류 시 빈 결과 반환
            return YOLOResult(
                model_name=self.model_name,
                model_version="error",
                processing_time=time.time() - start_time,
                frame_id=frame_id,
                timestamp=datetime.now(),
                objects=[],
                frame_shape=frame.shape,
                device_info=f"{self.device}_error"
            )

